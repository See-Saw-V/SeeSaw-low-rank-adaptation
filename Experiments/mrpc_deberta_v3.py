

!pip -q install -U transformers datasets accelerate sentencepiece scikit-learn scipy

import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score


# ================================================================
# CONFIG
# ================================================================

MODEL_NAME = "microsoft/deberta-v3-base"

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32

NUM_EPOCHS = 10

MAX_LENGTH = 128

LORA_LR = 1e-4
CLASSIFIER_LR = 2e-4

WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
GRAD_CLIP = 1.0

ALPHA = 32

TARGET_TOTAL_RANK = 827
MIN_RANK = 2
MAX_RANK = 64

# Adaptive controller
CONTROLLER_START_EPOCH = 2
CONTROLLER_INTERVAL = 1

EMA_DECAY = 0.85
RANK_TRANSFER_GAP = 0.25

SEED = 42

BEST_PATH = "mrpc_deberta_qkvo_adaptive_best.pt"
FINAL_PATH = "mrpc_deberta_qkvo_adaptive_final.pt"


# ================================================================
# SEED
# ================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ================================================================
# DATASET
# ================================================================

print("\nLoading MRPC...")

dataset = load_dataset("glue", "mrpc")

train_ds = dataset["train"]
dev_ds = dataset["validation"]

print("Train examples:", len(train_ds))
print("Validation examples:", len(dev_ds))


# ================================================================
# TOKENIZER
# ================================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_fn(batch):
    return tokenizer(
        batch["sentence1"],
        batch["sentence2"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


train_ds = train_ds.map(tokenize_fn, batched=True)
dev_ds = dev_ds.map(tokenize_fn, batched=True)

# MRPC label column is "label"
train_ds = train_ds.rename_column("label", "labels")
dev_ds = dev_ds.rename_column("label", "labels")

train_ds = train_ds.remove_columns(["sentence1", "sentence2", "idx"])
dev_ds = dev_ds.remove_columns(["sentence1", "sentence2", "idx"])

train_ds.set_format("torch")
dev_ds.set_format("torch")


# ================================================================
# DATALOADERS
# ================================================================

collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt",
)

train_loader = DataLoader(
    train_ds,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    collate_fn=collator,
)

dev_loader = DataLoader(
    dev_ds,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    collate_fn=collator,
)


# ================================================================
# MODEL
# ================================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

# Keep FP32
model = model.float()


# ================================================================
# ADAPTIVE LoRA
# ================================================================

class AdaptiveLoRA(nn.Module):

    def __init__(
        self,
        weight,
        bias,
        rank,
        alpha=32,
    ):
        super().__init__()

        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

        self.alpha = alpha
        self.rank = rank

        # Frozen original weight
        self.register_buffer(
            "original_weight",
            weight.detach().clone().float()
        )

        if bias is not None:
            self.register_buffer(
                "original_bias",
                bias.detach().clone().float()
            )
        else:
            self.original_bias = None

        # LoRA parameters
        self.A = nn.Parameter(
            torch.empty(rank, self.in_features)
        )

        self.B = nn.Parameter(
            torch.empty(self.out_features, rank)
        )

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

        self.importance_ema = 0.0

    @property
    def scaling(self):
        return self.alpha / self.rank

    def forward(self, x):

        base = torch.nn.functional.linear(
            x,
            self.original_weight,
            self.original_bias,
        )

        lora = torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.A),
            self.B,
        )

        return base + self.scaling * lora

    def resize_rank(self, new_rank):

        old_rank = self.rank

        new_rank = int(new_rank)

        if new_rank == old_rank:
            return

        device = self.A.device
        dtype = self.A.dtype

        new_A = torch.empty(
            new_rank,
            self.in_features,
            device=device,
            dtype=dtype,
        )

        new_B = torch.empty(
            self.out_features,
            new_rank,
            device=device,
            dtype=dtype,
        )

        nn.init.kaiming_uniform_(
            new_A,
            a=math.sqrt(5),
        )

        nn.init.zeros_(new_B)

        keep = min(old_rank, new_rank)

        with torch.no_grad():

            new_A[:keep].copy_(self.A[:keep])
            new_B[:, :keep].copy_(self.B[:, :keep])

        self.A = nn.Parameter(new_A)
        self.B = nn.Parameter(new_B)

        self.rank = new_rank


# ================================================================
# REPLACE PROJECTIONS
# ================================================================

def replace_linear_with_lora(parent, child_name, rank):

    original = getattr(parent, child_name)

    if not isinstance(original, nn.Linear):
        raise TypeError(
            f"{child_name} is not nn.Linear: {type(original)}"
        )

    adapter = AdaptiveLoRA(
        weight=original.weight,
        bias=original.bias,
        rank=rank,
        alpha=ALPHA,
    )

    setattr(parent, child_name, adapter)

    return adapter


# ================================================================
# INITIAL RANK ALLOCATION
# 37 modules × 17 + 11 modules × 18 = 827
# ================================================================

NUM_LAYERS = 12
NUM_MODULES = NUM_LAYERS * 4

base_rank = TARGET_TOTAL_RANK // NUM_MODULES
remainder = TARGET_TOTAL_RANK % NUM_MODULES

ranks = [base_rank] * NUM_MODULES

for i in range(remainder):
    ranks[i] += 1

assert sum(ranks) == TARGET_TOTAL_RANK

print("\nInitial rank allocation:")
print("Rank distribution:", {
    r: ranks.count(r) for r in sorted(set(ranks))
})
print("Total rank:", sum(ranks))


# ================================================================
# APPLY Q/K/V/O LoRA
# ================================================================

lora_modules = {}

module_idx = 0

for layer_idx in range(NUM_LAYERS):

    layer = model.deberta.encoder.layer[layer_idx]

    # Correct DeBERTa-v3 projection paths
    projections = [
        ("query", layer.attention.self, "query_proj"),
        ("key", layer.attention.self, "key_proj"),
        ("value", layer.attention.self, "value_proj"),
        ("output", layer.attention.output, "dense"),
    ]

    for logical_name, parent, child_name in projections:

        rank = ranks[module_idx]

        adapter = replace_linear_with_lora(
            parent,
            child_name,
            rank,
        )

        module_name = f"layer{layer_idx}.{logical_name}"

        lora_modules[module_name] = adapter

        module_idx += 1


print("\nLoRA modules:", len(lora_modules))

rank_distribution = {}

for module in lora_modules.values():
    rank_distribution[module.rank] = (
        rank_distribution.get(module.rank, 0) + 1
    )

print("Rank distribution:", rank_distribution)
print(
    "Total rank:",
    sum(m.rank for m in lora_modules.values())
)


# ================================================================
# FREEZE BASE MODEL
# ================================================================

for name, param in model.named_parameters():

    param.requires_grad = False

# Enable LoRA A/B
for module in lora_modules.values():

    module.A.requires_grad = True
    module.B.requires_grad = True

# Enable classifier
for param in model.classifier.parameters():
    param.requires_grad = True


# ================================================================
# MOVE EVERYTHING TO GPU
# ================================================================

model = model.float().to(device)


# ================================================================
# DEVICE / DTYPE CHECK
# ================================================================

first_lora = next(iter(lora_modules.values()))

print("\nDevice check:")
print(
    "Model:",
    model.deberta.embeddings.word_embeddings.weight.dtype,
    model.deberta.embeddings.word_embeddings.weight.device,
)

print(
    "LoRA A:",
    first_lora.A.dtype,
    first_lora.A.device,
)

print(
    "LoRA B:",
    first_lora.B.dtype,
    first_lora.B.device,
)

print(
    "Original weight:",
    first_lora.original_weight.dtype,
    first_lora.original_weight.device,
)


# ================================================================
# TRAINABLE PARAMETER COUNT
# ================================================================

trainable_params = [
    p for p in model.parameters()
    if p.requires_grad
]

num_trainable = sum(
    p.numel() for p in trainable_params
)

print("\nTrainable parameters:", f"{num_trainable:,}")


# ================================================================
# OPTIMIZER
# ================================================================

lora_params = []
classifier_params = []

for module in lora_modules.values():

    lora_params.extend([
        module.A,
        module.B,
    ])

classifier_params = list(
    model.classifier.parameters()
)

optimizer = torch.optim.AdamW(
    [
        {
            "params": lora_params,
            "lr": LORA_LR,
        },
        {
            "params": classifier_params,
            "lr": CLASSIFIER_LR,
        },
    ],
    weight_decay=WEIGHT_DECAY,
)


# ================================================================
# SCHEDULER
# ================================================================

num_training_steps = (
    len(train_loader) * NUM_EPOCHS
)

num_warmup_steps = int(
    WARMUP_RATIO * num_training_steps
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)


# ================================================================
# IMPORTANCE UPDATE
# ================================================================

def update_importance():

    for module in lora_modules.values():

        if module.B.grad is None:
            continue

        grad_score = (
            module.B.grad.detach()
            .abs()
            .mean()
            .item()
        )

        module.importance_ema = (
            EMA_DECAY * module.importance_ema
            + (1.0 - EMA_DECAY) * grad_score
        )


# ================================================================
# ADAPTIVE CONTROLLER
# ================================================================

def adaptive_controller():

    scores = {
        name: module.importance_ema
        for name, module in lora_modules.items()
    }

    if not scores:
        return False

    values = np.array(
        list(scores.values()),
        dtype=np.float64,
    )

    mean_score = values.mean()
    std_score = values.std() + 1e-12

    normalized = {
        name: (score - mean_score) / std_score
        for name, score in scores.items()
    }

    receiver_name = max(
        normalized,
        key=normalized.get,
    )

    donor_name = min(
        normalized,
        key=normalized.get,
    )

    receiver = lora_modules[receiver_name]
    donor = lora_modules[donor_name]

    receiver_score = normalized[receiver_name]
    donor_score = normalized[donor_name]

    gap = receiver_score - donor_score

    if gap < RANK_TRANSFER_GAP:
        return False

    if donor.rank <= MIN_RANK:
        return False

    if receiver.rank >= MAX_RANK:
        return False

    old_donor_rank = donor.rank
    old_receiver_rank = receiver.rank

    donor.resize_rank(
        donor.rank - 1
    )

    receiver.resize_rank(
        receiver.rank + 1
    )

    print(
        f"Controller: "
        f"{donor_name} {old_donor_rank}->{donor.rank} | "
        f"{receiver_name} {old_receiver_rank}->{receiver.rank}"
    )

    return True


# ================================================================
# REBUILD OPTIMIZER AFTER RANK CHANGE
# ================================================================

def rebuild_optimizer():

    global optimizer
    global scheduler

    lora_params = []

    for module in lora_modules.values():

        lora_params.extend([
            module.A,
            module.B,
        ])

    classifier_params = list(
        model.classifier.parameters()
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_params,
                "lr": LORA_LR,
            },
            {
                "params": classifier_params,
                "lr": CLASSIFIER_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    # Keep scheduler length consistent
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


# ================================================================
# EVALUATION
# ================================================================

@torch.no_grad()
def evaluate():

    model.eval()

    losses = []
    predictions = []
    labels_all = []

    for batch in dev_loader:

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        outputs = model(**batch)

        loss = outputs.loss

        losses.append(
            loss.item()
        )

        preds = torch.argmax(
            outputs.logits,
            dim=-1,
        )

        predictions.extend(
            preds.cpu().numpy()
        )

        labels_all.extend(
            batch["labels"].cpu().numpy()
        )

    eval_loss = np.mean(losses)

    accuracy = accuracy_score(
        labels_all,
        predictions,
    )

    f1 = f1_score(
        labels_all,
        predictions,
        zero_division=0,
    )

    return eval_loss, f1, accuracy


# ================================================================
# TRAINING
# ================================================================

best_f1 = -float("inf")

print("\n" + "=" * 70)
print("STARTING MRPC TRAINING")
print("=" * 70)

for epoch in range(1, NUM_EPOCHS + 1):

    model.train()

    train_losses = []

    for batch in train_loader:

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(**batch)

        loss = outputs.loss

        loss.backward()

        # Importance before gradients are cleared
        update_importance()

        torch.nn.utils.clip_grad_norm_(
            trainable_params,
            GRAD_CLIP,
        )

        optimizer.step()
        scheduler.step()

        train_losses.append(
            loss.item()
        )

    train_loss = np.mean(train_losses)

    eval_loss, f1, accuracy = evaluate()

    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    print(f"Train Loss : {train_loss:.4f}")
    print(f"Eval Loss  : {eval_loss:.4f}")
    print(f"F1         : {f1:.4f}")
    print(f"Accuracy   : {accuracy:.4f}")

    # Best checkpoint based on F1
    if f1 > best_f1:

        best_f1 = f1

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_f1": best_f1,
                "eval_loss": eval_loss,
                "accuracy": accuracy,
            },
            BEST_PATH,
        )

        print("✓ Best checkpoint saved")

    # Adaptive controller
    if (
        epoch >= CONTROLLER_START_EPOCH
        and (epoch - CONTROLLER_START_EPOCH)
        % CONTROLLER_INTERVAL == 0
    ):

        changed = adaptive_controller()

        if changed:
            rebuild_optimizer()


# ================================================================
# FINAL CHECKPOINT
# ================================================================

torch.save(
    {
        "epoch": NUM_EPOCHS,
        "model_state_dict": model.state_dict(),
        "best_f1": best_f1,
    },
    FINAL_PATH,
)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print("Best F1:", f"{best_f1:.4f}")
print("Best checkpoint:", BEST_PATH)
print("Final checkpoint:", FINAL_PATH)
