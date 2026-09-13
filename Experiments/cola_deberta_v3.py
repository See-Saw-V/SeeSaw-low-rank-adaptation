

!pip -q install -U transformers datasets scikit-learn accelerate sentencepiece

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
from sklearn.metrics import matthews_corrcoef, accuracy_score


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "microsoft/deberta-v3-base"

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32

NUM_EPOCHS = 25
MAX_LENGTH = 128

LORA_LR = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
GRAD_CLIP = 1.0

ALPHA = 32

TARGET_TOTAL_RANK = 827
MIN_RANK = 2
MAX_RANK = 64

CONTROLLER_START_EPOCH = 2
CONTROLLER_INTERVAL = 2

EMA_DECAY = 0.85
IMPORTANCE_GAP = 0.25

SEED = 42

BEST_PATH = "cola_deberta_qkvo_best.pt"
FINAL_PATH = "cola_deberta_qkvo_final.pt"


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# LOAD CoLA
# ============================================================

print("\nLoading CoLA...")

raw = load_dataset(
    "glue",
    "cola"
)

print(
    "Original columns:",
    raw["train"].column_names
)


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(batch):

    return tokenizer(
        batch["sentence"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


train_ds = raw["train"].map(
    tokenize,
    batched=True,
    remove_columns=[
        "sentence",
        "idx",
    ],
)

dev_ds = raw["validation"].map(
    tokenize,
    batched=True,
    remove_columns=[
        "sentence",
        "idx",
    ],
)


# ============================================================
# CoLA USES `label`
# ============================================================

train_ds = train_ds.rename_column(
    "label",
    "labels",
)

dev_ds = dev_ds.rename_column(
    "label",
    "labels",
)

train_ds.set_format("torch")
dev_ds.set_format("torch")


# ============================================================
# DATALOADERS
# ============================================================

collator = DataCollatorWithPadding(
    tokenizer=tokenizer
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

print(
    "Train examples:",
    len(train_ds)
)

print(
    "Validation examples:",
    len(dev_ds)
)

print(
    "Train batch size:",
    TRAIN_BATCH_SIZE
)

print(
    "Eval batch size:",
    EVAL_BATCH_SIZE
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading DeBERTa-v3-base...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)

model = model.float()
model = model.to(device)


# ============================================================
# ADAPTIVE LoRA
# ============================================================

class AdaptiveLoRA(nn.Module):

    def __init__(
        self,
        original_linear,
        rank,
        alpha,
    ):

        super().__init__()

        self.in_features = (
            original_linear.in_features
        )

        self.out_features = (
            original_linear.out_features
        )

        self.rank = rank
        self.alpha = alpha


        # Frozen original weight
        self.register_buffer(
            "weight_orig",
            original_linear.weight.detach()
            .clone()
            .float(),
        )


        # Frozen original bias
        if original_linear.bias is not None:

            self.register_buffer(
                "bias_orig",
                original_linear.bias.detach()
                .clone()
                .float(),
            )

        else:

            self.bias_orig = None


        # LoRA A
        self.A = nn.Parameter(
            torch.empty(
                rank,
                self.in_features,
                dtype=torch.float32,
            )
        )


        # LoRA B
        self.B = nn.Parameter(
            torch.zeros(
                self.out_features,
                rank,
                dtype=torch.float32,
            )
        )


        nn.init.normal_(
            self.A,
            std=0.02,
        )


    def forward(self, x):

        x = x.float()

        base = torch.nn.functional.linear(
            x,
            self.weight_orig,
            self.bias_orig,
        )

        update = torch.nn.functional.linear(
            x,
            self.A,
        )

        update = torch.nn.functional.linear(
            update,
            self.B,
        )

        update = update * (
            self.alpha /
            float(self.rank)
        )

        return base + update


    def resize_rank(self, new_rank):

        if new_rank == self.rank:
            return


        old_A = self.A.data
        old_B = self.B.data

        old_rank = self.rank


        new_A = torch.empty(
            new_rank,
            self.in_features,
            device=old_A.device,
            dtype=old_A.dtype,
        )

        new_B = torch.zeros(
            self.out_features,
            new_rank,
            device=old_B.device,
            dtype=old_B.dtype,
        )


        keep = min(
            old_rank,
            new_rank,
        )


        new_A[:keep].copy_(
            old_A[:keep]
        )

        new_B[:, :keep].copy_(
            old_B[:, :keep]
        )


        if new_rank > old_rank:

            nn.init.normal_(
                new_A[old_rank:],
                std=0.02,
            )


        self.A = nn.Parameter(
            new_A
        )

        self.B = nn.Parameter(
            new_B
        )

        self.rank = new_rank


# ============================================================
# CORRECT DeBERTa-v3 MODULE PATHS
#
# Q = layer.attention.self.query_proj
# K = layer.attention.self.key_proj
# V = layer.attention.self.value_proj
# O = layer.attention.output.dense
# ============================================================

target_modules = []

for layer_idx, layer in enumerate(
    model.deberta.encoder.layer
):

    # Query
    target_modules.append(
        (
            layer_idx,
            "query",
            layer.attention.self,
            "query_proj",
            layer.attention.self.query_proj,
        )
    )


    # Key
    target_modules.append(
        (
            layer_idx,
            "key",
            layer.attention.self,
            "key_proj",
            layer.attention.self.key_proj,
        )
    )


    # Value
    target_modules.append(
        (
            layer_idx,
            "value",
            layer.attention.self,
            "value_proj",
            layer.attention.self.value_proj,
        )
    )


    # Output
    target_modules.append(
        (
            layer_idx,
            "output",
            layer.attention.output,
            "dense",
            layer.attention.output.dense,
        )
    )


print(
    "\nLoRA modules:",
    len(target_modules)
)

assert len(target_modules) == 48


# ============================================================
# RANK ALLOCATION
#
# 11 × 18 + 37 × 17 = 827
# ============================================================

base_rank = (
    TARGET_TOTAL_RANK //
    len(target_modules)
)

extra = (
    TARGET_TOTAL_RANK %
    len(target_modules)
)

ranks = [
    base_rank +
    (1 if i < extra else 0)
    for i in range(
        len(target_modules)
    )
]

print(
    "Rank distribution:",
    {
        r: ranks.count(r)
        for r in sorted(set(ranks))
    }
)

print(
    "Total rank:",
    sum(ranks)
)

assert sum(ranks) == TARGET_TOTAL_RANK


# ============================================================
# INSERT LoRA
# ============================================================

adapters = {}

for i, (
    layer_idx,
    proj_name,
    parent,
    module_name,
    original_module,
) in enumerate(target_modules):

    adapter = AdaptiveLoRA(
        original_module,
        rank=ranks[i],
        alpha=ALPHA,
    )

    setattr(
        parent,
        module_name,
        adapter,
    )

    key = (
        f"layer{layer_idx}."
        f"{proj_name}"
    )

    adapters[key] = adapter


# ============================================================
# MOVE EVERYTHING TO DEVICE
# ============================================================

model = model.float().to(device)


# ============================================================
# FREEZE EVERYTHING
# ============================================================

for param in model.parameters():

    param.requires_grad = False


# ============================================================
# ENABLE LoRA
# ============================================================

for adapter in adapters.values():

    adapter.A.requires_grad = True
    adapter.B.requires_grad = True


# ============================================================
# ENABLE CLASSIFIER
# ============================================================

for param in model.classifier.parameters():

    param.requires_grad = True


# ============================================================
# DEVICE CHECK
# ============================================================

first_adapter = next(
    iter(adapters.values())
)

print("\nDevice check:")

print(
    "Model:",
    next(model.parameters()).dtype,
    next(model.parameters()).device,
)

print(
    "LoRA A:",
    first_adapter.A.dtype,
    first_adapter.A.device,
)

print(
    "LoRA B:",
    first_adapter.B.dtype,
    first_adapter.B.device,
)

print(
    "Original weight:",
    first_adapter.weight_orig.dtype,
    first_adapter.weight_orig.device,
)


# ============================================================
# TRAINABLE PARAMETERS
# ============================================================

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(
    "\nTrainable parameters:",
    f"{trainable_params:,}"
)


# ============================================================
# OPTIMIZER
# ============================================================

def make_optimizer():

    params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    return torch.optim.AdamW(
        params,
        lr=LORA_LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


optimizer = make_optimizer()


# ============================================================
# SCHEDULER
# ============================================================

total_steps = (
    NUM_EPOCHS *
    len(train_loader)
)

warmup_steps = int(
    WARMUP_RATIO *
    total_steps
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)


# ============================================================
# IMPORTANCE EMA
# ============================================================

importance_ema = {
    key: 0.0
    for key in adapters
}


# ============================================================
# TRAIN
# ============================================================

def train_epoch():

    model.train()

    total_loss = 0.0

    for batch in train_loader:

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }


        # CoLA labels are classification labels.
        batch["labels"] = (
            batch["labels"]
            .long()
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        outputs = model(
            **batch
        )

        loss = outputs.loss


        loss.backward()


        # ----------------------------------------
        # LoRA gradient importance
        # ----------------------------------------

        for key, adapter in adapters.items():

            if adapter.B.grad is not None:

                importance = (
                    adapter.B.grad
                    .detach()
                    .float()
                    .abs()
                    .mean()
                    .item()
                )

                importance_ema[key] = (
                    EMA_DECAY *
                    importance_ema[key]
                    +
                    (1.0 - EMA_DECAY) *
                    importance
                )


        # ----------------------------------------
        # Gradient clipping
        # ----------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )


        optimizer.step()
        scheduler.step()


        total_loss += loss.item()


    return (
        total_loss /
        len(train_loader)
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate():

    model.eval()

    predictions = []
    labels = []

    total_loss = 0.0


    for batch in dev_loader:

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }


        batch["labels"] = (
            batch["labels"]
            .long()
        )


        outputs = model(
            **batch
        )


        total_loss += (
            outputs.loss.item()
        )


        preds = (
            outputs.logits
            .argmax(dim=-1)
        )


        predictions.extend(
            preds.cpu()
            .numpy()
            .tolist()
        )


        labels.extend(
            batch["labels"]
            .cpu()
            .numpy()
            .tolist()
        )


    predictions = np.asarray(
        predictions
    )

    labels = np.asarray(
        labels
    )


    # ========================================================
    # MATTHEWS CORRELATION COEFFICIENT
    # ========================================================

    mcc = matthews_corrcoef(
        labels,
        predictions,
    )


    accuracy = accuracy_score(
        labels,
        predictions,
    )


    return (
        total_loss / len(dev_loader),
        mcc,
        accuracy,
    )


# ============================================================
# ADAPTIVE CONTROLLER
# ============================================================

def controller_step():

    global optimizer

    keys = list(
        adapters.keys()
    )


    scores = np.asarray(
        [
            importance_ema[k]
            for k in keys
        ],
        dtype=np.float64,
    )


    if np.all(scores <= 0):

        print(
            "Controller: "
            "no gradient signal."
        )

        return


    mean_score = scores.mean()

    std_score = (
        scores.std() +
        1e-12
    )


    normalized = (
        scores -
        mean_score
    ) / std_score


    receiver_idx = int(
        np.argmax(normalized)
    )

    donor_idx = int(
        np.argmin(normalized)
    )


    receiver = keys[
        receiver_idx
    ]

    donor = keys[
        donor_idx
    ]


    receiver_adapter = (
        adapters[receiver]
    )

    donor_adapter = (
        adapters[donor]
    )


    receiver_score = normalized[
        receiver_idx
    ]

    donor_score = normalized[
        donor_idx
    ]


    if (
        receiver_score -
        donor_score
        <
        IMPORTANCE_GAP
    ):

        print(
            "Controller: "
            "no rank transfer."
        )

        return


    if donor_adapter.rank <= MIN_RANK:

        print(
            "Controller: "
            "donor at minimum rank."
        )

        return


    if receiver_adapter.rank >= MAX_RANK:

        print(
            "Controller: "
            "receiver at maximum rank."
        )

        return


    old_receiver = (
        receiver_adapter.rank
    )

    old_donor = (
        donor_adapter.rank
    )


    receiver_adapter.resize_rank(
        old_receiver + 1
    )

    donor_adapter.resize_rank(
        old_donor - 1
    )


    optimizer = make_optimizer()


    print(
        f"Controller: "
        f"{donor} "
        f"{old_donor}"
        f"->{donor_adapter.rank} | "
        f"{receiver} "
        f"{old_receiver}"
        f"->{receiver_adapter.rank}"
    )


# ============================================================
# TRAINING LOOP
# ============================================================

best_mcc = -float("inf")


print("\n" + "=" * 70)
print("STARTING CoLA TRAINING")
print("=" * 70)


for epoch in range(
    1,
    NUM_EPOCHS + 1,
):

    train_loss = train_epoch()


    (
        eval_loss,
        mcc,
        accuracy,
    ) = evaluate()


    print(
        f"\nEpoch {epoch}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss : "
        f"{train_loss:.4f}"
    )

    print(
        f"Eval Loss  : "
        f"{eval_loss:.4f}"
    )

    print(
        f"MCC        : "
        f"{mcc:.4f}"
    )

    print(
        f"Accuracy   : "
        f"{accuracy:.4f}"
    )


    # ========================================================
    # SAVE BEST USING MCC
    # ========================================================

    if mcc > best_mcc:

        best_mcc = mcc

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "best_mcc":
                    best_mcc,
                "accuracy":
                    accuracy,
                "ranks": {
                    key: adapter.rank
                    for key, adapter
                    in adapters.items()
                },
            },
            BEST_PATH,
        )

        print(
            "✓ Best checkpoint saved"
        )


    # ========================================================
    # ADAPTIVE CONTROLLER
    # ========================================================

    if (
        epoch >=
        CONTROLLER_START_EPOCH
        and
        (
            (
                epoch -
                CONTROLLER_START_EPOCH
            )
            %
            CONTROLLER_INTERVAL
            == 0
        )
        and
        epoch < NUM_EPOCHS
    ):

        controller_step()


# ============================================================
# FINAL CHECKPOINT
# ============================================================

torch.save(
    {
        "epoch": NUM_EPOCHS,
        "model_state_dict":
            model.state_dict(),
        "best_mcc":
            best_mcc,
        "ranks": {
            key: adapter.rank
            for key, adapter
            in adapters.items()
        },
    },
    FINAL_PATH,
)


print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best MCC: "
    f"{best_mcc:.4f}"
)

print(
    f"Best checkpoint: "
    f"{BEST_PATH}"
)

print(
    f"Final checkpoint: "
    f"{FINAL_PATH}"
)
