
!pip -q install -U transformers datasets scikit-learn accelerate sentencepiece



import os
import gc
import copy
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
from sklearn.metrics import accuracy_score


# ============================================================
# 3. CONFIG
# ============================================================

MODEL_NAME = "microsoft/deberta-v3-base"

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32

NUM_EPOCHS = 5
MAX_LENGTH = 128

LR = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
GRAD_CLIP = 1.0

ALPHA = 32

# ------------------------------------------------------------
# SAME RANK BUDGET AS YOUR SCRIPT
# ------------------------------------------------------------

TARGET_TOTAL_RANK = 827

MIN_RANK = 2
MAX_RANK = 64

# ------------------------------------------------------------
# SEE-SAW CONTROLLER
# ------------------------------------------------------------

CONTROLLER_START_EPOCH = 2
CONTROLLER_INTERVAL = 1

# ------------------------------------------------------------
# EMA IMPORTANCE
# ------------------------------------------------------------

EMA_DECAY = 0.85

# ------------------------------------------------------------
# Z-SCORE GAP
# ------------------------------------------------------------

IMPORTANCE_GAP = 0.25

SEED = 42

BEST_PATH = "sst2_deberta_v3_seesaw_zscore_best.pt"
FINAL_PATH = "sst2_deberta_v3_seesaw_zscore_final.pt"


# ============================================================
# 4. SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 5. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 75)
print("Device:", device)
print("Model :", MODEL_NAME)
print("Task  : SST-2")
print("Method: See-Saw Dynamic LoRA + Z-score")
print("=" * 75)


# ============================================================
# 6. DATASET
# ============================================================

print("\nLoading SST-2...")

raw = load_dataset(
    "glue",
    "sst2"
)

print(
    "Train examples:",
    len(raw["train"])
)

print(
    "Validation examples:",
    len(raw["validation"])
)


# ============================================================
# 7. TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize(batch):

    return tokenizer(
        batch["sentence"],
        truncation=True,
        max_length=MAX_LENGTH
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
# 8. LABEL -> labels
# ============================================================

train_ds = train_ds.rename_column(
    "label",
    "labels"
)

dev_ds = dev_ds.rename_column(
    "label",
    "labels"
)


train_ds.set_format("torch")
dev_ds.set_format("torch")


# ============================================================
# 9. DATALOADERS
# ============================================================

collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

train_loader = DataLoader(
    train_ds,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    collate_fn=collator
)

dev_loader = DataLoader(
    dev_ds,
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    collate_fn=collator
)


# ============================================================
# 10. MODEL
# ============================================================

print("\nLoading DeBERTa-v3-base...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# Explicit FP32
model = model.float()
model = model.to(device)


# ============================================================
# 11. ADAPTIVE LoRA
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

        # ----------------------------------------------------
        # Frozen original weight
        # ----------------------------------------------------

        self.register_buffer(
            "weight_orig",
            original_linear.weight.detach()
            .clone()
            .float()
        )

        # ----------------------------------------------------
        # Frozen original bias
        # ----------------------------------------------------

        if original_linear.bias is not None:

            self.register_buffer(
                "bias_orig",
                original_linear.bias.detach()
                .clone()
                .float()
            )

        else:

            self.bias_orig = None

        # ----------------------------------------------------
        # LoRA A
        # ----------------------------------------------------

        self.A = nn.Parameter(
            torch.empty(
                rank,
                self.in_features,
                dtype=torch.float32
            )
        )

        # ----------------------------------------------------
        # LoRA B
        # ----------------------------------------------------

        self.B = nn.Parameter(
            torch.zeros(
                self.out_features,
                rank,
                dtype=torch.float32
            )
        )

        nn.init.normal_(
            self.A,
            std=0.02
        )


    def forward(self, x):

        # Everything FP32
        x = x.float()

        # ----------------------------------------------------
        # Frozen base
        # ----------------------------------------------------

        base = torch.nn.functional.linear(
            x,
            self.weight_orig,
            self.bias_orig
        )

        # ----------------------------------------------------
        # LoRA update
        # ----------------------------------------------------

        update = torch.nn.functional.linear(
            x,
            self.A
        )

        update = torch.nn.functional.linear(
            update,
            self.B
        )

        update = update * (
            self.alpha /
            float(self.rank)
        )

        return base + update


    def resize_rank(
        self,
        new_rank
    ):

        if new_rank == self.rank:
            return

        old_A = self.A.data
        old_B = self.B.data

        old_rank = self.rank

        # ----------------------------------------------------
        # New A
        # ----------------------------------------------------

        new_A = torch.empty(
            new_rank,
            self.in_features,
            device=old_A.device,
            dtype=old_A.dtype
        )

        # ----------------------------------------------------
        # New B
        # ----------------------------------------------------

        new_B = torch.zeros(
            self.out_features,
            new_rank,
            device=old_B.device,
            dtype=old_B.dtype
        )

        keep = min(
            old_rank,
            new_rank
        )

        with torch.no_grad():

            new_A[:keep].copy_(
                old_A[:keep]
            )

            new_B[:, :keep].copy_(
                old_B[:, :keep]
            )

            if new_rank > old_rank:

                nn.init.normal_(
                    new_A[old_rank:],
                    std=0.02
                )

        self.A = nn.Parameter(
            new_A,
            requires_grad=True
        )

        self.B = nn.Parameter(
            new_B,
            requires_grad=True
        )

        self.rank = new_rank


# ============================================================
# 12. DeBERTa-v3 Q/K/V/O MODULES
#
# Q = query_proj
# K = key_proj
# V = value_proj
# O = dense
# ============================================================

target_modules = []

for layer_idx, layer in enumerate(
    model.deberta.encoder.layer
):

    # --------------------------------------------------------
    # Query
    # --------------------------------------------------------

    target_modules.append(
        (
            layer_idx,
            "query",
            layer.attention.self,
            "query_proj",
            layer.attention.self.query_proj,
        )
    )

    # --------------------------------------------------------
    # Key
    # --------------------------------------------------------

    target_modules.append(
        (
            layer_idx,
            "key",
            layer.attention.self,
            "key_proj",
            layer.attention.self.key_proj,
        )
    )

    # --------------------------------------------------------
    # Value
    # --------------------------------------------------------

    target_modules.append(
        (
            layer_idx,
            "value",
            layer.attention.self,
            "value_proj",
            layer.attention.self.value_proj,
        )
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

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
# 13. RANK ALLOCATION
#
# 827 / 48
#
# 11 modules -> rank 18
# 37 modules -> rank 17
#
# 11*18 + 37*17 = 827
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
    "\nRank distribution:",
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
# 14. INSERT LoRA
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
        adapter
    )

    key = (
        f"layer{layer_idx}."
        f"{proj_name}"
    )

    adapters[key] = adapter


# ============================================================
# 15. MOVE MODEL TO DEVICE
# ============================================================

model = model.float().to(device)


# ============================================================
# 16. FREEZE EVERYTHING
# ============================================================

for param in model.parameters():
    param.requires_grad = False


# ============================================================
# 17. ENABLE LoRA
# ============================================================

for adapter in adapters.values():

    adapter.A.requires_grad = True
    adapter.B.requires_grad = True


# ============================================================
# 18. ENABLE SST-2 CLASSIFIER
# ============================================================

for param in model.classifier.parameters():
    param.requires_grad = True


# ============================================================
# 19. TRAINABLE PARAMETER COUNT
# ============================================================

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_percent = (
    100.0 *
    trainable_params /
    total_params
)

print("\n" + "=" * 75)
print("PARAMETER COUNT")
print("=" * 75)

print(
    "Trainable parameters:",
    f"{trainable_params:,}"
)

print(
    "Total parameters:",
    f"{total_params:,}"
)

print(
    "Trainable percentage:",
    f"{trainable_percent:.4f}%"
)

print(
    "Target rank:",
    TARGET_TOTAL_RANK
)


# ============================================================
# 20. OPTIMIZER
# ============================================================

def make_optimizer():

    params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    return torch.optim.AdamW(
        params,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


optimizer = make_optimizer()


# ============================================================
# 21. SCHEDULER
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
# 22. IMPORTANCE EMA
# ============================================================

importance_ema = {
    key: 0.0
    for key in adapters
}


# ============================================================
# 23. TRAIN EPOCH
# ============================================================

def train_epoch():

    model.train()

    total_loss = 0.0

    for batch in train_loader:

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        # ----------------------------------------------------
        # SST-2 classification labels
        # ----------------------------------------------------

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

        loss = outputs.loss.float()

        loss.backward()

        # ----------------------------------------------------
        # Importance:
        #
        # mean(abs(grad(B)))
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return (
        total_loss /
        len(train_loader)
    )


# ============================================================
# 24. EVALUATION
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

        loss = outputs.loss.float()

        total_loss += loss.item()

        preds = torch.argmax(
            outputs.logits,
            dim=-1
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

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return (
        total_loss /
        len(dev_loader),
        accuracy
    )


# ============================================================
# 25. Z-SCORE SEE-SAW CONTROLLER
#
# Raw score:
#     EMA(mean(abs(grad(B))))
#
# Z-score:
#     (score - mean(score)) / std(score)
#
# Receiver = maximum Z
# Donor    = minimum Z
#
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
        dtype=np.float64
    )

    # --------------------------------------------------------
    # No gradient signal
    # --------------------------------------------------------

    if np.all(scores <= 0):

        print(
            "Controller: no gradient signal."
        )

        return False

    # --------------------------------------------------------
    # Z-score normalization
    # --------------------------------------------------------

    mean_score = scores.mean()

    std_score = (
        scores.std() +
        1e-12
    )

    normalized = (
        scores -
        mean_score
    ) / std_score

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print("\nController diagnostics:")

    print(
        f"  Raw mean = "
        f"{mean_score:.8f}"
    )

    print(
        f"  Raw min  = "
        f"{scores.min():.8f}"
    )

    print(
        f"  Raw max  = "
        f"{scores.max():.8f}"
    )

    print(
        f"  Z mean   = "
        f"{normalized.mean():.6f}"
    )

    print(
        f"  Z std    = "
        f"{normalized.std():.6f}"
    )

    # --------------------------------------------------------
    # Receiver / donor
    # --------------------------------------------------------

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

    gap = (
        receiver_score -
        donor_score
    )

    print(
        f"  Receiver = {receiver} "
        f"(z={receiver_score:.4f}, "
        f"rank={receiver_adapter.rank})"
    )

    print(
        f"  Donor    = {donor} "
        f"(z={donor_score:.4f}, "
        f"rank={donor_adapter.rank})"
    )

    print(
        f"  Gap      = {gap:.4f}"
    )

    # --------------------------------------------------------
    # Gap threshold
    # --------------------------------------------------------

    if gap < IMPORTANCE_GAP:

        print(
            "Controller: no rank transfer."
        )

        return False

    # --------------------------------------------------------
    # Minimum donor rank
    # --------------------------------------------------------

    if donor_adapter.rank <= MIN_RANK:

        print(
            "Controller: donor at minimum rank."
        )

        return False

    # --------------------------------------------------------
    # Maximum receiver rank
    # --------------------------------------------------------

    if receiver_adapter.rank >= MAX_RANK:

        print(
            "Controller: receiver at maximum rank."
        )

        return False

    # --------------------------------------------------------
    # Old ranks
    # --------------------------------------------------------

    old_receiver = (
        receiver_adapter.rank
    )

    old_donor = (
        donor_adapter.rank
    )

    # --------------------------------------------------------
    # Transfer exactly ONE rank
    # --------------------------------------------------------

    receiver_adapter.resize_rank(
        old_receiver + 1
    )

    donor_adapter.resize_rank(
        old_donor - 1
    )

    # --------------------------------------------------------
    # Verify exact global rank budget
    # --------------------------------------------------------

    current_total_rank = sum(
        adapter.rank
        for adapter in adapters.values()
    )

    assert (
        current_total_rank ==
        TARGET_TOTAL_RANK
    )

    # --------------------------------------------------------
    # Rebuild optimizer
    # --------------------------------------------------------

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

    print(
        f"Total rank: "
        f"{current_total_rank}"
    )

    print(
        "Optimizer rebuilt after rank transfer."
    )

    return True


# ============================================================
# 26. TRAINING LOOP
# ============================================================

best_accuracy = -float("inf")
best_epoch = None
best_ranks = None
best_state = None

print("\n" + "=" * 75)
print("STARTING SST-2 TRAINING")
print("=" * 75)


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_loss = train_epoch()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    eval_loss, accuracy = evaluate()

    print(
        f"\nEpoch {epoch:02d}/{NUM_EPOCHS}"
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
        f"Accuracy   : "
        f"{accuracy:.4f}"
    )

    print(
        f"Total Rank : "
        f"{sum(a.rank for a in adapters.values())}"
    )

    # --------------------------------------------------------
    # BEST CHECKPOINT
    #
    # Save before adaptive resizing.
    # --------------------------------------------------------

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_epoch = epoch

        best_state = copy.deepcopy(
            model.state_dict()
        )

        best_ranks = {
            key: adapter.rank
            for key, adapter
            in adapters.items()
        }

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": best_state,
                "best_accuracy": best_accuracy,
                "ranks": best_ranks,
            },
            BEST_PATH
        )

        print(
            f"  ★ NEW BEST "
            f"Accuracy = "
            f"{best_accuracy:.4f}"
        )

    # --------------------------------------------------------
    # SEE-SAW CONTROLLER
    #
    # Epochs:
    # 2, 3, 4, ..., 24
    # because INTERVAL = 1
    # --------------------------------------------------------

    if (
        epoch >= CONTROLLER_START_EPOCH
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

        print(
            "\n  >>> SEE-SAW CONTROLLER"
        )

        controller_step()

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 27. RESTORE BEST CHECKPOINT
# ============================================================

print("\n" + "=" * 75)
print("RESTORING BEST CHECKPOINT")
print("=" * 75)

if best_state is not None:

    model.load_state_dict(
        best_state
    )

    # --------------------------------------------------------
    # Restore rank allocation as well
    # --------------------------------------------------------

    if best_ranks is not None:

        for key, best_rank in best_ranks.items():

            current_rank = (
                adapters[key].rank
            )

            if current_rank != best_rank:

                adapters[key].resize_rank(
                    best_rank
                )

        # Exact invariant
        restored_total = sum(
            adapter.rank
            for adapter in adapters.values()
        )

        assert (
            restored_total ==
            TARGET_TOTAL_RANK
        )


# ============================================================
# 28. FINAL BEST EVALUATION
# ============================================================

final_eval_loss, final_accuracy = evaluate()


# ============================================================
# 29. FINAL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("FINAL RESULTS")
print("=" * 75)

print(
    f"Best Epoch:       "
    f"{best_epoch}"
)

print(
    f"Best Accuracy:    "
    f"{best_accuracy:.4f}"
)

print(
    f"Restored Accuracy:"
    f" {final_accuracy:.4f}"
)

print(
    f"Final Eval Loss:  "
    f"{final_eval_loss:.4f}"
)

print(
    f"Total Rank:       "
    f"{sum(a.rank for a in adapters.values())}"
)


# ============================================================
# 30. FINAL RANK ALLOCATION
# ============================================================

print("\n" + "=" * 75)
print("BEST CHECKPOINT RANK ALLOCATION")
print("=" * 75)

for layer_idx in range(12):

    print(
        f"\nLayer {layer_idx}:"
    )

    for proj in [
        "query",
        "key",
        "value",
        "output"
    ]:

        key = (
            f"layer{layer_idx}."
            f"{proj}"
        )

        print(
            f"  {proj:8s}: "
            f"{adapters[key].rank}"
        )


# ============================================================
# 31. SAVE FINAL CHECKPOINT
# ============================================================

torch.save(
    {
        "epoch": best_epoch,
        "model_state_dict":
            model.state_dict(),
        "best_accuracy":
            best_accuracy,
        "ranks": {
            key: adapter.rank
            for key, adapter
            in adapters.items()
        },
        "target_total_rank":
            TARGET_TOTAL_RANK,
        "trainable_parameters":
            trainable_params,
    },
    FINAL_PATH
)


# ============================================================
# 32. FINAL ASSERTIONS
# ============================================================

assert len(adapters) == 48

assert (
    sum(
        adapter.rank
        for adapter in adapters.values()
    )
    ==
    TARGET_TOTAL_RANK
)

assert all(
    MIN_RANK <= adapter.rank <= MAX_RANK
    for adapter in adapters.values()
)


# ============================================================
# 33. DONE
# ============================================================

print("\n" + "=" * 75)
print("TRAINING COMPLETE")
print("=" * 75)

print(
    f"Best SST-2 Accuracy: "
    f"{best_accuracy:.4f}"
)

print(
    f"Best Epoch: "
    f"{best_epoch}"
)

print(
    f"Total Rank: "
    f"{sum(a.rank for a in adapters.values())}"
)

print(
    f"Trainable Parameters: "
    f"{trainable_params:,}"
)

print(
    f"Best checkpoint: "
    f"{BEST_PATH}"
)

print(
    f"Final checkpoint: "
    f"{FINAL_PATH}"
)

print("=" * 75)
