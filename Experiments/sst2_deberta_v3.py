

!pip -q install -U transformers datasets sentencepiece scikit-learn

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
from sklearn.metrics import accuracy_score


# ============================================================
# CONFIG
# ============================================================

SEED = 42

MODEL_NAME = "microsoft/deberta-v3-base"

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8

NUM_EPOCHS = 5
MAX_LENGTH = 128

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

BETAS = (0.9, 0.999)
EPS = 1e-8

GRAD_CLIP = 1.0

ALPHA = 32

# Total LoRA rank budget
TARGET_TOTAL_RANK = 827

MIN_RANK = 2
MAX_RANK = 64

CONTROLLER_START_EPOCH = 2
CONTROLLER_INTERVAL = 2

IMPORTANCE_EMA = 0.85
IMPORTANCE_GAP = 0.25


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Device: CPU")

print("=" * 70)


# ============================================================
# LOAD SST-2
# ============================================================

print("\nLoading SST-2...")

dataset = load_dataset("glue", "sst2")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize_fn(batch):

    return tokenizer(
        batch["sentence"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


tokenized = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=["sentence", "idx"],
)

tokenized = tokenized.rename_column(
    "label",
    "labels",
)


data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8
    if torch.cuda.is_available()
    else None,
)


train_loader = DataLoader(
    tokenized["train"],
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator,
    pin_memory=torch.cuda.is_available(),
)


val_loader = DataLoader(
    tokenized["validation"],
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator,
    pin_memory=torch.cuda.is_available(),
)


print("Train:", len(tokenized["train"]))
print("Validation:", len(tokenized["validation"]))
print("Train batch:", TRAIN_BATCH_SIZE)
print("Eval batch:", EVAL_BATCH_SIZE)
print("Steps/epoch:", len(train_loader))


# ============================================================
# MODEL
# ============================================================

print("\nLoading DeBERTa-v3-base...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
).float().to(device)


# ============================================================
# ADAPTIVE LoRA LINEAR
# ============================================================

class AdaptiveLoRALinear(nn.Module):

    def __init__(
        self,
        original_linear,
        rank,
        alpha,
        name,
    ):

        super().__init__()

        self.name = name

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        self.alpha = alpha
        self.rank = rank

        target_device = original_linear.weight.device


        # Frozen base weight
        self.weight = nn.Parameter(
            original_linear.weight
            .detach()
            .clone()
            .float()
            .to(target_device),
            requires_grad=False,
        )


        # Frozen base bias
        self.bias = None

        if original_linear.bias is not None:

            self.bias = nn.Parameter(
                original_linear.bias
                .detach()
                .clone()
                .float()
                .to(target_device),
                requires_grad=False,
            )


        # LoRA A
        self.A = nn.Parameter(
            torch.empty(
                rank,
                self.in_features,
                device=target_device,
                dtype=torch.float32,
            )
        )


        # LoRA B
        self.B = nn.Parameter(
            torch.zeros(
                self.out_features,
                rank,
                device=target_device,
                dtype=torch.float32,
            )
        )


        nn.init.normal_(
            self.A,
            mean=0.0,
            std=0.02,
        )


        self.scale = self.alpha / self.rank

        self.importance_ema = 0.0


    def forward(self, x):

        x = x.float()


        base = torch.nn.functional.linear(
            x,
            self.weight,
            self.bias,
        )


        h = torch.nn.functional.linear(
            x,
            self.A,
        )


        lora = torch.nn.functional.linear(
            h,
            self.B,
        )


        return base + self.scale * lora


    @torch.no_grad()
    def update_importance(self):

        if self.B.grad is None:
            return


        importance = (
            self.B.grad
            .detach()
            .float()
            .abs()
            .mean()
            .item()
        )


        if not math.isfinite(importance):
            return


        self.importance_ema = (
            IMPORTANCE_EMA
            * self.importance_ema
            +
            (1 - IMPORTANCE_EMA)
            * importance
        )


    @torch.no_grad()
    def resize_rank(self, new_rank):

        old_rank = self.rank


        if new_rank == old_rank:
            return


        device_ = self.A.device


        new_A = torch.empty(
            new_rank,
            self.in_features,
            device=device_,
            dtype=torch.float32,
        )


        new_B = torch.zeros(
            self.out_features,
            new_rank,
            device=device_,
            dtype=torch.float32,
        )


        keep = min(
            old_rank,
            new_rank,
        )


        new_A[:keep].copy_(
            self.A[:keep]
        )


        new_B[:, :keep].copy_(
            self.B[:, :keep]
        )


        if new_rank > old_rank:

            nn.init.normal_(
                new_A[old_rank:],
                mean=0.0,
                std=0.02,
            )


        self.A = nn.Parameter(
            new_A,
            requires_grad=True,
        )


        self.B = nn.Parameter(
            new_B,
            requires_grad=True,
        )


        self.rank = new_rank

        self.scale = self.alpha / self.rank


# ============================================================
# INSERT Q/K/V/O LoRA
# ============================================================

lora_modules = []


for layer_idx, layer in enumerate(
    model.deberta.encoder.layer
):

    # --------------------------------------------------------
    # Query / Key / Value
    # --------------------------------------------------------

    for proj_name in [
        "query_proj",
        "key_proj",
        "value_proj",
    ]:

        original = getattr(
            layer.attention.self,
            proj_name,
        )


        adapter = AdaptiveLoRALinear(
            original,
            rank=17,
            alpha=ALPHA,
            name=f"L{layer_idx}_{proj_name}",
        )


        setattr(
            layer.attention.self,
            proj_name,
            adapter,
        )


        lora_modules.append(adapter)


    # --------------------------------------------------------
    # Attention output
    # --------------------------------------------------------

    original = layer.attention.output.dense


    adapter = AdaptiveLoRALinear(
        original,
        rank=17,
        alpha=ALPHA,
        name=f"L{layer_idx}_output_dense",
    )


    layer.attention.output.dense = adapter


    lora_modules.append(adapter)


# ============================================================
# INITIAL RANK ALLOCATION
# ============================================================

# 48 modules total.
#
# 11 modules × 18
# 37 modules × 17
#
# 198 + 629 = 827

for i, module in enumerate(lora_modules):

    if i < 11:
        module.resize_rank(18)
    else:
        module.resize_rank(17)


assert len(lora_modules) == 48


# ============================================================
# FREEZE EVERYTHING
# ============================================================

for p in model.parameters():
    p.requires_grad = False


# ============================================================
# ENABLE LoRA
# ============================================================

for module in lora_modules:

    module.A.requires_grad = True
    module.B.requires_grad = True


# ============================================================
# ENABLE CLASSIFIER
# ============================================================

for p in model.classifier.parameters():
    p.requires_grad = True


# ============================================================
# PARAMETER COUNT
# ============================================================

def current_total_rank():

    return sum(
        m.rank
        for m in lora_modules
    )


def trainable_count():

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


print("\n" + "=" * 70)

print("LoRA modules:", len(lora_modules))

print(
    "Total LoRA rank:",
    current_total_rank()
)

print(
    "Total trainable parameters:",
    f"{trainable_count():,}"
)

print(
    "Trainable parameters (M):",
    f"{trainable_count() / 1e6:.4f}"
)

print("=" * 70)


assert (
    current_total_rank()
    ==
    TARGET_TOTAL_RANK
)


# ============================================================
# OPTIMIZER — ADAMW
# ============================================================

def build_optimizer():

    lora_params = (
        [m.A for m in lora_modules]
        +
        [m.B for m in lora_modules]
    )


    classifier_params = list(
        model.classifier.parameters()
    )


    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_params,
                "lr": LEARNING_RATE,
            },

            {
                "params": classifier_params,
                "lr": LEARNING_RATE,
            },
        ],

        betas=BETAS,
        eps=EPS,
        weight_decay=WEIGHT_DECAY,
    )


    return optimizer


optimizer = build_optimizer()


# ============================================================
# LINEAR LR SCHEDULER
# ============================================================

total_steps = (
    len(train_loader)
    * NUM_EPOCHS
)


warmup_steps = int(
    total_steps
    * WARMUP_RATIO
)


scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate():

    model.eval()

    predictions = []
    labels = []


    for batch in val_loader:

        batch = {
            k: v.to(
                device,
                non_blocking=True,
            )

            for k, v in batch.items()
        }


        outputs = model(
            **batch
        )


        predictions.extend(
            outputs.logits
            .argmax(-1)
            .cpu()
            .numpy()
            .tolist()
        )


        labels.extend(
            batch["labels"]
            .cpu()
            .numpy()
            .tolist()
        )


    return accuracy_score(
        labels,
        predictions,
    )


# ============================================================
# ADAPTIVE CONTROLLER
# ============================================================

def controller_step():

    global optimizer
    global scheduler


    # Update importance
    for m in lora_modules:
        m.update_importance()


    importance = np.array(
        [
            m.importance_ema
            for m in lora_modules
        ],
        dtype=np.float64,
    )


    if not np.isfinite(
        importance
    ).all():

        return False


    if importance.max() <= 0:

        return False


    normalized = (
        importance
        /
        (
            importance.mean()
            + 1e-12
        )
    )


    # --------------------------------------------------------
    # Receiver score
    # --------------------------------------------------------

    receiver_scores = np.array(
        [
            normalized[i]
            /
            math.sqrt(
                m.rank / 17
            )

            for i, m
            in enumerate(lora_modules)
        ]
    )


    # --------------------------------------------------------
    # Donor score
    # --------------------------------------------------------

    donor_scores = np.array(
        [
            normalized[i]
            *
            math.sqrt(
                m.rank / 17
            )

            for i, m
            in enumerate(lora_modules)
        ]
    )


    receiver_idx = int(
        np.argmax(
            receiver_scores
        )
    )


    donor_idx = int(
        np.argmin(
            donor_scores
        )
    )


    if receiver_idx == donor_idx:
        return False


    receiver = lora_modules[
        receiver_idx
    ]


    donor = lora_modules[
        donor_idx
    ]


    gap = (
        receiver_scores[
            receiver_idx
        ]
        -
        donor_scores[
            donor_idx
        ]
    ) / (
        abs(
            donor_scores[
                donor_idx
            ]
        )
        + 1e-12
    )


    print(
        f"Candidate: "
        f"{donor.name} -> "
        f"{receiver.name} "
        f"| gap={gap:.3f}"
    )


    if gap < IMPORTANCE_GAP:

        print(
            "Controller: no transfer."
        )

        return False


    if receiver.rank >= MAX_RANK:
        return False


    if donor.rank <= MIN_RANK:
        return False


    old_receiver_rank = receiver.rank
    old_donor_rank = donor.rank


    # Transfer one rank
    receiver.resize_rank(
        old_receiver_rank + 1
    )


    donor.resize_rank(
        old_donor_rank - 1
    )


    assert (
        current_total_rank()
        ==
        TARGET_TOTAL_RANK
    )


    # Rebuild optimizer because
    # Parameter objects changed.
    optimizer = build_optimizer()


    # Rebuild linear scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )


    print(
        f"TRANSFER: "
        f"{donor.name} "
        f"{old_donor_rank}->{donor.rank} | "
        f"{receiver.name} "
        f"{old_receiver_rank}->{receiver.rank}"
    )


    return True


# ============================================================
# TRAIN
# ============================================================

best_acc = 0.0

history = []


print("\n" + "=" * 70)
print("STARTING SST-2")
print("=" * 70)


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    model.train()

    running_loss = 0.0
    valid_steps = 0


    for step, batch in enumerate(
        train_loader,
        1,
    ):

        batch = {
            k: v.to(
                device,
                non_blocking=True,
            )

            for k, v in batch.items()
        }


        optimizer.zero_grad(
            set_to_none=True
        )


        outputs = model(
            **batch
        )


        loss = outputs.loss


        if not torch.isfinite(loss):

            print(
                f"Skipping invalid loss "
                f"at epoch={epoch}, "
                f"step={step}"
            )

            continue


        loss.backward()


        # Track LoRA importance
        for m in lora_modules:
            m.update_importance()


        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )


        optimizer.step()

        scheduler.step()


        running_loss += loss.item()

        valid_steps += 1


        if step % 250 == 0:

            print(
                f"Epoch {epoch}/{NUM_EPOCHS} | "
                f"Step {step}/{len(train_loader)} | "
                f"Loss "
                f"{running_loss / max(valid_steps, 1):.4f}"
            )


    # ========================================================
    # VALIDATION
    # ========================================================

    avg_loss = (
        running_loss
        /
        max(valid_steps, 1)
    )


    val_acc = evaluate()


    # ========================================================
    # CONTROLLER
    # ========================================================

    changed = False


    if (
        epoch >= CONTROLLER_START_EPOCH
        and
        epoch % CONTROLLER_INTERVAL == 0
    ):

        print(
            "\nRunning adaptive controller..."
        )


        changed = controller_step()


    # ========================================================
    # REPORT
    # ========================================================

    print(
        "\n" + "-" * 70
    )


    print(
        f"Epoch: "
        f"{epoch}/{NUM_EPOCHS}"
    )


    print(
        f"Training Loss: "
        f"{avg_loss:.4f}"
    )


    print(
        f"Validation Accuracy: "
        f"{val_acc * 100:.2f}%"
    )


    print(
        f"Total LoRA Rank: "
        f"{current_total_rank()}"
    )


    print(
        f"Trainable Params: "
        f"{trainable_count():,}"
    )


    print(
        "-" * 70
    )


    history.append(
        {
            "epoch": epoch,
            "loss": avg_loss,
            "val_accuracy": val_acc,
            "total_rank":
                current_total_rank(),
            "trainable_params":
                trainable_count(),
            "controller_changed":
                changed,
        }
    )


    # ========================================================
    # SAVE BEST
    # ========================================================

    if val_acc > best_acc:

        best_acc = val_acc


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "epoch":
                    epoch,

                "val_accuracy":
                    val_acc,

                "history":
                    history,

                "final_ranks":
                    [
                        m.rank
                        for m in lora_modules
                    ],
            },

            "sst2_deberta_qkvo_adamw_best.pt",
        )


        print(
            f"\n*** NEW BEST: "
            f"{best_acc * 100:.2f}% ***"
        )


# ============================================================
# FINAL SAVE
# ============================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "history":
            history,

        "best_accuracy":
            best_acc,

        "final_ranks":
            [
                m.rank
                for m in lora_modules
            ],
    },

    "sst2_deberta_qkvo_adamw_final.pt",
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINISHED")
print("=" * 70)


print(
    f"Best validation accuracy: "
    f"{best_acc * 100:.2f}%"
)


print(
    f"Final trainable parameters: "
    f"{trainable_count():,}"
)


print(
    f"Final total LoRA rank: "
    f"{current_total_rank()}"
)


# ============================================================
# FINAL RANK ALLOCATION
# ============================================================

print(
    "\nFinal rank allocation:"
)


for i in range(12):

    base = 4 * i

    q = lora_modules[base]
    k = lora_modules[base + 1]
    v = lora_modules[base + 2]
    o = lora_modules[base + 3]


    print(
        f"Layer {i:02d}: "
        f"Q={q.rank:2d} | "
        f"K={k.rank:2d} | "
        f"V={v.rank:2d} | "
        f"O={o.rank:2d}"
    )


print("\nSaved:")

print(
    "sst2_deberta_qkvo_adamw_best.pt"
)

print(
    "sst2_deberta_qkvo_adamw_final.pt"
)

print("\nDONE.")
