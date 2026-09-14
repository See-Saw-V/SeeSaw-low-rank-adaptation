# ============================================================
# Corresponds to: Table 1, AG News (BERT-base, Seesaw-v1)
# Paper: "Seesaw: Budget-Preserving Rank Reallocation for LoRA"
# Under review at ICLR 2027
# ============================================================



import os
import time
import pickle
import random
import numpy as np

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

MODEL_NAME = "bert-base-uncased"

INITIAL_RANK = 8
ALPHA = 16

BATCH_SIZE = 32

LEARNING_RATE = 2e-4

NUM_EPOCHS = 5

WARMUP_EPOCHS = 1

MAX_LENGTH = 128

MIN_RANK = 2
MAX_RANK = 16

MAX_RANK_MOVE = 1

OUTPUT_DIR = "./outputs/ag_news_bert_base"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


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
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("AG NEWS CONTROLLED A-LoRA")
print("=" * 70)

print("Model         :", MODEL_NAME)
print("Device        :", device)
print("Initial rank  :", INITIAL_RANK)
print("Alpha         :", ALPHA)
print("Batch size    :", BATCH_SIZE)
print("Learning rate :", LEARNING_RATE)
print("Epochs        :", NUM_EPOCHS)
print("Warm-up       :", WARMUP_EPOCHS)
print("Min rank      :", MIN_RANK)
print("Max rank      :", MAX_RANK)

print("=" * 70)


# ============================================================
# LOAD AG NEWS
# ============================================================

print("\nLoading AG News...")

dataset = load_dataset(
    "ag_news"
)

print(dataset)


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_function(examples):

    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)


# ============================================================
# COLLATE
# ============================================================

def collate_fn(batch):

    result = {

        "input_ids": torch.tensor(
            [
                x["input_ids"]
                for x in batch
            ],
            dtype=torch.long
        ),

        "attention_mask": torch.tensor(
            [
                x["attention_mask"]
                for x in batch
            ],
            dtype=torch.long
        ),

        "labels": torch.tensor(
            [
                x["label"]
                for x in batch
            ],
            dtype=torch.long
        )
    }

    if "token_type_ids" in batch[0]:

        result["token_type_ids"] = torch.tensor(

            [
                x["token_type_ids"]
                for x in batch
            ],

            dtype=torch.long
        )

    return result


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(

    tokenized_dataset["train"],

    batch_size=BATCH_SIZE,

    shuffle=True,

    collate_fn=collate_fn
)


test_loader = DataLoader(

    tokenized_dataset["test"],

    batch_size=BATCH_SIZE,

    shuffle=False,

    collate_fn=collate_fn
)


print()
print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(
    "Training samples   :",
    len(tokenized_dataset["train"])
)

print(
    "Test samples       :",
    len(tokenized_dataset["test"])
)

print(
    "Batches per epoch  :",
    len(train_loader)
)

print(
    "Total training batches:",
    len(train_loader) * NUM_EPOCHS
)

print("=" * 70)


# ============================================================
# ADAPTIVE LoRA LINEAR
# ============================================================

class AdaptiveLoRALinear(nn.Module):

    def __init__(
        self,
        original_linear,
        rank=8,
        alpha=16
    ):

        super().__init__()

        self.rank = rank

        self.alpha = alpha

        self.in_features = (
            original_linear.in_features
        )

        self.out_features = (
            original_linear.out_features
        )

        # ----------------------------------------------------
        # FROZEN BASE WEIGHT
        # ----------------------------------------------------

        self.weight = (
            original_linear.weight
        )

        self.weight.requires_grad = False


        # ----------------------------------------------------
        # FROZEN BIAS
        # ----------------------------------------------------

        self.bias = (
            original_linear.bias
        )

        if self.bias is not None:

            self.bias.requires_grad = False


        # ----------------------------------------------------
        # LoRA A
        # ----------------------------------------------------

        self.A = nn.Parameter(

            torch.randn(
                rank,
                self.in_features
            ) * 0.01
        )


        # ----------------------------------------------------
        # LoRA B
        # ----------------------------------------------------

        self.B = nn.Parameter(

            torch.zeros(
                self.out_features,
                rank
            )
        )


        self.scale = (
            alpha / rank
        )

        self.importance = 0.0


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        base = (
            x @ self.weight.T
        )

        if self.bias is not None:

            base = (
                base + self.bias
            )

        lora = (
            x @ self.A.T
        )

        lora = (
            lora @ self.B.T
        )

        return (
            base
            +
            self.scale * lora
        )


    # ========================================================
    # IMPORTANCE
    # ========================================================

    def compute_importance(self):

        if self.B.grad is None:

            self.importance = 0.0

        else:

            self.importance = (

                self.B.grad
                .detach()
                .abs()
                .mean()
                .item()
            )

        return self.importance


    # ========================================================
    # RESIZE RANK
    # ========================================================

    def resize_rank(
        self,
        new_rank
    ):

        new_rank = int(

            max(
                MIN_RANK,
                min(
                    MAX_RANK,
                    new_rank
                )
            )
        )


        if new_rank == self.rank:

            return False


        old_rank = self.rank


        old_A = (
            self.A.detach()
            .clone()
        )

        old_B = (
            self.B.detach()
            .clone()
        )


        new_A = torch.zeros(

            new_rank,

            self.in_features,

            device=old_A.device,

            dtype=old_A.dtype
        )


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


        # Preserve learned components

        new_A[:keep] = (
            old_A[:keep]
        )

        new_B[:, :keep] = (
            old_B[:, :keep]
        )


        # Initialize new components

        if new_rank > old_rank:

            nn.init.normal_(

                new_A[old_rank:],

                mean=0.0,

                std=0.01
            )


        self.rank = new_rank


        self.A = nn.Parameter(
            new_A
        )

        self.B = nn.Parameter(
            new_B
        )


        self.scale = (
            self.alpha / new_rank
        )


        return True


# ============================================================
# LOAD BERT
# ============================================================

print("\nLoading BERT...")

model = AutoModelForSequenceClassification.from_pretrained(

    MODEL_NAME,

    num_labels=4
)


# ============================================================
# FREEZE BERT
# ============================================================

for param in model.bert.parameters():

    param.requires_grad = False


# ============================================================
# CLASSIFIER TRAINABLE
# ============================================================

for param in model.classifier.parameters():

    param.requires_grad = True


# ============================================================
# INJECT A-LoRA INTO QUERY ONLY
# ============================================================

for layer in model.bert.encoder.layer:

    original_query = (
        layer.attention.self.query
    )

    layer.attention.self.query = (

        AdaptiveLoRALinear(

            original_linear=original_query,

            rank=INITIAL_RANK,

            alpha=ALPHA
        )
    )


# ============================================================
# MOVE MODEL
# ============================================================

model.to(device)


# ============================================================
# QUERY LAYERS
# ============================================================

query_layers = [

    layer.attention.self.query

    for layer in model.bert.encoder.layer
]


NUM_LAYERS = len(
    query_layers
)


# ============================================================
# GLOBAL RANK BUDGET
# ============================================================

TOTAL_RANK_BUDGET = (

    NUM_LAYERS
    *
    INITIAL_RANK
)


initial_ranks = [

    q.rank

    for q in query_layers
]


print()
print("=" * 70)
print("A-LoRA CONFIGURATION")
print("=" * 70)

print(
    "A-LoRA layers        :",
    NUM_LAYERS
)

print(
    "Initial ranks        :",
    initial_ranks
)

print(
    "Initial average rank :",
    sum(initial_ranks)
    /
    NUM_LAYERS
)

print(
    "Total rank budget    :",
    TOTAL_RANK_BUDGET
)


# ============================================================
# PARAMETER REPORT
# ============================================================

total_parameters = sum(

    p.numel()

    for p in model.parameters()
)


trainable_parameters = sum(

    p.numel()

    for p in model.parameters()

    if p.requires_grad
)


trainable_percentage = (

    trainable_parameters
    /
    total_parameters
    *
    100
)


print()
print("=" * 70)
print("PARAMETER REPORT")
print("=" * 70)

print(
    "Total parameters     :",
    f"{total_parameters:,}"
)

print(
    "Trainable parameters :",
    f"{trainable_parameters:,}"
)

print(
    "Trainable percentage :",
    f"{trainable_percentage:.4f}%"
)


# ============================================================
# OPTIMIZER
# ============================================================

def create_optimizer():

    return AdamW(

        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],

        lr=LEARNING_RATE
    )


optimizer = create_optimizer()


# ============================================================
# HISTORY
# ============================================================

loss_history = []

epoch_loss_history = []

accuracy_history = []

rank_history = []

layer_rank_history = []

importance_history = []


# ============================================================
# IMPORTANCE ACCUMULATION
# ============================================================

importance_sum = np.zeros(

    NUM_LAYERS,

    dtype=np.float64
)

importance_count = 0


# ============================================================
# RANK CONTROLLER
# ============================================================

def redistribute_rank_budget():

    current_ranks = [

        q.rank
        for q in query_layers
    ]


    importances = np.array(

        [
            q.importance
            for q in query_layers
        ],

        dtype=np.float64
    )


    if np.all(
        importances <= 0
    ):

        print(
            "No positive importance."
        )

        return False


    # Highest importance first

    high_order = np.argsort(
        -importances
    )


    # Lowest importance first

    low_order = np.argsort(
        importances
    )


    receiver_idx = None
    donor_idx = None


    # --------------------------------------------------------
    # FIND RECEIVER
    # --------------------------------------------------------

    for idx in high_order:

        if current_ranks[idx] < MAX_RANK:

            receiver_idx = int(idx)

            break


    # --------------------------------------------------------
    # FIND DONOR
    # --------------------------------------------------------

    for idx in low_order:

        if current_ranks[idx] > MIN_RANK:

            donor_idx = int(idx)

            break


    if (
        receiver_idx is None
        or donor_idx is None
    ):

        return False


    if receiver_idx == donor_idx:

        return False


    high_importance = (
        importances[
            receiver_idx
        ]
    )


    low_importance = (
        importances[
            donor_idx
        ]
    )


    # --------------------------------------------------------
    # IMPORTANCE GAP
    # --------------------------------------------------------

    relative_gap = (

        high_importance
        -
        low_importance

    ) / (

        high_importance
        +
        1e-12
    )


    # Same threshold as SST-2

    if relative_gap < 0.10:

        return False


    # --------------------------------------------------------
    # TRANSFER ONE RANK
    # --------------------------------------------------------

    old_receiver_rank = (
        current_ranks[
            receiver_idx
        ]
    )


    old_donor_rank = (
        current_ranks[
            donor_idx
        ]
    )


    query_layers[
        donor_idx
    ].resize_rank(

        old_donor_rank
        -
        MAX_RANK_MOVE
    )


    query_layers[
        receiver_idx
    ].resize_rank(

        old_receiver_rank
        +
        MAX_RANK_MOVE
    )


    # --------------------------------------------------------
    # VERIFY GLOBAL BUDGET
    # --------------------------------------------------------

    new_ranks = [

        q.rank
        for q in query_layers
    ]


    if sum(new_ranks) != TOTAL_RANK_BUDGET:

        raise RuntimeError(

            f"Rank budget violated: "
            f"{sum(new_ranks)} != "
            f"{TOTAL_RANK_BUDGET}"
        )


    print()
    print("=" * 60)
    print("RANK TRANSFER")
    print("=" * 60)

    print(

        f"Layer {donor_idx + 1}: "
        f"{old_donor_rank} -> "
        f"{query_layers[donor_idx].rank}"
    )

    print(

        f"Layer {receiver_idx + 1}: "
        f"{old_receiver_rank} -> "
        f"{query_layers[receiver_idx].rank}"
    )

    print(
        "Total rank:",
        sum(new_ranks)
    )

    print("=" * 60)


    return True


# ============================================================
# TRAINING
# ============================================================

training_start = time.time()

global_step = 0


print()
print("=" * 70)
print("STARTING AG NEWS A-LoRA TRAINING")
print("=" * 70)


for epoch in range(
    NUM_EPOCHS
):

    epoch_start = time.time()

    model.train()

    epoch_loss = 0.0


    # Reset importance

    importance_sum[:] = 0.0

    importance_count = 0


    if epoch < WARMUP_EPOCHS:

        phase = "WARM-UP"

    else:

        phase = "ADAPTIVE"


    print()
    print("=" * 70)

    print(
        f"EPOCH {epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        f"PHASE: {phase}"
    )

    print("=" * 70)


    # ========================================================
    # TRAINING BATCHES
    # ========================================================

    for batch_idx, batch in enumerate(

        train_loader,

        start=1
    ):

        global_step += 1


        batch = {

            k: v.to(device)

            for k, v in batch.items()
        }


        optimizer.zero_grad()


        # ----------------------------------------------------
        # FORWARD
        # ----------------------------------------------------

        model_inputs = {

            "input_ids":
                batch["input_ids"],

            "attention_mask":
                batch["attention_mask"],

            "labels":
                batch["labels"]
        }


        if "token_type_ids" in batch:

            model_inputs[
                "token_type_ids"
            ] = batch[
                "token_type_ids"
            ]


        outputs = model(
            **model_inputs
        )


        loss = outputs.loss

        loss_value = loss.item()


        loss_history.append(
            loss_value
        )


        epoch_loss += loss_value


        # ----------------------------------------------------
        # BACKWARD
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # IMPORTANCE
        # ----------------------------------------------------

        batch_importances = []


        for query in query_layers:

            query.compute_importance()


            batch_importances.append(
                query.importance
            )


        importance_sum += np.array(
            batch_importances
        )


        importance_count += 1


        # ----------------------------------------------------
        # OPTIMIZER STEP
        # ----------------------------------------------------

        optimizer.step()


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if global_step % 100 == 0:

            current_ranks = [

                q.rank

                for q in query_layers
            ]


            print(

                f"Step {global_step:5d} | "

                f"Loss {loss_value:.4f} | "

                f"Avg Rank "
                f"{sum(current_ranks) / NUM_LAYERS:.2f} | "

                f"Rank Range "
                f"{min(current_ranks)}-"
                f"{max(current_ranks)}"
            )


    # ========================================================
    # AVERAGE IMPORTANCE
    # ========================================================

    if importance_count > 0:

        epoch_importances = (

            importance_sum
            /
            importance_count
        )

    else:

        epoch_importances = (
            np.zeros(NUM_LAYERS)
        )


    for i, query in enumerate(
        query_layers
    ):

        query.importance = float(
            epoch_importances[i]
        )


    importance_history.append(
        epoch_importances.tolist()
    )


    # ========================================================
    # RANK ADAPTATION
    # ========================================================

    rank_changed = False


    if epoch >= WARMUP_EPOCHS:

        print()
        print(
            "Applying epoch-level "
            "rank adaptation..."
        )


        rank_changed = (
            redistribute_rank_budget()
        )


        if rank_changed:

            optimizer = (
                create_optimizer()
            )

        else:

            print(
                "No rank transfer."
            )


    else:

        print()
        print(
            "Warm-up complete."
        )

        print(
            "No rank adaptation."
        )


    # ========================================================
    # RANK REPORT
    # ========================================================

    current_ranks = [

        q.rank

        for q in query_layers
    ]


    current_total_rank = sum(
        current_ranks
    )


    if current_total_rank != TOTAL_RANK_BUDGET:

        raise RuntimeError(

            "GLOBAL RANK BUDGET VIOLATION"
        )


    average_rank = (

        current_total_rank
        /
        NUM_LAYERS
    )


    rank_history.append(
        average_rank
    )

    layer_rank_history.append(
        current_ranks.copy()
    )


    # ========================================================
    # EPOCH LOSS
    # ========================================================

    avg_epoch_loss = (

        epoch_loss
        /
        len(train_loader)
    )


    epoch_loss_history.append(
        avg_epoch_loss
    )


    # ========================================================
    # EVALUATION
    # ========================================================

    model.eval()

    correct = 0
    total = 0


    with torch.no_grad():

        for batch in test_loader:

            batch = {

                k: v.to(device)

                for k, v in batch.items()
            }


            model_inputs = {

                "input_ids":
                    batch["input_ids"],

                "attention_mask":
                    batch["attention_mask"]
            }


            if "token_type_ids" in batch:

                model_inputs[
                    "token_type_ids"
                ] = batch[
                    "token_type_ids"
                ]


            outputs = model(
                **model_inputs
            )


            predictions = (

                outputs.logits
                .argmax(
                    dim=1
                )
            )


            correct += (

                predictions
                ==
                batch["labels"]

            ).sum().item()


            total += (
                batch["labels"]
                .size(0)
            )


    accuracy = (

        correct
        /
        total
    )


    accuracy_history.append(
        accuracy
    )


    # ========================================================
    # EPOCH REPORT
    # ========================================================

    epoch_time = (

        time.time()
        -
        epoch_start

    ) / 60


    print()
    print("=" * 70)

    print(
        f"EPOCH {epoch + 1}/{NUM_EPOCHS} COMPLETE"
    )

    print(
        f"Training Loss : "
        f"{avg_epoch_loss:.6f}"
    )

    print(
        f"Accuracy      : "
        f"{accuracy * 100:.4f}%"
    )

    print(
        f"Average Rank  : "
        f"{average_rank:.2f}"
    )

    print(
        f"Total Rank    : "
        f"{current_total_rank}"
    )

    print(
        "Layer Ranks   :",
        current_ranks
    )

    print(
        "Rank Changed  :",
        rank_changed
    )

    print(
        f"Epoch Time    : "
        f"{epoch_time:.2f} minutes"
    )

    print("=" * 70)


    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint_path = os.path.join(

        OUTPUT_DIR,

        f"alora_agnews_epoch{epoch + 1}.pth"
    )


    history_path = os.path.join(

        OUTPUT_DIR,

        f"alora_agnews_history_epoch{epoch + 1}.pkl"
    )


    torch.save(

        model.state_dict(),

        checkpoint_path
    )


    history = {

        "dataset":
            "AG News",

        "method":
            "Controlled A-LoRA",

        "initial_rank":
            INITIAL_RANK,

        "alpha":
            ALPHA,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "warmup_epochs":
            WARMUP_EPOCHS,

        "loss_history":
            loss_history,

        "epoch_loss_history":
            epoch_loss_history,

        "accuracy_history":
            accuracy_history,

        "rank_history":
            rank_history,

        "layer_rank_history":
            layer_rank_history,

        "importance_history":
            importance_history,

        "initial_ranks":
            initial_ranks,

        "final_ranks":
            current_ranks,

        "total_rank_budget":
            TOTAL_RANK_BUDGET
    }


    with open(
        history_path,
        "wb"
    ) as f:

        pickle.dump(
            history,
            f
        )


    print(
        "Checkpoint saved:",
        checkpoint_path
    )


# ============================================================
# FINAL RESULT
# ============================================================

training_time = (

    time.time()
    -
    training_start

) / 60


final_ranks = [

    q.rank

    for q in query_layers
]


final_average_rank = (

    sum(final_ranks)
    /
    NUM_LAYERS
)


print()
print("=" * 70)
print("FINAL AG NEWS A-LoRA RESULT")
print("=" * 70)

print(
    f"Final Accuracy        : "
    f"{accuracy_history[-1] * 100:.4f}%"
)

print(
    f"Best Accuracy         : "
    f"{max(accuracy_history) * 100:.4f}%"
)

print(
    f"Training Time         : "
    f"{training_time:.2f} minutes"
)

print(
    f"Total Parameters      : "
    f"{total_parameters:,}"
)

print(
    f"Trainable Parameters  : "
    f"{trainable_parameters:,}"
)

print(
    f"Trainable Percentage  : "
    f"{trainable_percentage:.4f}%"
)

print(
    f"Initial Average Rank  : "
    f"{INITIAL_RANK:.4f}"
)

print(
    f"Final Average Rank    : "
    f"{final_average_rank:.4f}"
)

print(
    "Initial Ranks         :",
    initial_ranks
)

print(
    "Final Ranks           :",
    final_ranks
)

print(
    "Initial Total Budget  :",
    TOTAL_RANK_BUDGET
)

print(
    "Final Total Budget    :",
    sum(final_ranks)
)

print("=" * 70)


# ============================================================
# FINAL SAVE
# ============================================================

final_model_path = os.path.join(

    OUTPUT_DIR,

    "alora_agnews_final.pth"
)


final_history_path = os.path.join(

    OUTPUT_DIR,

    "alora_agnews_final.pkl"
)


torch.save(

    model.state_dict(),

    final_model_path
)


final_history = {

    "dataset":
        "AG News",

    "method":
        "Controlled A-LoRA",

    "final_accuracy":
        accuracy_history[-1],

    "best_accuracy":
        max(accuracy_history),

    "training_time":
        training_time,

    "total_parameters":
        total_parameters,

    "trainable_parameters":
        trainable_parameters,

    "trainable_percentage":
        trainable_percentage,

    "initial_rank":
        INITIAL_RANK,

    "initial_ranks":
        initial_ranks,

    "final_ranks":
        final_ranks,

    "initial_average_rank":
        INITIAL_RANK,

    "final_average_rank":
        final_average_rank,

    "initial_total_rank":
        TOTAL_RANK_BUDGET,

    "final_total_rank":
        sum(final_ranks),

    "alpha":
        ALPHA,

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "epochs":
        NUM_EPOCHS,

    "warmup_epochs":
        WARMUP_EPOCHS,

    "seed":
        SEED
}


with open(

    final_history_path,

    "wb"

) as f:

    pickle.dump(
        final_history,
        f
    )


print()
print("Final model saved:")
print(final_model_path)

print()
print("Final history saved:")
print(final_history_path)

print()
print("=" * 70)
print("AG NEWS A-LoRA COMPLETE")
print("=" * 70)
