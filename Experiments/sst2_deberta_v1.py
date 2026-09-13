# ============================================================
# Corresponds to: Table 2, SST-2 (DeBERTa-v1-base, Seesaw-v2)
# Paper: "Seesaw: Budget-Preserving Rank Reallocation for LoRA"
# Under review at ICLR 2027
# ============================================================


import os
import gc
import copy
import math
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from sklearn.metrics import accuracy_score


# ============================================================
# 1. CONFIG
# ============================================================

MODEL_NAME = "microsoft/deberta-base"
DATASET_NAME = "glue"
DATASET_CONFIG = "sst2"

SEED = 42

NUM_EPOCHS = 5
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32

MAX_LENGTH = 128

LR = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
MAX_GRAD_NORM = 1.0

# LoRA
ALPHA = 32

TOTAL_RANK = 827
MIN_RANK = 2
MAX_RANK = 64

# See-Saw controller
CONTROLLER_START_EPOCH = 2
CONTROLLER_INTERVAL = 1

# EMA importance
EMA_DECAY = 0.85

# Z-score minimum normalized importance gap
IMPORTANCE_GAP = 0.25

OUTPUT_DIR = "./sst2_deberta_v1_seesaw"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DEVICE:", device)
print("MODEL :", MODEL_NAME)
print("=" * 70)


# ============================================================
# 4. LOAD SST-2
# ============================================================

print("\nLoading SST-2...")

dataset = load_dataset(
    DATASET_NAME,
    DATASET_CONFIG
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)


def tokenize_function(examples):

    return tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=MAX_LENGTH
    )


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True
)

# Keep only model inputs + labels
tokenized_dataset = tokenized_dataset.rename_column(
    "label",
    "labels"
)

columns_to_remove = [
    c for c in tokenized_dataset["train"].column_names
    if c not in ["input_ids", "attention_mask", "token_type_ids", "labels"]
]

tokenized_dataset = tokenized_dataset.remove_columns(
    columns_to_remove
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True
)

train_loader = DataLoader(
    tokenized_dataset["train"],
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=data_collator
)

eval_loader = DataLoader(
    tokenized_dataset["validation"],
    batch_size=EVAL_BATCH_SIZE,
    shuffle=False,
    collate_fn=data_collator
)

print("Train examples:", len(tokenized_dataset["train"]))
print("Validation examples:", len(tokenized_dataset["validation"]))


# ============================================================
# 5. LOAD DEBERTA-V1
# ============================================================

print("\nLoading DeBERTa-v1...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model = model.float()
model.to(device)


# ============================================================
# 6. ROBUST Q/K/V/O DISCOVERY
#
# This avoids hard-coding query_proj/key_proj/value_proj.
# It searches the actual Linear modules in your installed
# Transformers implementation.
# ============================================================

def discover_qkvo(model):

    discovered = []

    encoder_layers = model.deberta.encoder.layer

    print("\nDiscovering attention projections...")

    for layer_idx, layer in enumerate(encoder_layers):

        self_attn = layer.attention.self
        output_attn = layer.attention.output

        # ----------------------------------------------------
        # Self-attention linear modules
        # ----------------------------------------------------

        self_linears = {}

        for name, module in self_attn.named_modules():

            if isinstance(module, nn.Linear):
                self_linears[name] = module

        # ----------------------------------------------------
        # Find Q/K/V using names OR semantic ordering
        # ----------------------------------------------------

        def find_projection(names):

            for candidate in names:

                if candidate in self_linears:
                    return candidate, self_linears[candidate]

                # Also allow nested paths
                for actual_name, module in self_linears.items():

                    if actual_name.endswith(candidate):
                        return actual_name, module

            return None, None


        q_name, q_module = find_projection([
            "query_proj",
            "query",
            "q_proj"
        ])

        k_name, k_module = find_projection([
            "key_proj",
            "key",
            "k_proj"
        ])

        v_name, v_module = find_projection([
            "value_proj",
            "value",
            "v_proj"
        ])


        # ----------------------------------------------------
        # If names differ completely, use Linear module order.
        #
        # DeBERTa attention normally contains Q/K/V projections
        # as the first three Linear modules.
        # ----------------------------------------------------

        if q_module is None or k_module is None or v_module is None:

            linear_items = list(self_linears.items())

            if len(linear_items) >= 3:

                q_name, q_module = linear_items[0]
                k_name, k_module = linear_items[1]
                v_name, v_module = linear_items[2]


        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        output_linears = {}

        for name, module in output_attn.named_modules():

            if isinstance(module, nn.Linear):
                output_linears[name] = module


        o_module = None
        o_name = None

        for candidate in [
            "dense",
            "output",
            "out_proj",
            "o_proj"
        ]:

            if candidate in output_linears:

                o_name = candidate
                o_module = output_linears[candidate]
                break

            for actual_name, module in output_linears.items():

                if actual_name.endswith(candidate):

                    o_name = actual_name
                    o_module = module
                    break

            if o_module is not None:
                break


        # ----------------------------------------------------
        # If output name is unusual, use first Linear
        # ----------------------------------------------------

        if o_module is None:

            if len(output_linears) >= 1:

                o_name, o_module = list(
                    output_linears.items()
                )[0]


        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if q_module is None or k_module is None or v_module is None:

            print("\nCould not identify Q/K/V in layer", layer_idx)

            print("\nSelf-attention structure:")
            print(self_attn)

            print("\nLinear modules:")

            for name, module in self_linears.items():
                print(name, module)

            raise RuntimeError(
                f"Could not identify Q/K/V in layer {layer_idx}"
            )


        if o_module is None:

            print("\nCould not identify O in layer", layer_idx)

            print("\nAttention output structure:")
            print(output_attn)

            raise RuntimeError(
                f"Could not identify output projection "
                f"in layer {layer_idx}"
            )


        discovered.append({
            "key": f"layer{layer_idx}.query",
            "layer": layer_idx,
            "kind": "query",
            "parent": self_attn,
            "name": q_name,
            "module": q_module
        })

        discovered.append({
            "key": f"layer{layer_idx}.key",
            "layer": layer_idx,
            "kind": "key",
            "parent": self_attn,
            "name": k_name,
            "module": k_module
        })

        discovered.append({
            "key": f"layer{layer_idx}.value",
            "layer": layer_idx,
            "kind": "value",
            "parent": self_attn,
            "name": v_name,
            "module": v_module
        })

        discovered.append({
            "key": f"layer{layer_idx}.output",
            "layer": layer_idx,
            "kind": "output",
            "parent": output_attn,
            "name": o_name,
            "module": o_module
        })


    return discovered


qkvo_modules = discover_qkvo(model)


print("\n" + "=" * 70)
print("DISCOVERED Q/K/V/O MODULES")
print("=" * 70)

for item in qkvo_modules:

    module = item["module"]

    print(
        f"{item['key']:25s} "
        f"{item['name']:20s} "
        f"{tuple(module.weight.shape)}"
    )


assert len(qkvo_modules) == 48, (
    f"Expected 48 Q/K/V/O modules, "
    f"but found {len(qkvo_modules)}"
)


# ============================================================
# 7. LORA MODULE
# ============================================================

class LoRALinear(nn.Module):

    def __init__(
        self,
        original_layer,
        rank,
        alpha=32
    ):

        super().__init__()

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        self.alpha = alpha
        self.rank = rank

        # Frozen original layer
        self.weight = nn.Parameter(
            original_layer.weight.detach()
                .float()
                .clone(),
            requires_grad=False
        )

        if original_layer.bias is not None:

            self.bias = nn.Parameter(
                original_layer.bias.detach()
                    .float()
                    .clone(),
                requires_grad=False
            )

        else:

            self.bias = None


        # LoRA parameters
        self.A = nn.Parameter(
            torch.empty(
                rank,
                self.in_features,
                dtype=torch.float32
            )
        )

        self.B = nn.Parameter(
            torch.zeros(
                self.out_features,
                rank,
                dtype=torch.float32
            )
        )

        # Standard LoRA initialization
        nn.init.kaiming_uniform_(
            self.A,
            a=math.sqrt(5)
        )

        self.scaling = self.alpha / max(rank, 1)


    def forward(self, x):

        original = torch.nn.functional.linear(
            x,
            self.weight,
            self.bias
        )

        lora = torch.nn.functional.linear(
            torch.nn.functional.linear(
                x,
                self.A
            ),
            self.B
        )

        return original + self.scaling * lora


# ============================================================
# 8. INITIAL RANK ALLOCATION
#
# 48 modules
# 11 modules x 18
# 37 modules x 17
#
# 11*18 + 37*17 = 827
# ============================================================

num_modules = len(qkvo_modules)

base_rank = TOTAL_RANK // num_modules
remainder = TOTAL_RANK % num_modules

ranks = {}

for i, item in enumerate(qkvo_modules):

    if i < remainder:
        rank = base_rank + 1
    else:
        rank = base_rank

    ranks[item["key"]] = rank


assert sum(ranks.values()) == TOTAL_RANK

print("\nInitial rank allocation:")
print("Rank 18:", sum(r == 18 for r in ranks.values()))
print("Rank 17:", sum(r == 17 for r in ranks.values()))
print("Total  :", sum(ranks.values()))


# ============================================================
# 9. REPLACE LINEAR MODULES WITH LORA
# ============================================================

def set_child_module(parent, name, new_module):

    parts = name.split(".")

    current = parent

    for part in parts[:-1]:

        current = getattr(current, part)

    setattr(
        current,
        parts[-1],
        new_module
    )


lora_adapters = {}

for item in qkvo_modules:

    key = item["key"]

    original_module = item["module"]

    rank = ranks[key]

    lora = LoRALinear(
        original_module,
        rank=rank,
        alpha=ALPHA
    )

    lora = lora.to(device).float()

    set_child_module(
        item["parent"],
        item["name"],
        lora
    )

    lora_adapters[key] = lora


# ============================================================
# 10. FREEZE EVERYTHING EXCEPT LORA
# ============================================================

for param in model.parameters():

    param.requires_grad = False


for adapter in lora_adapters.values():

    adapter.A.requires_grad = True
    adapter.B.requires_grad = True


trainable_params = [
    p
    for p in model.parameters()
    if p.requires_grad
]


trainable_count = sum(
    p.numel()
    for p in trainable_params
)

total_count = sum(
    p.numel()
    for p in model.parameters()
)

print("\n" + "=" * 70)
print("PARAMETER COUNTS")
print("=" * 70)

print(
    f"Trainable parameters: {trainable_count:,}"
)

print(
    f"Total parameters:     {total_count:,}"
)

print(
    f"Trainable %:          "
    f"{100 * trainable_count / total_count:.4f}%"
)


# ============================================================
# 11. OPTIMIZER
# ============================================================

def make_optimizer():

    params = []

    for adapter in lora_adapters.values():

        params.append(adapter.A)
        params.append(adapter.B)

    return torch.optim.AdamW(
        params,
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )


optimizer = make_optimizer()


num_training_steps = (
    NUM_EPOCHS * len(train_loader)
)

num_warmup_steps = int(
    WARMUP_RATIO * num_training_steps
)


scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)


# ============================================================
# 12. EMA IMPORTANCE
# ============================================================

importance_ema = {
    key: 0.0
    for key in lora_adapters
}


# ============================================================
# 13. COMPUTE Z-SCORE IMPORTANCE
# ============================================================

def get_zscore_importance():

    raw_scores = {}

    for key, adapter in lora_adapters.items():

        if adapter.B.grad is None:

            raw_scores[key] = 0.0

        else:

            importance = (
                adapter.B.grad
                .detach()
                .float()
                .abs()
                .mean()
                .item()
            )

            importance_ema[key] = (
                EMA_DECAY * importance_ema[key]
                + (1.0 - EMA_DECAY) * importance
            )

            raw_scores[key] = importance_ema[key]


    scores = np.array(
        list(raw_scores.values()),
        dtype=np.float64
    )


    mean_score = scores.mean()

    std_score = scores.std()

    if std_score < 1e-12:

        normalized = np.zeros_like(scores)

    else:

        normalized = (
            scores - mean_score
        ) / (
            std_score + 1e-12
        )


    z_scores = {
        key: float(z)
        for key, z in zip(
            raw_scores.keys(),
            normalized
        )
    }


    return raw_scores, z_scores


# ============================================================
# 14. SEE-SAW RANK TRANSFER
# ============================================================

def resize_lora_rank(
    key,
    new_rank
):

    adapter = lora_adapters[key]

    old_rank = adapter.rank

    if new_rank == old_rank:
        return


    device_local = adapter.A.device


    # --------------------------------------------------------
    # New A
    # --------------------------------------------------------

    new_A = torch.empty(
        new_rank,
        adapter.in_features,
        dtype=torch.float32,
        device=device_local
    )

    nn.init.kaiming_uniform_(
        new_A,
        a=math.sqrt(5)
    )


    # Preserve existing A components
    copy_rank = min(
        old_rank,
        new_rank
    )

    with torch.no_grad():

        new_A[:copy_rank].copy_(
            adapter.A[:copy_rank]
        )


    # --------------------------------------------------------
    # New B
    # --------------------------------------------------------

    new_B = torch.zeros(
        adapter.out_features,
        new_rank,
        dtype=torch.float32,
        device=device_local
    )

    with torch.no_grad():

        new_B[:, :copy_rank].copy_(
            adapter.B[:, :copy_rank]
        )


    # --------------------------------------------------------
    # Replace parameters
    # --------------------------------------------------------

    adapter.A = nn.Parameter(
        new_A,
        requires_grad=True
    )

    adapter.B = nn.Parameter(
        new_B,
        requires_grad=True
    )

    adapter.rank = new_rank

    adapter.scaling = (
        adapter.alpha /
        max(new_rank, 1)
    )


def seesaw_controller():

    raw_scores, z_scores = get_zscore_importance()


    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    values = np.array(
        list(z_scores.values()),
        dtype=np.float64
    )

    print("\nController diagnostics:")

    print(
        f"  Raw mean = "
        f"{np.mean(list(raw_scores.values())):.8f}"
    )

    print(
        f"  Raw min  = "
        f"{min(raw_scores.values()):.8f}"
    )

    print(
        f"  Raw max  = "
        f"{max(raw_scores.values()):.8f}"
    )

    print(
        f"  Z mean   = "
        f"{values.mean():.6f}"
    )

    print(
        f"  Z std    = "
        f"{values.std():.6f}"
    )


    # --------------------------------------------------------
    # Receiver = highest Z-score
    # Donor     = lowest Z-score
    # --------------------------------------------------------

    receiver = max(
        z_scores,
        key=z_scores.get
    )

    donor = min(
        z_scores,
        key=z_scores.get
    )


    receiver_score = z_scores[receiver]
    donor_score = z_scores[donor]

    gap = (
        receiver_score
        - donor_score
    )


    print(
        f"  Receiver = {receiver} "
        f"(z={receiver_score:.4f}, "
        f"rank={ranks[receiver]})"
    )

    print(
        f"  Donor    = {donor} "
        f"(z={donor_score:.4f}, "
        f"rank={ranks[donor]})"
    )

    print(
        f"  Gap      = {gap:.4f}"
    )


    # --------------------------------------------------------
    # Check gap
    # --------------------------------------------------------

    if gap < IMPORTANCE_GAP:

        print(
            "  No transfer: "
            "importance gap below threshold."
        )

        return False


    # --------------------------------------------------------
    # Check rank limits
    # --------------------------------------------------------

    if ranks[donor] <= MIN_RANK:

        print(
            "  No transfer: donor at MIN_RANK."
        )

        return False


    if ranks[receiver] >= MAX_RANK:

        print(
            "  No transfer: receiver at MAX_RANK."
        )

        return False


    # --------------------------------------------------------
    # Transfer exactly one rank
    # --------------------------------------------------------

    old_receiver_rank = ranks[receiver]
    old_donor_rank = ranks[donor]


    new_receiver_rank = (
        old_receiver_rank + 1
    )

    new_donor_rank = (
        old_donor_rank - 1
    )


    resize_lora_rank(
        receiver,
        new_receiver_rank
    )

    resize_lora_rank(
        donor,
        new_donor_rank
    )


    ranks[receiver] = new_receiver_rank
    ranks[donor] = new_donor_rank


    # --------------------------------------------------------
    # Exact budget invariant
    # --------------------------------------------------------

    current_total = sum(
        ranks.values()
    )

    assert current_total == TOTAL_RANK


    print(
        f"  TRANSFER: "
        f"{donor} {old_donor_rank}->{new_donor_rank} "
        f"| "
        f"{receiver} {old_receiver_rank}->{new_receiver_rank}"
    )

    print(
        f"  Total rank = {current_total}"
    )


    return True


# ============================================================
# 15. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate():

    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0
    total_examples = 0


    for batch in eval_loader:

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        outputs = model(**batch)

        loss = outputs.loss

        logits = outputs.logits

        predictions = (
            torch.argmax(
                logits,
                dim=-1
            )
        )


        batch_size = batch["labels"].size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_examples += batch_size


        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_labels.extend(
            batch["labels"]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )


    avg_loss = (
        total_loss /
        max(total_examples, 1)
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    return avg_loss, accuracy


# ============================================================
# 16. TRAINING
# ============================================================

best_accuracy = -float("inf")
best_epoch = None
best_state = None

history = []

print("\n")
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(
    1,
    NUM_EPOCHS + 1
):

    model.train()

    running_loss = 0.0
    num_examples = 0


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for step, batch in enumerate(
        train_loader,
        start=1
    ):

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


        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            trainable_params,
            MAX_GRAD_NORM
        )


        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer.step()

        scheduler.step()


        batch_size = (
            batch["labels"].size(0)
        )

        running_loss += (
            loss.item() * batch_size
        )

        num_examples += batch_size


    train_loss = (
        running_loss /
        max(num_examples, 1)
    )


    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    val_loss, accuracy = evaluate()


    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "accuracy": accuracy,
        "total_rank": sum(ranks.values())
    })


    print(
        f"\nEpoch {epoch:02d}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Accuracy: {accuracy:.4f}"
    )


    # --------------------------------------------------------
    # BEST CHECKPOINT
    #
    # Save best BEFORE controller resizing.
    # --------------------------------------------------------

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_epoch = epoch

        best_state = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            best_state,
            os.path.join(
                OUTPUT_DIR,
                "sst2_deberta_v1_seesaw_best.pt"
            )
        )

        print(
            f"  ★ NEW BEST "
            f"Accuracy = {best_accuracy:.4f}"
        )


    # --------------------------------------------------------
    # SEE-SAW CONTROLLER
    #
    # Controller begins after epoch 2 and runs every 2 epochs.
    # Therefore: 2,4,6,...,24
    # --------------------------------------------------------

    if (
        epoch >= CONTROLLER_START_EPOCH
        and
        epoch % CONTROLLER_INTERVAL == 0
        and
        epoch < NUM_EPOCHS
    ):

        print(
            "\n  >>> SEE-SAW CONTROLLER"
        )


        transferred = (
            seesaw_controller()
        )


        # ----------------------------------------------------
        # Rebuild optimizer after rank resize
        # ----------------------------------------------------

        if transferred:

            optimizer = make_optimizer()

            print(
                "  Optimizer rebuilt after "
                "rank transfer."
            )


        # ----------------------------------------------------
        # Exact rank invariant
        # ----------------------------------------------------

        assert sum(
            ranks.values()
        ) == TOTAL_RANK


    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 17. RESTORE BEST CHECKPOINT
# ============================================================

print("\n")
print("=" * 70)
print("RESTORING BEST CHECKPOINT")
print("=" * 70)

if best_state is not None:

    model.load_state_dict(
        best_state
    )


# ============================================================
# 18. FINAL EVALUATION
# ============================================================

final_val_loss, final_accuracy = evaluate()


print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(
    f"Best epoch:       {best_epoch}"
)

print(
    f"Best Accuracy:    {best_accuracy:.4f}"
)

print(
    f"Final Val Loss:   {final_val_loss:.4f}"
)

print(
    f"Final Accuracy:   {final_accuracy:.4f}"
)

print(
    f"Total rank:       {sum(ranks.values())}"
)


# ============================================================
# 19. FINAL RANK ALLOCATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL SEE-SAW RANK ALLOCATION")
print("=" * 70)

for layer_idx in range(12):

    print(
        f"\nLayer {layer_idx}:"
    )

    for kind in [
        "query",
        "key",
        "value",
        "output"
    ]:

        key = (
            f"layer{layer_idx}.{kind}"
        )

        print(
            f"  {kind:8s}: "
            f"{ranks[key]}"
        )


# ============================================================
# 20. SAVE FINAL RANK ALLOCATION
# ============================================================

rank_path = os.path.join(
    OUTPUT_DIR,
    "final_rank_allocation.txt"
)

with open(
    rank_path,
    "w"
) as f:

    f.write(
        "DeBERTa-v1 SST-2 See-Saw "
        "Final Rank Allocation\n"
    )

    f.write(
        "=" * 60 + "\n"
    )

    for key in sorted(ranks.keys()):

        f.write(
            f"{key}: {ranks[key]}\n"
        )

    f.write(
        f"\nTotal rank: {sum(ranks.values())}\n"
    )


# ============================================================
# 21. SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    OUTPUT_DIR,
    "training_history.txt"
)

with open(
    history_path,
    "w"
) as f:

    f.write(
        "epoch,train_loss,val_loss,accuracy,total_rank\n"
    )

    for row in history:

        f.write(
            f"{row['epoch']},"
            f"{row['train_loss']:.8f},"
            f"{row['val_loss']:.8f},"
            f"{row['accuracy']:.8f},"
            f"{row['total_rank']}\n"
        )


# ============================================================
# 22. FINAL ASSERTIONS
# ============================================================

assert len(lora_adapters) == 48

assert sum(
    ranks.values()
) == TOTAL_RANK

assert all(
    MIN_RANK <= r <= MAX_RANK
    for r in ranks.values()
)


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(
    f"Best SST-2 Accuracy: {best_accuracy:.4f}"
)

print(
    f"Best Epoch:          {best_epoch}"
)

print(
    f"Final Rank Budget:   {sum(ranks.values())}"
)

print(
    f"Checkpoint:          "
    f"{OUTPUT_DIR}/sst2_deberta_v1_seesaw_best.pt"
)

print(
    f"Rank allocation:     {rank_path}"
)

print(
    f"Training history:    {history_path}"
)

print("=" * 70)
