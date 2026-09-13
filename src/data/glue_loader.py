from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding


TASK_CONFIG = {
    "cola":   {"glue_name": "cola",   "text_keys": ["sentence"], "num_labels": 2},
    "sst2":   {"glue_name": "sst2",   "text_keys": ["sentence"], "num_labels": 2},
    "mrpc":   {"glue_name": "mrpc",   "text_keys": ["sentence1", "sentence2"], "num_labels": 2},
    "stsb":   {"glue_name": "stsb",   "text_keys": ["sentence1", "sentence2"], "num_labels": 1},
    "qnli":   {"glue_name": "qnli",   "text_keys": ["question", "sentence"], "num_labels": 2},
    "rte":    {"glue_name": "rte",    "text_keys": ["sentence1", "sentence2"], "num_labels": 2},
    "agnews": {"glue_name": "ag_news", "text_keys": ["text"], "num_labels": 4},
}


def load_glue_task(task, model_name, max_length, train_batch_size, eval_batch_size):
    if task not in TASK_CONFIG:
        raise ValueError(f"Unsupported task: {task}")

    cfg = TASK_CONFIG[task]
    raw = load_dataset("glue", cfg["glue_name"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        args = [batch[k] for k in cfg["text_keys"]]
        return tokenizer(*args, truncation=True, max_length=max_length)

    remove_cols = cfg["text_keys"] + ["idx"]

    train_ds = raw["train"].map(tokenize, batched=True, remove_columns=remove_cols)
    dev_ds = raw["validation"].map(tokenize, batched=True, remove_columns=remove_cols)

    train_ds = train_ds.rename_column("label", "labels")
    dev_ds = dev_ds.rename_column("label", "labels")

    train_ds.set_format("torch")
    dev_ds.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=train_batch_size, shuffle=True, collate_fn=collator
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=eval_batch_size, shuffle=False, collate_fn=collator
    )

    return train_loader, dev_loader, cfg["num_labels"]
