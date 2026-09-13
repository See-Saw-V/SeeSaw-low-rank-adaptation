import argparse
import random

import numpy as np
import torch
import yaml
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

from src.controller.seesaw_controller import SeesawController
from src.data.glue_loader import load_glue_task
from src.evaluate import evaluate
from src.models.model_utils import insert_lora_adapters, make_optimizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, train_loader, optimizer, scheduler, controller, device, grad_clip, task):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["labels"] = batch["labels"].long() if task != "stsb" else batch["labels"].float()

        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        if controller is not None:
            controller.accumulate_batch()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = "deberta" if "deberta" in config["model"]["name"] else "bert"

    train_loader, dev_loader, num_labels = load_glue_task(
        task=config["data"]["task"],
        model_name=config["model"]["name"],
        max_length=config["data"]["max_length"],
        train_batch_size=config["data"]["train_batch_size"],
        eval_batch_size=config["data"]["eval_batch_size"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model"]["name"], num_labels=num_labels
    )
    model = model.float().to(device)

    adapters = insert_lora_adapters(
        model,
        backbone=backbone,
        target_projections=config["model"]["target_modules"],
        total_rank=config["lora"]["target_total_rank"],
        alpha=config["lora"]["alpha"],
    )
    model = model.float().to(device)

    optimizer = make_optimizer(
        model, config["training"]["lora_lr"], config["training"]["weight_decay"]
    )

    total_steps = config["training"]["num_epochs"] * len(train_loader)
    warmup_steps = int(config["training"]["warmup_ratio"] * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    controller = None
    if config["controller"]["enabled"]:
        controller = SeesawController(
            adapters,
            aggregation=config["controller"]["aggregation"],
            ema_decay=config["controller"].get("ema_decay") or 0.85,
            normalization=config["controller"]["normalization"],
            importance_gap=config["controller"]["importance_gap"],
            min_rank=config["lora"]["min_rank"],
            max_rank=config["lora"]["max_rank"],
            target_total_rank=config["lora"]["target_total_rank"],
        )

    best_metric = -float("inf")
    task = config["data"]["task"]
    primary_metric = "mcc" if task == "cola" else ("pearson" if task == "stsb" else "accuracy")

    for epoch in range(1, config["training"]["num_epochs"] + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, controller,
            device, config["training"]["grad_clip"], task,
        )

        if controller is not None and controller.aggregation == "epoch_average":
            controller.end_epoch()

        metrics = evaluate(model, dev_loader, device, task=task)

        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        print(f"Train Loss : {train_loss:.4f}")
        for k, v in metrics.items():
            print(f"{k:12s}: {v:.4f}")

        if metrics[primary_metric] > best_metric:
            best_metric = metrics[primary_metric]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_metric": best_metric,
                    "ranks": {k: a.rank for k, a in adapters.items()},
                },
                config["checkpoint"]["best_path"],
            )
            print("Best checkpoint saved")

        if (
            controller is not None
            and epoch >= config["controller"]["start_epoch"]
            and (epoch - config["controller"]["start_epoch"]) % config["controller"]["interval"] == 0
            and epoch < config["training"]["num_epochs"]
        ):
            result = controller.step(
                lambda: make_optimizer(
                    model, config["training"]["lora_lr"], config["training"]["weight_decay"]
                )
            )
            if result and result["status"] == "transfer":
                optimizer = result["optimizer"]
                print(
                    f"Controller: {result['donor']} {result['donor_old_rank']}->"
                    f"{result['donor_new_rank']} | {result['receiver']} "
                    f"{result['receiver_old_rank']}->{result['receiver_new_rank']}"
                )
                with open(config["checkpoint"]["transfer_log"], "a") as f:
                    f.write(f"epoch={epoch} {result}\n")
            elif result:
                print(f"Controller: {result['status']}")

    print(f"\nTraining complete. Best {primary_metric}: {best_metric:.4f}")
    if controller is not None:
        print(f"Total transfers: {controller.transfer_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
