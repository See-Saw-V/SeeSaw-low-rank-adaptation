import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_recall_fscore_support
from scipy.stats import pearsonr, spearmanr


@torch.no_grad()
def evaluate(model, dev_loader, device, task="cola"):
    model.eval()

    predictions, labels = [], []
    total_loss = 0.0

    for batch in dev_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["labels"] = batch["labels"].long() if task != "stsb" else batch["labels"].float()

        outputs = model(**batch)
        total_loss += outputs.loss.item()

        if task == "stsb":
            preds = outputs.logits.squeeze(-1)
        else:
            preds = outputs.logits.argmax(dim=-1)

        predictions.extend(preds.cpu().numpy().tolist())
        labels.extend(batch["labels"].cpu().numpy().tolist())

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    avg_loss = total_loss / len(dev_loader)

    if task == "cola":
        mcc = matthews_corrcoef(labels, predictions)
        accuracy = accuracy_score(labels, predictions)
        return {"loss": avg_loss, "mcc": mcc, "accuracy": accuracy}

    elif task == "stsb":
        pearson = pearsonr(labels, predictions)[0]
        spearman = spearmanr(labels, predictions)[0]
        return {"loss": avg_loss, "pearson": pearson, "spearman": spearman}

    else:
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary" if len(set(labels)) == 2 else "macro"
        )
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
