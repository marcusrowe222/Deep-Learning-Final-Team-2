"""Quick training loop: Poisson NLL loss + AdamW (as in Grover et al.), one fold or full 5-fold CV."""
import copy
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from Models.convnext_cnn import ConvNeXtCNN
from data_loader import get_data_v2, N_FOLDS

CENTER = slice(128, 384)  # central 256 of the 512 bins = central 1,024 bp


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, max_batches=None):
    """Mean loss and Pearson r (over all central-256 bins, pooled across windows)."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    cnt = sx = sy = sxx = syy = sxy = 0.0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += loss_fn(pred, y).item()
        n_batches += 1
        p = pred[:, CENTER].double().flatten()
        t = y[:, CENTER].double().flatten()
        cnt += p.numel()
        sx += p.sum().item(); sy += t.sum().item()
        sxx += (p * p).sum().item(); syy += (t * t).sum().item(); sxy += (p * t).sum().item()
    denom = math.sqrt(max(cnt * sxx - sx ** 2, 0) * max(cnt * syy - sy ** 2, 0))
    pearson = (cnt * sxy - sx * sy) / denom if denom > 0 else float("nan")
    return total_loss / max(n_batches, 1), pearson


def run_fold(cell_line, fold, epochs=10, batch_size=64, lr=1e-3, weight_decay=1e-2,
             patience=3, num_workers=0, max_batches=None, data_kwargs=None,
             ckpt_dir="checkpoints", seed=0, log_every=100):
    """Train a fresh model on one fold. Returns a dict of results.

    max_batches: cap batches per epoch / eval pass (use a small number for a quick smoke test).
    data_kwargs: extra arguments for get_data_v2 (e.g. data_dir=...).
    """
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_kwargs = data_kwargs or {}

    train_ds = get_data_v2(cell_line, fold, "train", **data_kwargs)
    val_ds = get_data_v2(cell_line, fold, "val", **data_kwargs)
    test_ds = get_data_v2(cell_line, fold, "test", **data_kwargs)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    model = ConvNeXtCNN().to(device)                      # fresh weights for every fold
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # the model ends in Softplus, so its output is already a rate (not a log-rate)
    loss_fn = torch.nn.PoissonNLLLoss(log_input=False)

    best_val, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        t0, running = time.time(), 0.0
        for step, (x, y) in enumerate(train_dl):
            if max_batches is not None and step >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            pred = model(x)
            assert pred.shape == y.shape, f"model output {tuple(pred.shape)} != target {tuple(y.shape)}"
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            running += loss.item()
            if (step + 1) % log_every == 0:
                print(f"  fold {fold} epoch {epoch} step {step + 1}: train loss {running / log_every:.4f}")
                running = 0.0

        val_loss, val_r = evaluate(model, val_dl, loss_fn, device, max_batches)
        print(f"fold {fold} epoch {epoch}: val loss {val_loss:.4f}  val Pearson {val_r:.4f}  ({time.time() - t0:.0f}s)")

        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(best_state, os.path.join(ckpt_dir, f"{cell_line}_fold{fold}.pt"))
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"fold {fold}: early stop (no val improvement for {patience} epochs)")
                break

    model.load_state_dict(best_state)                     # test once, with the best-validation weights
    test_loss, test_r = evaluate(model, test_dl, loss_fn, device, max_batches)
    print(f"fold {fold} TEST: loss {test_loss:.4f}  Pearson {test_r:.4f}")
    return {"fold": fold, "best_val_loss": best_val, "test_loss": test_loss, "test_pearson": test_r}


if __name__ == "__main__":                                # required on Windows when num_workers > 0
    # One fold:
    results = [run_fold("GM12878", fold=0)]

    # Full 5-fold cross validation (uncomment, and comment out the line above):
    # results = [run_fold("GM12878", fold) for fold in range(N_FOLDS)]

    for r in results:
        print(r)
