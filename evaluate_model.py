import torch
from torch.utils.data import DataLoader

from Models.convnext_cnn import ConvNeXtCNN
from data_loader import get_data_v2, N_FOLDS
from train import evaluate

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyfaidx import Fasta

from data_loader import CELL_LINES, WindowDataset, get_split_chroms


def test_model(cell_line, fold, batch_size=64, max_batches=None):
    """We train separate models for each cell line rather than one model shared across cell lines.
    With five-fold cross-validation, that means five trained models per cell line per architecture—one for each chromosome split. 
    Across four cell lines, that's 20 training runs for one architecture.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Testing with device: {device}')
    model = ConvNeXtCNN().to(device)
    checkpoint = torch.load(
        f"checkpoints/{cell_line}_fold{fold}.pt",
        map_location=device,
        weights_only=True,
    )
    if "model_state_dict" not in checkpoint or "chromosomes" not in checkpoint:
        raise ValueError(
            "Checkpoint has no chromosome split metadata. Its training split must be "
            "verified before testing; train a new checkpoint to save that metadata."
        )
    if checkpoint.get("cell_line") != cell_line or checkpoint.get("fold") != fold:
        raise ValueError("Checkpoint cell line or fold does not match the requested test.")
    if checkpoint["chromosomes"] != get_split_chroms(fold):
        raise ValueError(
            "Checkpoint chromosome splits differ from the current data loader. "
            "Restore the original splits before testing."
        )
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Test chromosomes:", checkpoint["chromosomes"]["test"])
    test_data = get_data_v2(cell_line, fold, "test")
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    loss_fn = torch.nn.PoissonNLLLoss(log_input=False)
    loss, pearson = evaluate(model, test_loader, loss_fn, device, max_batches)

    print(f"Test loss: {loss:.4f}")
    print(f"Test Pearson r: {pearson:.4f}")
    return loss, pearson



def plot_peaks(cell_line, fold, peak_file, n_peaks=3):
    """Plot measured and predicted ATAC-seq signals side by side near peaks.

    Select the first n_peaks eligible summits on the checkpoint's held-out test
    chromosomes. Each prediction uses 2,048 bp of DNA centered on the summit;
    the plots show the central 1,024 bp in 4-bp bins. Signals use the loader's
    max-binning and log(1 + signal p-value) transformation, not raw read counts.

    Args:
        cell_line: Cell line associated with the checkpoint and signal bigWig.
        fold: Fold number associated with the checkpoint.
        peak_file: Path to the cell line's narrowPeak BED file (may be gzipped).
        n_peaks: Number of eligible peaks to plot.

    Returns:
        A matplotlib Figure, which can be saved using fig.savefig().

    These plots are qualitative examples, not a performance summary. Peak
    windows are not filtered by the test loader's blacklist or unmappability
    filters.
    """
    if n_peaks < 1:
        raise ValueError("n_peaks must be at least 1")

    checkpoint = torch.load(
        f"checkpoints/{cell_line}_fold{fold}.pt",
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("cell_line") != cell_line or checkpoint.get("fold") != fold:
        raise ValueError("Checkpoint does not match the requested cell line and fold")
    if checkpoint.get("chromosomes") != get_split_chroms(fold):
        raise ValueError("Checkpoint splits do not match the current data loader")

    model = ConvNeXtCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # narrowPeak column 10 gives the summit's offset from the peak start.
    peaks = pd.read_csv(peak_file, sep="\t", header=None, usecols=[0, 1, 2, 9])
    peaks.columns = ["chrom", "start", "end", "summit_offset"]
    peaks = peaks[
        peaks["chrom"].isin(checkpoint["chromosomes"]["test"])
        & (peaks["summit_offset"] >= 0)
        & (peaks["summit_offset"] < peaks["end"] - peaks["start"])
    ].copy()
    peaks["summit"] = peaks["start"] + peaks["summit_offset"]

    fasta_path = "data/training/hg38.fa"
    bigwig_path = f"data/training/{CELL_LINES[cell_line]}.bigWig"

    # Keep peaks with enough sequence on both sides for a 2,048-bp window.
    genome = Fasta(fasta_path)
    try:
        lengths = {chrom: len(genome[chrom]) for chrom in peaks["chrom"].unique()}
    finally:
        genome.close()

    peaks = peaks[
        (peaks["summit"] >= 1024)
        & (peaks["summit"] + 1024 <= peaks["chrom"].map(lengths))
    ].head(n_peaks)

    if len(peaks) < n_peaks:
        raise ValueError(f"Only {len(peaks)} eligible test peaks found")

    windows = [
        (row.chrom, int(row.summit) - 1024)
        for row in peaks.itertuples()
    ]
    dataset = WindowDataset(fasta_path, bigwig_path, windows)

    fig, axes = plt.subplots(n_peaks, 2, figsize=(12, 3 * n_peaks), squeeze=False)

    try:
        for i, row in enumerate(peaks.itertuples()):
            x, actual = dataset[i]

            with torch.no_grad():
                predicted = model(x.unsqueeze(0))[0]

            # Plot the central 1,024 bp: 256 bins of 4 bp each.
            actual = actual[128:384].numpy()
            predicted = predicted[128:384].numpy()
            positions = row.summit - 512 + np.arange(256) * 4 + 2

            axes[i, 0].plot(positions, actual, color="steelblue")
            axes[i, 1].plot(positions, predicted, color="darkorange")

            ymax = max(actual.max(), predicted.max(), 0.01)
            for ax in axes[i]:
                ax.set_ylim(0, ymax * 1.05)
                ax.set_xlabel(f"{row.chrom} position (bp)")
                ax.set_ylabel("log(1 + signal p-value)")

            axes[i, 0].set_title(f"Actual — {row.chrom}:{row.summit:,}")
            axes[i, 1].set_title(f"Predicted — {row.chrom}:{row.summit:,}")
    finally:
        if dataset._genome is not None:
            dataset._genome.close()
        if dataset._bw is not None:
            dataset._bw.close()

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    loss, pearson = test_model("GM12878", fold=0)
