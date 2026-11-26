
from pyfaidx import Fasta
import pandas as pd
import numpy as np
import pybigtools
from pathlib import Path
import random

CHROMS = [f"chr{i}" for i in range(1, 23)]
N_FOLDS = 5
FOLD_SEED = 42  

def make_folds(seed: int = FOLD_SEED, n_folds: int = N_FOLDS) -> list[list[str]]:
    """Randomly deal chr1-chr22 into n_folds groups (sizes 5,5,4,4,4)."""
    shuffled = CHROMS.copy()
    random.Random(seed).shuffle(shuffled)
    return [sorted(shuffled[i::n_folds], key=lambda c: int(c[3:])) for i in range(n_folds)]

FOLDS = make_folds()

def get_split_chroms(fold: int, val_offset: int = 1) -> dict[str, list[str]]:
    """Chromosomes for one fold: test = fold, val = the next fold, train = the other three."""
    if not 0 <= fold < N_FOLDS:
        raise ValueError(f"fold must be in 0..{N_FOLDS - 1}, got {fold}")
    test_idx = fold
    val_idx = (fold + val_offset) % N_FOLDS
    train_idx = [i for i in range(N_FOLDS) if i not in (test_idx, val_idx)]
    return {
        "train": [c for i in train_idx for c in FOLDS[i]],
        "val": FOLDS[val_idx],
        "test": FOLDS[test_idx],
    }

def all_fold_splits():
    """Yield (fold, split_dict) for full 5-fold cross validation."""
    for fold in range(N_FOLDS):
        yield fold, get_split_chroms(fold)

def get_training_datasets(reference_genome: Fasta, train_chroms: list[int]):
    print(reference_genome.keys())

    autosomes = [f"chr{i}" for i in train_chroms]

    for chrom in autosomes:
        print(chrom, len(reference_genome[chrom]))

def one_hot_encode_base_pairs(sequence: list[str]) -> list[list[int]]:
    """sequence like ["A", "C", "G", "T"]
    Return example: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    """
    bases = np.array(list(sequence))
    possible_bases = np.array(["A", "G", "C", "T"])

    # return 1-D array into column to compare with possible_bases array
    one_hot = (bases[:, None] == possible_bases).astype(np.uint8)
    return one_hot

def get_blacklist_file(blacklist_file: str, chrom):
    bed = pd.read_csv(blacklist_file, delimiter='\t', header=None, names=['chr', 'start', 'end'], usecols=[0, 1, 2])
    return bed[bed["chr"] == chrom]

def overlaps_blacklist(blacklist, window_start, window_end):
    overlaps = (
        (blacklist["start"] <= window_end)
        & (blacklist["end"] >= window_start)
    )
    return overlaps.any()

def too_unmappable(unmappable, window_start, window_end, threshold=0.35):

    # get all the overlaps inside the window for every overlap in file
    overlaps = unmappable[
        (unmappable["start"] < window_end)
        & (unmappable["end"] > window_start)
    ].sort_values("start")

    total = 0
    covered_until = window_start

    # Count number of overlaps
    for interval in overlaps.itertuples():
        start = max(interval.start, window_start, covered_until)
        end = min(interval.end, window_end)

        if end > start:
            total += end - start
            covered_until = end

    return total > (threshold * (window_end - window_start))


def get_data(fasta_path, bigwig_path, chrom="chr1", start=0, stop=None, blacklist_file=None, unmappable_file=None):
    window_size = 2048

    genome = Fasta(str(fasta_path))
    bw = pybigtools.open(bigwig_path)

    if chrom not in genome or chrom not in bw.chroms():
        print(f"{chrom} is not present in both files")

    chrom_length = len(genome[chrom])
    if bw.chroms()[chrom] != chrom_length:
        print(f"Chromosome lengths don't match for {chrom}")

    if stop is None:
        stop = chrom_length

    if blacklist_file:
        blacklist = get_blacklist_file(blacklist_file, chrom)
    if unmappable_file:
        unmappable = get_blacklist_file(unmappable_file, chrom)

    # loop over whole length jumping in intervals of 'window_size'
    for window_start in range(start, stop - window_size + 1, window_size):
        window_end = window_start + window_size
        if i := overlaps_blacklist(blacklist, window_start, window_end):
            # print(f"overlaps_blacklist: {i}")
            continue
        if i := too_unmappable(unmappable, window_start, window_end):
            # print(f"too_unmappable: {i}")
            continue

        # X: DNA sequence in ASAP's A, G, C, T column order.
        sequence = genome[chrom][window_start:window_end].seq.upper()
        x = one_hot_encode_base_pairs(sequence)

        # Y: replace missing signal with zero, then max-bin and log.
        atac_signal = bw.values(chrom, window_start, window_end, fillna=0)
        
        atac_signal = np.asarray(atac_signal, dtype=np.float32)
        # print(f'signal: {atac_signal.shape}')

        # rearrange the 2048 values into 512 rows of 4 columns each
        # also maximum binning to reduce the signal to one value per 4-base pair region
        y = atac_signal.reshape(512, 4).max(axis=1)
        # Atac-seq signals need to be normalized so they are not too large for training

        y = np.log1p(y)

        yield window_start, x, y

    genome.close()
    bw.close()

data_dir = Path("data/training")
bl_folder = 'data/blacklist'
blacklist_file = f'{bl_folder}/hg38.blacklist.rep.bed'
unmappable_file = f'{bl_folder}/umap_k36_t10_l32_hg38.bed'

accessions = ['ENCFF262URW', 'ENCFF357GNC', 'ENCFF667MDI', 'ENCFF770EAV']
fasta_path = data_dir / "hg38.fa"
bigwig_path = data_dir / f"{accessions[0]}.bigWig"

# for win, x, y in get_data(fasta_path, bigwig_path, start=1_000_000, stop=1_004_096, blacklist_file=blacklist_file, unmappable_file=unmappable_file):
# # for win, x, y in get_data(accession, blacklist_file=blacklist_file, unmappable_file=unmappable_file):
#     pass
#     print(f'window: {win}')
#     print(f'x.shape: {x.shape}\n{x}')
#     print(f'Y: {y.shape}\n{y}')
#     print('---------------\n')
#     # We will reduce the X from 2048 to 512 inside the model with ConvNet/pooling





# =====================================================================
# get_data_v2: fold-aware, torch-ready replacement for get_data
# =====================================================================
import torch
from torch.utils.data import Dataset

WINDOW_SIZE = 2048
BIN_SIZE = 4
N_BINS = WINDOW_SIZE // BIN_SIZE  # 512

CELL_LINES = {
    "GM12878": "ENCFF667MDI",
    "K562": "ENCFF357GNC",
    "IMR-90": "ENCFF770EAV",
    "HepG2": "ENCFF262URW",
}

# byte -> one-hot row, column order A, G, C, T (anything else, e.g. N, stays all zeros)
_ONE_HOT_LUT = np.zeros((256, 4), dtype=np.float32)
for _col, _base in enumerate("AGCT"):
    _ONE_HOT_LUT[ord(_base), _col] = 1.0


def _load_intervals(bed_path):
    """Read a BED file once into {chrom: (starts, ends)} with overlapping intervals merged."""
    bed = pd.read_csv(bed_path, sep="\t", header=None, usecols=[0, 1, 2], names=["chr", "start", "end"])
    out = {}
    for chrom, grp in bed.groupby("chr"):
        grp = grp.sort_values("start")
        starts, ends = grp["start"].to_numpy(np.int64), grp["end"].to_numpy(np.int64)
        merged_s, merged_e = [starts[0]], [ends[0]]
        for s, e in zip(starts[1:], ends[1:]):
            if s <= merged_e[-1]:
                merged_e[-1] = max(merged_e[-1], e)
            else:
                merged_s.append(s)
                merged_e.append(e)
        out[chrom] = (np.array(merged_s, np.int64), np.array(merged_e, np.int64))
    return out


def _covered_before(x, s, e):
    """Total length covered by the disjoint sorted intervals (s, e) in [0, x), for an array of x."""
    if len(s) == 0:
        return np.zeros_like(x)
    cum = np.concatenate([[0], np.cumsum(e - s)])
    k = np.searchsorted(s, x, side="right")  # intervals with start <= x
    k_prev = np.maximum(k - 1, 0)
    partial = np.clip(np.minimum(x, e[k_prev]) - s[k_prev], 0, None)
    return np.where(k > 0, cum[k_prev] + partial, 0)


def _valid_window_starts(chrom_len, blacklist, unmappable, stride=WINDOW_SIZE, max_unmappable=0.35):
    """All window starts on one chromosome that avoid the blacklist and are mappable enough."""
    starts = np.arange(0, chrom_len - WINDOW_SIZE + 1, stride, dtype=np.int64)
    ends = starts + WINDOW_SIZE
    keep = np.ones(len(starts), dtype=bool)

    if blacklist is not None:
        bs, be = blacklist
        # last blacklist interval that starts before the window ends; overlap if it ends after the window starts
        idx = np.searchsorted(bs, ends, side="left") - 1
        has = idx >= 0
        overlap = np.zeros(len(starts), dtype=bool)
        overlap[has] = be[idx[has]] > starts[has]
        keep &= ~overlap

    if unmappable is not None:
        us, ue = unmappable
        covered = _covered_before(ends, us, ue) - _covered_before(starts, us, ue)
        keep &= covered <= max_unmappable * WINDOW_SIZE

    return starts[keep]


class WindowDataset(Dataset):
    """One item = (x, y).  x: (4, 2048) float32 one-hot (A,G,C,T).  y: (512,) float32 log1p of the 4-bp max signal."""

    def __init__(self, fasta_path, bigwig_path, windows, augment=False, jitter=0, crop_target=False):
        self.fasta_path = str(fasta_path)
        self.bigwig_path = str(bigwig_path)
        self.windows = list(windows)  # [(chrom, start), ...]
        self.augment = augment
        self.jitter = jitter
        self.crop_target = crop_target
        self._genome = None  # opened lazily, so every DataLoader worker gets its own handles
        self._bw = None
        self._chrom_len = None

    def __len__(self):
        return len(self.windows)

    def _open(self):
        self._genome = Fasta(self.fasta_path)
        self._bw = pybigtools.open(self.bigwig_path)
        self._chrom_len = {c: len(self._genome[c]) for c in {c for c, _ in self.windows}}

    def __getitem__(self, i):
        if self._genome is None:
            self._open()
        chrom, start = self.windows[i]

        if self.augment and self.jitter:
            start += int(np.random.randint(-self.jitter, self.jitter + 1))
            start = min(max(start, 0), self._chrom_len[chrom] - WINDOW_SIZE)
        end = start + WINDOW_SIZE

        seq = self._genome[chrom][start:end].seq.upper()
        x = _ONE_HOT_LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)].T  # (4, 2048)

        signal = np.asarray(self._bw.values(chrom, start, end, fillna=0), dtype=np.float32)
        y = np.log1p(signal.reshape(N_BINS, BIN_SIZE).max(axis=1))

        if self.augment and np.random.rand() < 0.5:
            # reverse complement: with channel order A,G,C,T the complement is a channel flip
            x = x[::-1, ::-1]
            y = y[::-1]

        if self.crop_target:  # central 1,024 bp = central 256 bins
            y = y[N_BINS // 4: 3 * N_BINS // 4]

        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def get_data_v2(
    cell_line,
    fold,
    split,
    data_dir="data/training",
    blacklist_file="data/blacklist/hg38.blacklist.rep.bed",
    unmappable_file="data/blacklist/umap_k36_t10_l32_hg38.bed",
    fasta_name="hg38.fa",
    augment=None,
    jitter=0,
    crop_target=False,
    stride=WINDOW_SIZE,
    max_unmappable=0.35,
    verbose=True,
):
    """Build the Dataset for one cell line / one fold / one split ('train', 'val' or 'test').

    augment defaults to True for 'train' and False otherwise.
    For full cross-validation call this inside `for fold in range(N_FOLDS)`.
    """
    if cell_line not in CELL_LINES:
        raise ValueError(f"cell_line must be one of {list(CELL_LINES)}, got {cell_line!r}")
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")
    if augment is None:
        augment = split == "train"

    data_dir = Path(data_dir)
    fasta_path = data_dir / fasta_name
    bigwig_path = data_dir / f"{CELL_LINES[cell_line]}.bigWig"
    for p in (fasta_path, bigwig_path):
        if not Path(p).exists():
            raise FileNotFoundError(f"{p} not found (did you extract hg38.fa.gz?)")

    chroms = get_split_chroms(fold)[split]

    blacklist = _load_intervals(blacklist_file) if blacklist_file else {}
    unmappable = _load_intervals(unmappable_file) if unmappable_file else {}

    genome = Fasta(str(fasta_path))
    bw = pybigtools.open(str(bigwig_path))
    bw_chroms = dict(bw.chroms())
    windows = []
    try:
        for chrom in chroms:
            if chrom not in genome or chrom not in bw_chroms:
                raise ValueError(f"{chrom} missing from the FASTA or the bigWig")
            if len(genome[chrom]) != bw_chroms[chrom]:
                raise ValueError(f"{chrom} length differs: FASTA={len(genome[chrom])}, bigWig={bw_chroms[chrom]}")
            starts = _valid_window_starts(
                len(genome[chrom]),
                blacklist.get(chrom) if blacklist_file else None,
                unmappable.get(chrom) if unmappable_file else None,
                stride=stride,
                max_unmappable=max_unmappable,
            )
            windows.extend((chrom, int(s)) for s in starts)
            if verbose:
                total = (len(genome[chrom]) - WINDOW_SIZE) // stride + 1
                print(f"{cell_line} fold {fold} {split}: {chrom}: kept {len(starts):,} of {total:,} windows")
    finally:
        genome.close()
        bw.close()

    if verbose:
        print(f"{cell_line} fold {fold} {split}: {len(windows):,} windows total")
    return WindowDataset(fasta_path, bigwig_path, windows, augment=augment, jitter=jitter, crop_target=crop_target)
