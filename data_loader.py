
from pyfaidx import Fasta
import pandas as pd
import numpy as np
import pybigtools
from pyfaidx import Fasta
from pathlib import Path


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

accessions = ['ENCFF262URW', 'ENCFF262URW', 'ENCFF667MDI', 'ENCFF770EAV']
fasta_path = data_dir / "hg38.fa"
bigwig_path = data_dir / f"{accessions[0]}.bigWig"

for win, x, y in get_data(fasta_path, bigwig_path, start=1_000_000, stop=1_004_096, blacklist_file=blacklist_file, unmappable_file=unmappable_file):
# for win, x, y in get_data(accession, blacklist_file=blacklist_file, unmappable_file=unmappable_file):
    pass
    print(f'window: {win}')
    print(f'x.shape: {x.shape}\n{x}')
    print(f'Y: {y.shape}\n{y}')
    print('---------------\n')
    # We will reduce the X from 2048 to 512 inside the model with ConvNet/pooling