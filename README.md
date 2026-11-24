## Cell-line ATAC-seq data

These files are listed in Supplemental Table S1 of Grover et al.
All use the GRCh38 genome assembly. The bigWigs contain signal
p-values; the BED files contain pseudoreplicated peaks.
 - Click on the bigWig link and click "Download ENCFF667MDI" to download to your computer

| Cell line | ENCODE experiment | Signal bigWig | Peak BED |
|---|---|---|---|
| GM12878 | [ENCSR637XSC](https://www.encodeproject.org/experiments/ENCSR637XSC/) | [ENCFF667MDI](https://www.encodeproject.org/files/ENCFF667MDI/) | [ENCFF748UZH](https://www.encodeproject.org/files/ENCFF748UZH/) |
| K562 | [ENCSR868FGK](https://www.encodeproject.org/experiments/ENCSR868FGK/) | [ENCFF357GNC](https://www.encodeproject.org/files/ENCFF357GNC/) | [ENCFF333TAT](https://www.encodeproject.org/files/ENCFF333TAT/) |
| IMR-90 | [ENCSR200OML](https://www.encodeproject.org/experiments/ENCSR200OML/) | [ENCFF770EAV](https://www.encodeproject.org/files/ENCFF770EAV/) | [ENCFF243NTP](https://www.encodeproject.org/files/ENCFF243NTP/) |
| HepG2 | [ENCSR291GJU](https://www.encodeproject.org/experiments/ENCSR291GJU/) | [ENCFF262URW](https://www.encodeproject.org/files/ENCFF262URW/) | [ENCFF439EIO](https://www.encodeproject.org/files/ENCFF439EIO/) |

### How the files are used

- **bigWig:** Provides the numerical target values the model learns to predict from DNA sequence.
- **BED:** Identifies accessible regions for exploring the data and evaluating predictions around peaks.
- **Reference genome:** Provides the matching GRCh38 DNA sequence used as the model's input.

### Reference genome

The [UCSC hg38 FASTA](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/initial/hg38.fa.gz) is a suitable starting point (about 938 MB compressed). It contains the initial GRCh38 assembly and uses chromosome names such as `chr1`. The sequence is soft-masked: lowercase bases represent repeats, not missing DNA. See the [UCSC description](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/) for details.

The authors' [training example](https://github.com/BoevaLab/ASAP/blob/main/tutorials/train.py) refers to an uncompressed `hg38.fa`. The exact FASTA file or patch version used in the paper has not been verified. Before processing, check that chromosome names and lengths match the bigWig files.