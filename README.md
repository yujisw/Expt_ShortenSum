# Expt_ShortenSum

# Data Preprocess Instructions

## General Data Preprocess

### 0. Install poetry
Run

```
curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
echo 'export PATH="$HOME/.poetry/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
poetry -V
```

*Our scripts are based on poetry.* If you do not want to use poetry, please modify `Makefile` as necessary.

### 1. Download data
Download the `stories` directories from [here](http://cs.nyu.edu/~kcho/DMQA/) for *both CNN and Daily Mail* in `data` directory, and then unzip them with the commands below

```
tar -xvf data/cnn_stories.tgz -C data
tar -xvf data/dailymail_stories.tgz -C data
```

### 2. Process into .source and .target files
Run

```
python data_utils/make_datafiles.py /path/to/cnn/stories /path/to/dailymail/stories /path/to/data/dir
```

replacing `/path/to/cnn/stories` with the path to where you saved the `cnn/stories` directory that you downloaded; similarly for `dailymail/stories`.

For each of the URL lists (`all_train.txt`, `all_val.txt` and `all_test.txt`), the corresponding stories are read from file and written to text files `train.source`, `train.target`, `val.source`, `val.target`, and `test.source` and `test.target`. These will be placed in the newly created `cnn_dm` directory.

The output is now suitable for feeding to the BPE preprocessing step of BART fine-tuning.

## Data Preprocess for Our Proposal

### 1. Split by length of summaries
Run
<!-- TODO -->
```
python hogehoge.py /path/to/input/data /path/to/output/data
```

## Training

### 1. Download pre-trained models (BART-large)
Run

```
wget -N -P data 'https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz'
tar -xvf data/bart.large.tar.gz -C data
```

Model | Description | # params | Download
---|---|---|---
`bart.base` | BART model with 6 encoder and decoder layers | 140M | [bart.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz)
`bart.large` | BART model with 12 encoder and decoder layers | 400M | [bart.large.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz)
`bart.large.mnli` | `bart.large` finetuned on `MNLI` | 400M | [bart.large.mnli.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz)
`bart.large.cnn` | `bart.large` finetuned on `CNN-DM` | 400M | [bart.large.cnn.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz)
`bart.large.xsum` | `bart.large` finetuned on `Xsum` | 400M | [bart.large.xsum.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz)

### 2. Finetune baseline
Run

```
make finetune-baseline
```
