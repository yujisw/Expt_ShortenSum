# Expt_ShortenSum

### Set up environment
#### 1. Install poetry
Run

```
curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
echo 'export PATH="$HOME/.poetry/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
poetry -V
```

*Our scripts are based on poetry.* If you do not want to use poetry, please modify `Makefile` as necessary.

#### 2. Install necessary packages
Run

```
make install
make setup-rouge
```

# Data Preprocess Instructions

## Download dataset

### A. CNN DailyMail
Download the `stories` directories from [here](http://cs.nyu.edu/~kcho/DMQA/) for *both CNN and Daily Mail* in `data` directory, and then unzip them with the commands below

```
tar -xvf data/cnn_stories.tgz -C data
tar -xvf data/dailymail_stories.tgz -C data
make format-cnndm
```

For each of the URL lists (`all_train.txt`, `all_val.txt` and `all_test.txt`), the corresponding stories are read from file and written to text files `train.source`, `train.target`, `val.source`, `val.target`, and `test.source` and `test.target`. These will be placed in the newly created `cnn_dm` directory.


### B. XSum

```
make download-and-format-xsum
```

The output is now suitable for feeding to the BPE preprocessing step of BART fine-tuning.


## Data Preprocess for Our Proposal

Run

```
make bpe-preprocess DATASET={cnn_dm or xsum}
```

# Training

## 1. Download　and process pre-trained weights (BART-large)
Run

```
wget -N -P data 'https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz'
tar -xvf data/bart.large.tar.gz -C data
make preprocess-proposal
```

Model | Description | # params | Download
---|---|---|---
`bart.base` | BART model with 6 encoder and decoder layers | 140M | [bart.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz)
`bart.large` | BART model with 12 encoder and decoder layers | 400M | [bart.large.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz)
`bart.large.mnli` | `bart.large` finetuned on `MNLI` | 400M | [bart.large.mnli.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.mnli.tar.gz)
`bart.large.cnn` | `bart.large` finetuned on `CNN-DM` | 400M | [bart.large.cnn.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.cnn.tar.gz)
`bart.large.xsum` | `bart.large` finetuned on `Xsum` | 400M | [bart.large.xsum.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.xsum.tar.gz)

## 2. Finetune baseline
Run

```
make finetune-baseline-large DATASET={cnn_dm or xsum}
```

The checkpoint files are saved in `output/finetune-baseline-large-YYYYMMDDHHMMSS`

## 3. Finetune proposal
Run

```
make finetune-proposal-large DATASET={cnn_dm or xsum} SUMTOPK_CONFIG={query_before_attention or after_attention}
```

The checkpoint file are saved as `output/finetune-baseline-large-YYYYMMDDHHMMSS/checkpoint_best.pt`.  


If you use normalization of SumTop-k output or Run the command below

```
make finetune-proposal-large-normalize DATASET={cnn_dm or xsum} SUMTOPK_CONFIG={query_before_attention or after_attention}
make generate-proposal DATASET={cnn_dm or xsum} BEAM_ARGS={least or least_xsum} TRAIN_DEST_DIR=[CHECKPOINT_DIR]
```

The checkpoint file are saved as `output/finetune-proposal-large-YYYYMMDDHHMMSS/checkpoint_best.pt`.  

## Generate summaries
### For baseline
Run

```
make generate-baseline DATASET={cnn_dm or xsum} BEAM_ARGS={cnn or xsum} TRAIN_DEST_DIR=[CHECKPOINT_DIR]
```

The generated summaries file is saved as `output/finetune-baseline-large-YYYYMMDDHHMMSS/test_{cnn_dm or xsum}.hypo_{cnn or xsum}_args`.

### For proposal

**Gold length**

Run

```
make generate-proposal DATASET={cnn_dm or xsum} BEAM_ARGS={least or least_xsum} TRAIN_DEST_DIR=[CHECKPOINT_DIR]
```

The generated summaries file is saved as `output/finetune-proposal-large-YYYYMMDDHHMMSS/test_{cnn_dm or xsum}.hypo_{least or least_xsum}_args`.


**With Fixed length**

Run

```
make generate-proposal-fixed-len DATASET={cnn_dm or xsum} BEAM_ARGS={least or least_xsum} TRAIN_DEST_DIR=[CHECKPOINT_DIR] FIXED_LENGTH={10 ~ 90}
```

The generated summaries file is saved as `output/finetune-proposal-large-YYYYMMDDHHMMSS/test_{cnn_dm or xsum}.hypo[FIXED_LENGTH]_{least or least_xsum}_args`.


# Evaluation

## ROUGE scores
Run

```
make calc-rouge DATASET={cnn_dm or xsum} BEAM_ARGS={least or least_xsum} TRAIN_DEST_DIR=[CHECKPOINT_DIR]
```

The result file is saved as `output/finetune-proposal-large-YYYYMMDDHHMMSS/test_{least or least_xsum}.result_{least or least_xsum}_args`.


## Overlap between top k tokens and generated summary
Run

```
make calc-faithful-score DATASET={cnn_dm or xsum} BEAM_ARGS={least or least_xsum} TRAIN_DEST_DIR=[CHECKPOINT_DIR]
```

The overall score is printed at the end of log

# Reranking token scores
Run

```
make enlighten-missed-tokens DATASET={cnn_dm or xsum} BEAM_ARGS={least or least_xsum} TRAIN_DEST_DIR=[CHECKPOINT_DIR]
```
