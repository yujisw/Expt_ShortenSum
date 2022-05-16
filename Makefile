POETRY_RUN := poetry run

TASK=data/cnn_dm
TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=0
LR=3e-03
MAX_TOKENS=2048
UPDATE_FREQ=128
PRETRAINED_BASE_PATH=data/bart.base/model.pt
PRETRAINED_LARGE_PATH=data/bart.large/model.pt

# Output data path
DATE_INFO := $(shell date +'%Y%m%d%H%M%S')
OUTPUT_DIR := output
OUTPUT_DIR_PREFIX := finetune-baseline-large
TRAIN_DEST_DIR = ${OUTPUT_DIR}/${OUTPUT_DIR_PREFIX}_${DATE_INFO}
# TENSORBOARD_LOG_DIR = tensorboard/${OUTPUT_DIR_PREFIX}_${DATE_INFO}
LOG_FILE_PATH = log/${OUTPUT_DIR_PREFIX}_${DATE_INFO}.log

# Command Setting
CUDA_USE_DEVICES := 0

notebook:
	${POETRY_RUN} jupyter lab

bpe-preprocess:
	python data_utils/make_datafiles.py data/cnn/stories data/dailymail/stories data
	@echo BPE preprocess
	wget -N -P temp 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
	wget -N -P temp 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
	wget -N -P temp 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
	bash data_utils/bpe_preprocess.sh
	@echo Binarize dataset
	${POETRY_RUN} fairseq-preprocess \
		--source-lang "source" \
		--target-lang "target" \
		--trainpref "${TASK}/train.bpe" \
		--validpref "${TASK}/val.bpe" \
		--testpref "${TASK}/test.bpe" \
		--destdir "${TASK}-bin/" \
		--workers 60 \
		--srcdict data/dict.txt \
		--tgtdict data/dict.txt;

install:
	poetry install
	@echo Installing the correct version for your environment
	${POETRY_RUN} pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

finetune-baseline-large:
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} python train_fairseq/train.py data/cnn_dm-bin \
		--save-dir ${TRAIN_DEST_DIR} \
		--log-format simple \
		--restore-file ${PRETRAINED_LARGE_PATH} \
		--max-tokens ${MAX_TOKENS} \
		--task translation \
		--source-lang source --target-lang target \
		--truncate-source \
		--layernorm-embedding \
		--share-all-embeddings \
		--share-decoder-input-output-embed \
		--reset-optimizer --reset-dataloader --reset-meters \
		--required-batch-size-multiple 1 \
		--arch bart_large \
		--criterion label_smoothed_cross_entropy \
		--label-smoothing 0.1 \
		--dropout 0.1 --attention-dropout 0.1 \
		--weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
		--clip-norm 0.1 \
		--lr-scheduler polynomial_decay --lr ${LR} --total-num-update ${TOTAL_NUM_UPDATES} --warmup-updates ${WARMUP_UPDATES} \
		--fp16 --update-freq ${UPDATE_FREQ} \
		--skip-invalid-size-inputs-valid-test \
		--find-unused-parameters --save-interval-updates 10000 \
		> ${LOG_FILE_PATH};

finetune-proposal-large:
	${eval OUTPUT_DIR_PREFIX := finetune-proposal-large}
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} python train_fairseq/train.py data/cnn_dm-bin \
		--save-dir ${TRAIN_DEST_DIR} \
		--log-format simple \
		--restore-file ${PRETRAINED_LARGE_PATH} \
		--max-tokens ${MAX_TOKENS} \
		--task translation \
		--source-lang source --target-lang target \
		--truncate-source \
		--layernorm-embedding \
		--share-all-embeddings \
		--share-decoder-input-output-embed \
		--reset-optimizer --reset-dataloader --reset-meters \
		--required-batch-size-multiple 1 \
		--user-dir train_fairseq \
		--arch proposed_model_large \
		--criterion label_smoothed_cross_entropy \
		--label-smoothing 0.1 \
		--dropout 0.1 --attention-dropout 0.1 \
		--weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
		--clip-norm 0.1 \
		--lr-scheduler polynomial_decay --lr ${LR} --total-num-update ${TOTAL_NUM_UPDATES} --warmup-updates ${WARMUP_UPDATES} \
		--fp16 --update-freq ${UPDATE_FREQ} \
		--skip-invalid-size-inputs-valid-test \
		--find-unused-parameters \
		--validate-interval-updates 1000 \
		--extract-num 256 --use-wandb \
		> ${LOG_FILE_PATH};

# Usage: make generate-baseline TRAIN_DEST_DIR=hogehoge
generate-baseline:
	${eval OUTPUT_DIR_PREFIX := generate-baseline}
	cp data/cnn_dm-bin/dict.source.txt ${TRAIN_DEST_DIR}/
	cp data/cnn_dm-bin/dict.target.txt ${TRAIN_DEST_DIR}/
	CUDA_VISIBLE_DEVICES=1 ${POETRY_RUN} python train_fairseq/generate.py \
		--model-dir ${TRAIN_DEST_DIR} \
		--model-file checkpoint_best.pt \
		--src data/cnn_dm/test.source \
		--out ${TRAIN_DEST_DIR}/test.hypo

# Usage: make generate-proposal TRAIN_DEST_DIR=hogehoge
generate-proposal:
	${eval OUTPUT_DIR_PREFIX := generate-proposal}
	cp data/cnn_dm-bin/dict.source.txt ${TRAIN_DEST_DIR}/
	cp data/cnn_dm-bin/dict.target.txt ${TRAIN_DEST_DIR}/
	CUDA_VISIBLE_DEVICES=1 ${POETRY_RUN} python train_fairseq/generate.py --use-proposal \
		--model-dir ${TRAIN_DEST_DIR} \
		--model-file checkpoint_best.pt \
		--src data/cnn_dm/test.source \
		--out ${TRAIN_DEST_DIR}/test.hypo

params-tune-proposal-large:
	${eval OUTPUT_DIR_PREFIX := params-tune-proposal-large}
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} python train_fairseq/params_tuner.py data/cnn_dm-bin \
		--save-dir ${TRAIN_DEST_DIR} \
		--log-format simple \
		--restore-file ${PRETRAINED_LARGE_PATH} \
		--max-tokens ${MAX_TOKENS} \
		--task translation \
		--source-lang source --target-lang target \
		--truncate-source \
		--layernorm-embedding \
		--share-all-embeddings \
		--share-decoder-input-output-embed \
		--reset-optimizer --reset-dataloader --reset-meters \
		--required-batch-size-multiple 1 \
		--user-dir train_fairseq \
		--arch proposed_model_large \
		--criterion label_smoothed_cross_entropy \
		--label-smoothing 0.1 \
		--dropout 0.1 --attention-dropout 0.1 \
		--weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
		--clip-norm 0.1 \
		--lr-scheduler polynomial_decay --lr ${LR} --total-num-update ${TOTAL_NUM_UPDATES} --warmup-updates ${WARMUP_UPDATES} \
		--fp16 --update-freq ${UPDATE_FREQ} \
		--skip-invalid-size-inputs-valid-test \
		--find-unused-parameters --save-interval-updates 1000 \
		> ${LOG_FILE_PATH};

# DO NOT USE THIS COMMAND DURING TRAINING
remove-empty-output-dir:
	find ${OUTPUT_DIR} -type d -empty | xargs rm -r