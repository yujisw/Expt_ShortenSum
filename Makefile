POETRY_RUN := poetry run

TASK=data/cnn_dm
INPUT_DATA_DIR=data/cnn_dm-bin
TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=16
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=128
PRETRAINED_BASE_PATH=data/bart.base/model.pt
PRETRAINED_LARGE_PATH=data/bart.large/model.pt
PRETRAINED_LARGE_PATH_FOR_EXTRACTOR=data/bart.with.extractor.large/model.pt
PRETRAINED_LARGE_CNN_PATH=data/bart.large.cnn/model.pt

# Output data path
DATE_INFO := $(shell date +'%Y%m%d%H%M%S')
OUTPUT_DIR := output
OUTPUT_DIR_PREFIX := finetune-baseline-large
TRAIN_DEST_DIR = ${OUTPUT_DIR}/${OUTPUT_DIR_PREFIX}_${DATE_INFO}
# TENSORBOARD_LOG_DIR = tensorboard/${OUTPUT_DIR_PREFIX}_${DATE_INFO}
LOG_FILE_PATH = log/${OUTPUT_DIR_PREFIX}_${DATE_INFO}.log

SPLIT = test

# Command Setting
CUDA_USE_DEVICES := 0

notebook:
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} jupyter lab

# If you want to use latest version of pytorch for cu113 or cu114, please use the command below intead of the 3rd line
# ${POETRY_RUN} pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
install:
	poetry install
	@echo Installing the correct version for your environment
	${POETRY_RUN} pip uninstall torch
	${POETRY_RUN} pip install torch --extra-index-url https://download.pytorch.org/whl/cu113

# if you already have cloned files2rouge repository, comment out "git clone ~~~"
setup-rouge:
	${POETRY_RUN} pip install -U git+https://github.com/pltrdy/pyrouge
	git clone https://github.com/pltrdy/files2rouge.git
	cd files2rouge && ${POETRY_RUN} python setup_rouge.py
	cd files2rouge && ${POETRY_RUN} python setup.py install
	wget -N -P data http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
	unzip data/stanford-corenlp-full-2016-10-31.zip -d data

bpe-preprocess:
	python data_utils/make_datafiles.py data/cnn/stories data/dailymail/stories data
	@echo BPE preprocess
	wget -N -P data 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
	wget -N -P data 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
	wget -N -P data 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
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

# Rewrite the length range info as necessary.
split-data-by-length:
	@echo Split data by length
	${POETRY_RUN} python data_utils/split_data_by_length.py --upper 49
	@echo Binarize dataset
	${POETRY_RUN} fairseq-preprocess \
		--source-lang "source" \
		--target-lang "target" \
		--trainpref "${TASK}/train.-49.bpe" \
		--validpref "${TASK}/val.-49.bpe" \
		--testpref "${TASK}/test.-49.bpe" \
		--destdir "${TASK}.-49-bin/" \
		--workers 60 \
		--srcdict data/dict.txt \
		--tgtdict data/dict.txt;

preprocess-extractor:
	mkdir -p data/bart.with.extractor.large
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} python data_utils/rename_weights_for_extractor.py

finetune-baseline-large:
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} python train_fairseq/train.py ${INPUT_DATA_DIR} \
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
		--find-unused-parameters \
		--validate-interval-updates 200 \
		 --use-wandb \
		> ${LOG_FILE_PATH};

finetune-proposal-large:
	${eval OUTPUT_DIR_PREFIX := finetune-proposal-large}
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} python train_fairseq/train.py ${INPUT_DATA_DIR} \
		--save-dir ${TRAIN_DEST_DIR} \
		--no-progress-bar --log-interval 20 --log-format simple \
		--restore-file ${PRETRAINED_LARGE_PATH_FOR_EXTRACTOR} \
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
		--validate-interval-updates 200 \
		--extract-num 256 --use-wandb \
		> ${LOG_FILE_PATH};

# Usage: make generate-baseline TRAIN_DEST_DIR=hogehoge
generate-baseline:
	${eval OUTPUT_DIR_PREFIX := generate-baseline}
	cp ${INPUT_DATA_DIR}/dict.source.txt ${TRAIN_DEST_DIR}/
	cp ${INPUT_DATA_DIR}/dict.target.txt ${TRAIN_DEST_DIR}/
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} python train_fairseq/generate.py \
		--model-dir ${TRAIN_DEST_DIR} \
		--model-file checkpoint_best.pt \
		--src data/cnn_dm/${SPLIT}.source \
		--out ${TRAIN_DEST_DIR}/${SPLIT}.hypo

# Usage: make generate-proposal TRAIN_DEST_DIR=hogehoge
generate-proposal:
	${eval OUTPUT_DIR_PREFIX := generate-proposal}
	cp ${INPUT_DATA_DIR}/dict.source.txt ${TRAIN_DEST_DIR}/
	cp ${INPUT_DATA_DIR}/dict.target.txt ${TRAIN_DEST_DIR}/
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} python train_fairseq/generate.py --use-proposal \
		--model-dir ${TRAIN_DEST_DIR} \
		--model-file checkpoint_best.pt \
		--src data/cnn_dm/${SPLIT}.source \
		--out ${TRAIN_DEST_DIR}/${SPLIT}.hypo

calc-rouge:
	export CLASSPATH=data/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar
	cat ${TRAIN_DEST_DIR}/${SPLIT}.hypo | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${TRAIN_DEST_DIR}/${SPLIT}.hypo.tokenized
	cat ${TASK}/${SPLIT}.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${TASK}/${SPLIT}.target.tokenized
	${POETRY_RUN} files2rouge ${TRAIN_DEST_DIR}/${SPLIT}.hypo.tokenized ${TASK}/${SPLIT}.target.tokenized > ${TRAIN_DEST_DIR}/${SPLIT}.result

params-tune-proposal-large:
	${eval OUTPUT_DIR_PREFIX := params-tune-proposal-large}
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} python train_fairseq/params_tuner.py ${INPUT_DATA_DIR} \
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