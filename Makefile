POETRY_RUN := poetry run
EXPORT_CORENLP := export CLASSPATH=data/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar

DATASET=cnn_dm
TEXT_DATA_DIR=data/${DATASET}
INPUT_DATA_DIR=${TEXT_DATA_DIR}-bin
TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=16
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=128
PRETRAINED_BASE_PATH=data/bart.base/model.pt
PRETRAINED_LARGE_PATH=data/bart.large/model.pt
PRETRAINED_LARGE_PATH_FOR_EXTRACTOR=data/bart.with.extractor.large/model.pt
PRETRAINED_LARGE_PATH_FOR_PROPOSAL=data/bart.extractor.in.encoder.large/model.pt
PRETRAINED_LARGE_CNN_PATH=data/bart.large.cnn/model.pt

INIT_TOPK_EPS=0.001
MIN_TOPK_EPS=0.001

# Args for enlightening
MISS_THRESHOLD=0.2
MAX_FREQ=3
ENLIGHTEN_WIDTH_COEF=1

# Output data path
DATE_INFO := $(shell date +'%Y%m%d%H%M%S')
OUTPUT_DIR := output
OUTPUT_DIR_PREFIX := finetune-baseline-large
TRAIN_DEST_DIR = ${OUTPUT_DIR}/${OUTPUT_DIR_PREFIX}_${DATE_INFO}
# TENSORBOARD_LOG_DIR = tensorboard/${OUTPUT_DIR_PREFIX}_${DATE_INFO}
LOG_FILE_PATH = log/${OUTPUT_DIR_PREFIX}_${DATE_INFO}.log

SPLIT = test
BEAM_ARGS = least

# Command Setting
CUDA_USE_DEVICES := 0
PORT_NUMBER := 8888

notebook:
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} jupyter lab --port ${PORT_NUMBER}

# If you want to use latest version of pytorch for cu113 or cu114, please use the command below intead of the 3rd line
# ${POETRY_RUN} pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
install:
	poetry install
	@echo Installing the correct version for your environment
	${POETRY_RUN} pip uninstall torch
	${POETRY_RUN} pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# if you already have cloned files2rouge repository, comment out "git clone ~~~"
setup-rouge:
	${POETRY_RUN} pip install -U git+https://github.com/pltrdy/pyrouge
	git clone https://github.com/pltrdy/files2rouge.git
	cd files2rouge && ${POETRY_RUN} python setup_rouge.py
	cd files2rouge && ${POETRY_RUN} python setup.py install
	wget -N -P data http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
	unzip data/stanford-corenlp-full-2016-10-31.zip -d data

format-cnndm:
	@echo Download and Convert CNN/DM data to appropriate format
	${POETRY_RUN} python data_utils/make_datafiles.py data/cnn/stories data/dailymail/stories data

download-and-format-xsum:
	@echo Convert XSUM data to appropriate format
	${POETRY_RUN} python data_utils/get_xsum.py

bpe-preprocess:
	@echo BPE preprocess
	wget -N -P data 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
	wget -N -P data 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
	wget -N -P data 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'
	bash data_utils/bpe_preprocess.sh ${TEXT_DATA_DIR}
	@echo Binarize dataset
	${POETRY_RUN} fairseq-preprocess \
		--source-lang "source" \
		--target-lang "target" \
		--trainpref "${TEXT_DATA_DIR}/train.bpe" \
		--validpref "${TEXT_DATA_DIR}/val.bpe" \
		--testpref "${TEXT_DATA_DIR}/test.bpe" \
		--destdir "${INPUT_DATA_DIR}" \
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
		--trainpref "${TEXT_DATA_DIR}/train.-49.bpe" \
		--validpref "${TEXT_DATA_DIR}/val.-49.bpe" \
		--testpref "${TEXT_DATA_DIR}/test.-49.bpe" \
		--destdir "${TEXT_DATA_DIR}.-49-bin/" \
		--workers 60 \
		--srcdict data/dict.txt \
		--tgtdict data/dict.txt;

preprocess-extractor:
	mkdir -p data/bart.with.extractor.large
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} python data_utils/rename_weights_for_extractor.py

preprocess-proposal:
	mkdir -p data/bart.extractor.in.encoder.large
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} python data_utils/rename_weights_for_proposal.py

make_oracle_length_file:
	${POETRY_RUN} python data_utils/make_oracle_length_data.py --dataset ${DATASET}

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

# CPUで回す場合はfp16オプションを外す
finetune-proposal-large-on-cpu:
	${eval OUTPUT_DIR_PREFIX := finetune-proposal-large-on-cpu}
	CUDA_VISIBLE_DEVICES="" ${POETRY_RUN} python train_fairseq/train.py ${INPUT_DATA_DIR} \
		--save-dir ${TRAIN_DEST_DIR} \
		--no-progress-bar --log-interval 1 --log-format simple \
		--restore-file ${PRETRAINED_LARGE_PATH_FOR_PROPOSAL} \
		--max-tokens ${MAX_TOKENS} \
		--task proposal_task \
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
		--update-freq 1 \
		--skip-invalid-size-inputs-valid-test \
		--find-unused-parameters \
		--validate-interval-updates 200 \
		--use-differentiable-topk \
		--apply-formula-to-extract-num --alpha-for-extract-num 5.0 --beta-for-extract-num 50 \
		--token-scoring-fn "self_attention" --when-to-extract "before_attention" \
		--use-wandb \
		> ${LOG_FILE_PATH};

finetune-proposal-large:
	${eval OUTPUT_DIR_PREFIX := finetune-proposal-large}
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} python train_fairseq/train.py ${INPUT_DATA_DIR} \
		--save-dir ${TRAIN_DEST_DIR} \
		--no-progress-bar --log-interval 20 --log-format simple \
		--restore-file ${PRETRAINED_LARGE_PATH_FOR_PROPOSAL} \
		--max-tokens ${MAX_TOKENS} \
		--task proposal_task \
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
		--validate-interval-updates 200 --no-epoch-checkpoints \
		--use-differentiable-topk \
		--apply-formula-to-extract-num --alpha-for-extract-num 1.0 --beta-for-extract-num 0 \
		--token-scoring-fn "self_attention" --when-to-extract "query_before_attention" \
		--sorted-topk --init-topk-eps ${INIT_TOPK_EPS} --normalization-after-soft-masking \
		--use-wandb \
		> ${LOG_FILE_PATH};

# Usage: make generate-baseline TRAIN_DEST_DIR=hogehoge
generate-baseline:
	${eval OUTPUT_DIR_PREFIX := generate-baseline}
	cp ${INPUT_DATA_DIR}/dict.source.txt ${TRAIN_DEST_DIR}/
	cp ${INPUT_DATA_DIR}/dict.target.txt ${TRAIN_DEST_DIR}/
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} python train_fairseq/generate.py \
		--model-dir ${TRAIN_DEST_DIR} \
		--model-file checkpoint_best.pt \
		--src data/${DATASET}/${SPLIT}.source \
		--beam-args ${BEAM_ARGS} \
		--out ${TRAIN_DEST_DIR}/${SPLIT}_${DATASET}.hypo_${BEAM_ARGS}_args

# Usage: make generate-proposal TRAIN_DEST_DIR=hogehoge
generate-proposal:
	${eval OUTPUT_DIR_PREFIX := generate-proposal}
	cp ${INPUT_DATA_DIR}/dict.source.txt ${TRAIN_DEST_DIR}/
	cp ${INPUT_DATA_DIR}/dict.target.txt ${TRAIN_DEST_DIR}/
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} python train_fairseq/generate_with_desired_length.py --use-proposal \
		--model-dir ${TRAIN_DEST_DIR} \
		--model-file checkpoint_best.pt \
		--src data/${DATASET}/${SPLIT}.source \
		--desired-length data/desired_lengths/${DATASET}/${SPLIT}.oracle${LEN_SUFFIX} \
		--beam-args ${BEAM_ARGS} \
		--topk-eps ${MIN_TOPK_EPS} \
		--out ${TRAIN_DEST_DIR}/${SPLIT}_${DATASET}.hypo${LEN_SUFFIX}_${BEAM_ARGS}_args

# Usage: make generate-proposal-fixed-len TRAIN_DEST_DIR=hogehoge FIXED_LENGTH=70
generate-proposal-fixed-len:
	${eval OUTPUT_DIR_PREFIX := generate-proposal-fixed-len}
	cp ${INPUT_DATA_DIR}/dict.source.txt ${TRAIN_DEST_DIR}/
	cp ${INPUT_DATA_DIR}/dict.target.txt ${TRAIN_DEST_DIR}/
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} python train_fairseq/generate_with_fixed_length.py --use-proposal \
		--model-dir ${TRAIN_DEST_DIR} \
		--model-file checkpoint_best.pt \
		--src data/${DATASET}/${SPLIT}.source \
		--fixed-length ${FIXED_LENGTH} \
		--beam-args ${BEAM_ARGS} \
		--out ${TRAIN_DEST_DIR}/${SPLIT}_${DATASET}.hypo${FIXED_LENGTH}_${BEAM_ARGS}_args

# Usage: make generate-proposal TRAIN_DEST_DIR=hogehoge
generate-proposal-topk-randperm:
	${eval OUTPUT_DIR_PREFIX := generate-proposal}
	$(eval HYPO_SUFFIX := ${LEN_SUFFIX}_topk_randperm_${BEAM_ARGS}_args)
	cp ${INPUT_DATA_DIR}/dict.source.txt ${TRAIN_DEST_DIR}/
	cp ${INPUT_DATA_DIR}/dict.target.txt ${TRAIN_DEST_DIR}/
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} python train_fairseq/generate_with_desired_length.py --use-proposal \
		--model-dir ${TRAIN_DEST_DIR} \
		--model-file checkpoint_best.pt \
		--src data/${DATASET}/${SPLIT}.source \
		--desired-length data/desired_lengths/${DATASET}/${SPLIT}.oracle${LEN_SUFFIX} \
		--beam-args ${BEAM_ARGS} \
		--topk-eps ${MIN_TOPK_EPS} \
		--topk-randperm \
		--out ${TRAIN_DEST_DIR}/${SPLIT}_${DATASET}.hypo${HYPO_SUFFIX}

generate-before-enlightening:
	$(eval MISSED_PREFIX := ${SPLIT}_${DATASET}_missed${MISS_THRESHOLD})
	$(eval UNMODIFIED_PREFIX := ${MISSED_PREFIX}_unmodified)
	$(eval HYPO_SUFFIX := ${LEN_SUFFIX}_${BEAM_ARGS}_args)
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} python train_fairseq/generate_with_desired_length.py --use-proposal \
		--model-dir ${TRAIN_DEST_DIR} \
		--model-file checkpoint_best.pt \
		--src ${TRAIN_DEST_DIR}/${MISSED_PREFIX}.source \
		--desired-length ${TRAIN_DEST_DIR}/${MISSED_PREFIX}.oracle${LEN_SUFFIX} \
		--beam-args ${BEAM_ARGS} \
		--topk-eps ${MIN_TOPK_EPS} \
		--out ${TRAIN_DEST_DIR}/${UNMODIFIED_PREFIX}.hypo${HYPO_SUFFIX}
	${EXPORT_CORENLP} && cat ${TRAIN_DEST_DIR}/${UNMODIFIED_PREFIX}.hypo${HYPO_SUFFIX} | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${TRAIN_DEST_DIR}/${UNMODIFIED_PREFIX}.hypo${HYPO_SUFFIX}.tokenized
	${POETRY_RUN} files2rouge ${TRAIN_DEST_DIR}/${UNMODIFIED_PREFIX}.hypo${HYPO_SUFFIX}.tokenized ${TRAIN_DEST_DIR}/${MISSED_PREFIX}.target.tokenized > ${TRAIN_DEST_DIR}/${UNMODIFIED_PREFIX}.result${HYPO_SUFFIX}

enlighten-missed-tokens:
	${POETRY_RUN} python train_fairseq/prepare_enlighten_missed_tokens.py \
		--train-dest-dir ${TRAIN_DEST_DIR} \
		--dataset ${DATASET} --beam-args ${BEAM_ARGS} \
		--miss-threshold ${MISS_THRESHOLD} --max-freq ${MAX_FREQ}
	$(eval MISSED_PREFIX := ${SPLIT}_${DATASET}_missed${MISS_THRESHOLD})
	$(eval MODIFIED_PREFIX := ${MISSED_PREFIX}_${MAX_FREQ}_${ENLIGHTEN_WIDTH_COEF}_modified)
	$(eval HYPO_SUFFIX := ${LEN_SUFFIX}_${BEAM_ARGS}_args)
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} python train_fairseq/generate_with_enlighten_indices.py --use-proposal \
		--model-dir ${TRAIN_DEST_DIR} \
		--model-file checkpoint_best.pt \
		--src ${TRAIN_DEST_DIR}/${MISSED_PREFIX}.source \
		--desired-length ${TRAIN_DEST_DIR}/${MISSED_PREFIX}.oracle${LEN_SUFFIX} \
		--enlighten-indices ${TRAIN_DEST_DIR}/${MISSED_PREFIX}_${MAX_FREQ}.enlighten_indices_${BEAM_ARGS}_args \
		--beam-args ${BEAM_ARGS} \
		--topk-eps ${MIN_TOPK_EPS} \
		--enlighten-width-coef ${ENLIGHTEN_WIDTH_COEF} \
		--out ${TRAIN_DEST_DIR}/${MODIFIED_PREFIX}.hypo${HYPO_SUFFIX}
	${POETRY_RUN} files2rouge ${TRAIN_DEST_DIR}/${MISSED_PREFIX}.hypo${HYPO_SUFFIX}.tokenized ${TRAIN_DEST_DIR}/${MISSED_PREFIX}.target.tokenized > ${TRAIN_DEST_DIR}/${MISSED_PREFIX}.result${HYPO_SUFFIX}
	${EXPORT_CORENLP} && cat ${TRAIN_DEST_DIR}/${MODIFIED_PREFIX}.hypo${HYPO_SUFFIX} | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${TRAIN_DEST_DIR}/${MODIFIED_PREFIX}.hypo${HYPO_SUFFIX}.tokenized
	${POETRY_RUN} files2rouge ${TRAIN_DEST_DIR}/${MODIFIED_PREFIX}.hypo${HYPO_SUFFIX}.tokenized ${TRAIN_DEST_DIR}/${MISSED_PREFIX}.target.tokenized > ${TRAIN_DEST_DIR}/${MODIFIED_PREFIX}.result${HYPO_SUFFIX}

calc-rouge:
	$(eval HYPO_SUFFIX := ${LEN_SUFFIX}_${BEAM_ARGS}_args)
	${EXPORT_CORENLP} && cat ${TRAIN_DEST_DIR}/${SPLIT}_${DATASET}.hypo${HYPO_SUFFIX} | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${TRAIN_DEST_DIR}/${SPLIT}_${DATASET}.hypo${HYPO_SUFFIX}.tokenized
	${EXPORT_CORENLP} && cat ${TEXT_DATA_DIR}/${SPLIT}.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${TEXT_DATA_DIR}/${SPLIT}.target.tokenized
	${POETRY_RUN} files2rouge ${TRAIN_DEST_DIR}/${SPLIT}_${DATASET}.hypo${HYPO_SUFFIX}.tokenized ${TEXT_DATA_DIR}/${SPLIT}.target.tokenized > ${TRAIN_DEST_DIR}/${SPLIT}_${DATASET}.result${HYPO_SUFFIX}

calc-rouge-topk-randperm:
	$(eval HYPO_SUFFIX := ${LEN_SUFFIX}_topk_randperm_${BEAM_ARGS}_args)
	${EXPORT_CORENLP} && cat ${TRAIN_DEST_DIR}/${SPLIT}_${DATASET}.hypo${HYPO_SUFFIX} | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${TRAIN_DEST_DIR}/${SPLIT}_${DATASET}.hypo${HYPO_SUFFIX}.tokenized
	${EXPORT_CORENLP} && cat ${TEXT_DATA_DIR}/${SPLIT}.target | java edu.stanford.nlp.process.PTBTokenizer -ioFileList -preserveLines > ${TEXT_DATA_DIR}/${SPLIT}.target.tokenized
	${POETRY_RUN} files2rouge ${TRAIN_DEST_DIR}/${SPLIT}_${DATASET}.hypo${HYPO_SUFFIX}.tokenized ${TEXT_DATA_DIR}/${SPLIT}.target.tokenized > ${TRAIN_DEST_DIR}/${SPLIT}_${DATASET}.result${HYPO_SUFFIX}

# Usage: make calc-faithful-score TRAIN_DEST_DIR=hogehoge
calc-faithful-score:
	${eval OUTPUT_DIR_PREFIX := generate-proposal}
	$(eval HYPO_SUFFIX := ${LEN_SUFFIX}_${BEAM_ARGS}_args)
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} python train_fairseq/export_topk_tokens.py \
		--model-dir ${TRAIN_DEST_DIR} \
		--model-file checkpoint_best.pt \
		--src data/${DATASET}/${SPLIT}.source \
		--gen ${TRAIN_DEST_DIR}/${SPLIT}_${DATASET}.hypo${HYPO_SUFFIX} \
		--desired-length data/desired_lengths/${DATASET}/${SPLIT}.oracle${LEN_SUFFIX} \
		--bolded-out ${TRAIN_DEST_DIR}/${SPLIT}_${DATASET}.bolded_src${LEN_SUFFIX} \
		--score-out ${TRAIN_DEST_DIR}/${SPLIT}_${DATASET}.faithful_scores${HYPO_SUFFIX} \
		--topk-eps ${MIN_TOPK_EPS}

calc-faithful-score-topk-randperm:
	${eval OUTPUT_DIR_PREFIX := generate-proposal}
	$(eval HYPO_SUFFIX := ${LEN_SUFFIX}_topk_randperm_${BEAM_ARGS}_args)
	CUDA_VISIBLE_DEVICES=${CUDA_USE_DEVICES} ${POETRY_RUN} python train_fairseq/export_topk_tokens.py \
		--model-dir ${TRAIN_DEST_DIR} \
		--model-file checkpoint_best.pt \
		--src data/${DATASET}/${SPLIT}.source \
		--gen ${TRAIN_DEST_DIR}/${SPLIT}_${DATASET}.hypo${HYPO_SUFFIX} \
		--desired-length data/desired_lengths/${DATASET}/${SPLIT}.oracle${LEN_SUFFIX} \
		--bolded-out ${TRAIN_DEST_DIR}/${SPLIT}_${DATASET}.bolded_src${LEN_SUFFIX}_topk_randperm \
		--score-out ${TRAIN_DEST_DIR}/${SPLIT}_${DATASET}.faithful_scores${HYPO_SUFFIX} \
		--topk-eps ${MIN_TOPK_EPS}

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
