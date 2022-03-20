BASE_DIR := $(shell pwd)
POETRY_RUN := poetry run
SCRIPT_DIR := ${BASE_DIR}/train
MODE := finetune
DATE_INFO := $(shell date +'%Y%m%d%H%M%S')
OUTPUT_DIR := ${BASE_DIR}/output/${MODE}_${DATE_INFO}
OUTPUT_LOG_FILE := ${BASE_DIR}/log/${MODE}_${DATE_INFO}.log

install:
	poetry install
	echo "install the correct version for your environment"
	poetry run pip install torch==1.8.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

notebook:
	${POETRY_RUN} jupyter-lab

finetune-baseline:
	${POETRY_RUN} python ${SCRIPT_DIR}/finetune.py \
		--run_name ShortenSum_${DATE_INFO} \
		--do_train \
		--do_eval \
		--do_predict \
    	--model_name_or_path 'facebook/bart-large-cnn' \
    	--dataset_name cnn_dailymail \
    	--dataset_config "3.0.0" \
    	--output_dir ${OUTPUT_DIR} \
		--max_source_length 512 \
		--max_target_length 100 \
		--per_device_train_batch_size=1 \
		--per_device_eval_batch_size=1 \
		--num_train_epochs 5 \
		--logging_steps 100 --save_steps 1000 \
		--evaluation_strategy "epoch" \
		--gradient_accumulation_steps 64 \
		--metric_for_best_model 'loss' \
		> ${OUTPUT_LOG_FILE}

# 		--max_train_samples 8 --max_eval_samples 4 --max_predict_samples 4 \

generate-baseline:
	${POETRY_RUN} python ${SCRIPT_DIR}/finetune.py \
		--run_name ShortenSum_${DATE_INFO} \
		--do_predict \
    	--model_name_or_path 'google/pegasus-xsum' \
    	--dataset_name cnn_dailymail \
    	--dataset_config "3.0.0" \
    	--output_dir ${OUTPUT_DIR} \
		--max_source_length 512 \
		--max_target_length 100 \
		--per_device_eval_batch_size=4 \
		--num_beams=8 --predict_with_generate \
		--resume_from_checkpoint ${BASE_DIR}/output/finetune_20220124212001 \
		> ${OUTPUT_LOG_FILE}

finetune-baseline-no-trainer:
	${POETRY_RUN} python ${SCRIPT_DIR}/finetune_no_trainer.py \
    	--model_name_or_path 'google/pegasus-xsum' \
    	--dataset_name cnn_dailymail \
    	--dataset_config "3.0.0" \
    	--output_dir ${OUTPUT_DIR}/${MODE}_$(shell date +'%Y%m%d%H%M%S') \
		--max_source_length 512 \
		--max_target_length 100 \
		--per_device_train_batch_size=1 \
		--per_device_eval_batch_size=1 \

# finetune-proposal:
# 	${POETRY_RUN} python ${SCRIPT_DIR}/finetune.py \
# 		--run_name ShortenSum_${DATE_INFO} \
# 		--do_train \
# 		--do_eval \
# 		--do_predict \
#     	--model_name_or_path 'google/pegasus-large' \
#     	--dataset_name cnn_dailymail \
#     	--dataset_config "3.0.0" \
#     	--output_dir ${OUTPUT_DIR} \
# 		--max_source_length 512 \
# 		--max_target_length 100 \
# 		--per_device_train_batch_size=4 \
# 		--per_device_eval_batch_size=4 \
# 		--num_train_epochs 5 \
# 		--logging_steps 100 --save_steps 1000 \
# 		--evaluation_strategy "no" \
# 		--gradient_accumulation_steps 16 \
# 		--metric_for_best_model 'loss' \
# 		--resume_from_checkpoint ${BASE_DIR}/output/finetune_20211210005917 \
# 		> ${OUTPUT_LOG_FILE}
