TASK=data/cnn_dm
for SPLIT in train val test
do
  for LANG in source target
  do
    poetry run python data_utils/multiprocessing_bpe_encoder.py \
    --encoder-json data/encoder.json \
    --vocab-bpe data/vocab.bpe \
    --inputs "$TASK/$SPLIT.$LANG" \
    --outputs "$TASK/$SPLIT.bpe.$LANG" \
    --workers 60 \
    --keep-empty;
  done
done