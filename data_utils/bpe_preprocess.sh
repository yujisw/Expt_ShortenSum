# The number of arguments specified at runtime, i.e., the value of the variable $#, must be 1 or an error is thrown.
if [ $# -ne 1 ]; then
  echo "There are $# arguments specified." 1>&2
  echo "1 argument is required to execute it." 1>&2
  exit 1
fi

TASK=$1
echo "BPE preprocessing on $TASK..."

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