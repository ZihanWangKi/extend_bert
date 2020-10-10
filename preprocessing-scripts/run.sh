# ./run.sh path_to_data_directory [path_to_vocabulary_file]
#########################################Parameters#########################################
# Path to the folder that contains /txt/[text files for pretraining].
DATA_FOLDER=test_data_folder
# default params
THREADS=12

CREATE_VOCAB=1 # whether to create vocabulary
VOCAB_SIZE=10000
if [ -d "$1" ]; then
  DATA_FOLDER=${1%/}
  if [ -f "$2" ]; then
    # use precreated vocabulary
    CREATE_VOCAB=0
    VOCAB_SIZE=$(wc -l $2 | cut -d' ' -f1)
    VOCAB=$2
  fi
fi

if [ $CREATE_VOCAB -eq 1 ]; then
  VOCAB=$DATA_FOLDER/vocab.txt
fi

# Google cloud bucket name, needs to be changed!
GC_BUCKET_NAME=google_cloud_bucket_name

# Path to the BERT github repo, default is inside this folder
export BERT_PATH=bert/

#########################################INITIAIZATION#########################################


if [ ! -d "$DATA_FOLDER" ]; then
  echo "$DATA_FOLDER" does not exist!
  exit
fi
if [ ! -d "$DATA_FOLDER/txt" ]; then
  echo "$DATA_FOLDER" does not contain a txt folder!
  exit
fi
if [ ! -d "$BERT_PATH" ]; then
  echo $BERT_PATH does not exist, you can clone the repo from https://github.com/google-research/bert
  exit
fi

SHARD_DIR=$DATA_FOLDER/shards
DATA_DIR=$DATA_FOLDER/data
LOG_DIR=$DATA_FOLDER/logs
CONFIG_PATH=$DATA_FOLDER/bert_config.json
# size of the vocabulary to create, will fill with at least 256 [UNUSED].

mkdir -p ${SHARD_DIR}
mkdir -p ${DATA_DIR}
mkdir -p ${LOG_DIR}


#########################################SHARDING#########################################

echo ""
echo "BERT_BASE_DIR: ${DATA_FOLDER}"
echo "TEXT FILES: `ls ${DATA_FOLDER}/txt`"
echo ""
read -p "Continue (Y/N) " continue
if ! ( [ "${continue}" = "y" ] || [ "${continue}" = "Y" ] ); then exit 1; else echo "Continuing ... "; fi

python shuffle_shard.py --fnames "$DATA_FOLDER"/txt/* --outdir "$SHARD_DIR"

#########################################Vocabulary#########################################
# This writes the .model and .vocab files to the local directory.
if [ $CREATE_VOCAB -eq 1 ]; then
  python mkvocab.py --tokenized_dir "$SHARD_DIR" --out_vocab "$VOCAB"_unextended --vocab_size "$VOCAB_SIZE"
fi
python create_config.py --config_path "$CONFIG_PATH" --vocab_size "$VOCAB_SIZE"

# also takes care of updating the vocab_size
python extend_vocab.py bert-base-multilingual-cased-vocab.txt "$VOCAB"_unextended "$VOCAB" "$CONFIG_PATH"

for f in $SHARD_DIR/*; do
   ((i=i%THREADS)); ((i++==0)) && echo ... && wait
   echo processing $f
   python $BERT_PATH/create_pretraining_data.py \
       --input_file=${f} \
       --output_file=$DATA_DIR/"$(basename ${f})".tfrecord \
       --vocab_file=$VOCAB \
       --do_lower_case=False \
       --max_seq_length=128 \
       --do_whole_word_mask=True \
       --max_predictions_per_seq=20 \
       --masked_lm_prob=0.15 \
       --random_seed=12345 \
       --dupe_factor=5 &> $LOG_DIR/"$(basename ${f})".log &
done
wait

gsutil -m cp -r "$DATA_FOLDER"/ gs://$GC_BUCKET_NAME/"$(basename "${DATA_FOLDER}")"
