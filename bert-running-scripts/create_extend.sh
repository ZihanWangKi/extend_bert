## we create the initial model which accounts for additional vocab size
GC_BUCKET_NAME=name_of_google_cloud_bucket
folder=test_data_folder
B=gs://${GC_BUCKET_NAME}/${folder}
gsutil cp -r $B .

python ./bert/run_pretraining.py \
  --input_file=$folder/data/shard_00.tfrecord \
  --output_dir=$folder/base_model_tmp \
  --do_train=True \
  --bert_config_file=$folder/bert_config.json \
  --train_batch_size=8 \
  --max_seq_length=128 \
  --num_train_steps=1 \
  --num_warmup_steps=0 \

python import_weights.py --import_from_checkpoint multi_cased_L-12_H-768_A-12/bert_model.ckpt \
                         --import_to_checkpoint $B/base_model_tmp/model.ckpt-0 \
                         --output_dir $B/extended_base_model \
                         --run 1

python import_weights.py --import_from_checkpoint multi_cased_L-12_H-768_A-12/bert_model.ckpt \
                         --import_to_checkpoint $folder/base_model_tmp/model.ckpt-0 \
                         --output_dir $folder/extended_base_model \
                         --run 2

gsutil cp -r $folder/extended_base_model $B