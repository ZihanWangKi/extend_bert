sudo apt-get update
sudo apt-get install git
sudo apt-get install zip
sudo apt-get install tmux
sudo apt-get install wget
sudo apt-get install python-pip
pip install --upgrade pip
pip install numpy
pip install tensorflow==1.15
pip install --upgrade google-api-python-client
pip install --upgrade oauth2client
git clone https://github.com/google-research/bert.git

GC_BUCKET_NAME=name_of_google_cloud_bucket

# run.sh starts bert training
echo "tpu=\${1}
B=gs://${GC_BUCKET_NAME}/test_data_folder

python ./bert/run_pretraining.py \\
       --input_file=$B/data/*.tfrecord \\
       --output_dir=$B/pretraining_output \\
       --init_checkpoint=$B/extended_base_model \\
       --do_train=True \\
       --do_eval=True \\
       --bert_config_file=$B/bert_config.json \\
       --train_batch_size=32 \\
       --max_seq_length=128 \\
       --use_tpu=True \\
       --tpu_name=\${tpu} \\
       --num_tpu_cores=8 \\
       --max_predictions_per_seq=20 \\
       --num_train_steps=500000 \\
       --num_warmup_steps=10000 \\
       --save_checkpoints_steps=25000 \\
       --learning_rate=2e-5 2>&1 | tee log-\${tpu}.txt
" >> run.sh

wget https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
unzip multi_cased_L-12_H-768_A-12.zip