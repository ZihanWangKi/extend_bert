## Preprocessing-scripts
#### Preparation
- [Wikipedia](https://dumps.wikimedia.org/) text data if training from scratch. 
- [gsutil](https://cloud.google.com/storage/docs/gsutil_install) to copy the files to the bucket.

We recommend a tensorflow version of 1.15 to match the environment we used.  
Here, we provide a basic usage of our scripts.  
``init.sh`` will clone the official BERT repo, and create a ``test_data_folder`` with dummy text. You can try out the pipeline
with this dummy folder first.    
``preprocess_corpus.py`` takes in a text file and tokenizes it.  
``run.sh`` will shard the text files, create vocabulary for it, merge vocabulary with Multilingual BERT's, create bert-readable tensorflow records, and upload to google cloud.  
``run.sh`` requires a valid google cloud storage bucket to upload the data to gcloud. 

#### Execution
An example run that creates data that from a piece of dummy text:
```bash
./init.sh
python preprocess_corpus.py \
    --corpus test_data_folder/raw_txt/test.txt \
    --output test_data_folder/txt/en.txt
./run.sh test_data_folder
```

