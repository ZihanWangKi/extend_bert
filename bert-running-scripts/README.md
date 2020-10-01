## Pre-training BERT
#### Preparation
- Google Cloud Bucket Storage
- Google Cloud Instance
- Google Cloud Tpu

When creating a google cloud instance, make sure full api access is turned on. Also ensure that you have >= 15G disk space and >= 16G cpu memory.
These are needed to create the extended model. You may reduce 
the cpu to the normal 2cores/4G after creation is done (before actually continue training).  
Correctly set GC_BUCKET_NAME in ``init-gcloud-server.sh`` and ``create-extend.sh`` to your cloud storage bucket name. 
Also modify the folder name in both scripts to the corresponding folder uploaded to cloud storage from preprocessing.  

#### Execution
In a google cloud instance, run ``init-gcloud-server.sh`` and ``run.sh`` will be created. Then run ``create-extend.sh`` to create an initial checkpoint.  
Finally run ``run.sh`` and pass it a tpu name to initiate bert training.  