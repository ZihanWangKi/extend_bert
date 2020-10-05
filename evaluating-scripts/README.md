## Evaluating-scripts
#### Preparation (NER)
- Prepare the ner data and bert models.

Follow ``init.sh`` and add the current directories path to python's path.  
Set up PROJECT_MODELS and PROJECT_SCRIPTS in ``train.sh``.
Set up ner data path and bert model path in base.jsonnet.
``./train.sh 0 allennlp-config/base.jsonnet`` will execute ner training.
