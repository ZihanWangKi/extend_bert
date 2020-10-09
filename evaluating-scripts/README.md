## Evaluating-scripts
#### Preparation (NER)
- Prepare the ner data and bert models.

We used an environment with python3.6+ and allennlp 0.9

1. Add the python lib to python's path, e.g.
```echo /path/to/allennlp-lib >> virtual_env/lib/python/site-packages/mpath.pth```
so that the custom allennlp script can be used.

2. Set up PROJECT_MODELS and PROJECT_SCRIPTS in ``train.sh``.
3. Set up ner data path and bert model path in base.jsonnet.
4. ```./train.sh GPU_INDEX allennlp-config/base.jsonnet``` will execute ner training.
