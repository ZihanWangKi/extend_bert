# Extended M-BERT
<h5 align="center">Extending Multilingual BERT to Low-Resource Languages </h5>

## Motivation
Low resource languages usually lack sufficient task-related labelled data. Multilingual BERT (M-BERT) has been a 
success to help transferring knowledge from high resource languages to low reosurce ones. 
In this work, we show that by adapting a continue training framework, this transfer ability can be significantly improved.

## Results
We perform an extensive set of experiments with Named Entity Recognition (NER) on 27 languages, 
only 16 of which are in M-BERT, and show an average increase of about 6% F1 on M-BERT languages and 23% F1 increase on new languages. 

## Scripts

#### Creating pre-training data

See [preprocessing-scripts](preprocessing-scripts) for how to create the tfrecords for BERT continue training.

#### Pre-training BERT

Assuming that the base model to extend is multilingual BERT, you can follow [bert-running-scripts](bert-running-scripts)
to initialize a model and start training. Our code supports google cloud with tpus. 

#### Evaluating

We provide code for evaluating on NER. See [evaluating-scripts](evaluating-scripts).

## Requirements

You can do ```pip install -r requirements.txt``` with python3.6+ to have the same environment as ours.

## Citation
Please cite the following paper if you find our paper useful. Thanks!

>Wang, Zihan, Karthikeyan K, Stephen Mayhew, and Dan Roth. "Extending Multilingual BERT to Low-Resource Languages." arXiv preprint arXiv:2004.13640 (2020).

```
@article{wang2019cross,
  title={Extending Multilingual BERT to Low-Resource Languages},
  author={Wang, Zihan and K, Karthikeyan and Mayhew, Stephen and Roth, Dan},
  journal={arXiv preprint arXiv:2004.13640},
  year={2020}
}
```
