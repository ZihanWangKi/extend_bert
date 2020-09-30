import sentencepiece as spm
import random
import os
import sys
import json
import argparse
from pathlib import Path


'''
Generates vocabulary.
# input: 
    - tokenized_dir: directory containing shards of the corpuses. The corpuses should be tokenized
    - out_vocab: {base_dir}/OUT_FILE_NAME. Stores the output vocabulary. Also generates a bert_config.json in base_dir
    - vocab_size: size of vocabulary of desire. Should > 256    
# output:
    - intermediate files: ${dir_name}.sentpiece.model, ${dir_name}.sentpiece.vocab
    - vocabulary
    - bert_config.json
'''

NUM_PLACEHOLDERS = 256 # placeholder in the vocabulary

parser = argparse.ArgumentParser()
parser.add_argument("--tokenized_dir", "-t", required=True,
                    help="Path to directory containing shards of the tokenized corpus")
parser.add_argument("--out_vocab", "-o", required=True,
                    help="Path to vocabulary output file")
parser.add_argument("--vocab_size", "-v", type=int, default=30000,
                    help="Vocabulary size")

args = parser.parse_args()
print(vars(args))

assert os.path.isdir(args.tokenized_dir), "tokenized_dir must be a directory"
if os.path.exists(args.out_vocab):
    print(f"{args.out_vocab} exists, aborting...")
    sys.exit(0)

assert args.vocab_size > NUM_PLACEHOLDERS, "vocab_size should be more than 256"

dir_name = Path(args.tokenized_dir).stem
MODEL_PREFIX = f"{dir_name}.sentpiece"

fnames = list(map(lambda f: os.path.join(args.tokenized_dir, f), os.listdir(args.tokenized_dir)))
comma_sep_fnames = ",".join(fnames)

SPM_COMMAND = ('--input={} --model_prefix={} --normalization_rule_name=identity '
               '--vocab_size={} --num_threads=24 --input_sentence_size=2000000 '
               '--shuffle_input_sentence=true --model_type=unigram '
               '--bos_id=-1 --eos_id=-1 --hard_vocab_limit=false').format(
    comma_sep_fnames, MODEL_PREFIX,
    args.vocab_size - NUM_PLACEHOLDERS)



spm.SentencePieceTrainer.Train(SPM_COMMAND)


def read_sentencepiece_vocab(filepath):
    voc = []
    with open(filepath, encoding='utf-8') as fi:
        for line in fi:
            voc.append(line.split("\t")[0])
    # skip the first <unk> token
    voc = voc[1:]
    return voc


snt_vocab = read_sentencepiece_vocab("{}.vocab".format(MODEL_PREFIX))
print("Learnt vocab size: {}".format(len(snt_vocab)))
print("Sample tokens: {}".format(random.sample(snt_vocab, 10)))


def parse_sentencepiece_token(token):
    if token.startswith("▁"): # this character is not the usual "_" (ascii: 95), it is used in sentencepiece
        return token[1:]
    else:
        return "##" + token


bert_vocab = list(map(parse_sentencepiece_token, snt_vocab))

ctrl_symbols = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
bert_vocab = ctrl_symbols + bert_vocab

bert_vocab += ["[UNUSED_{}]".format(i) for i in range(args.vocab_size - len(bert_vocab))]
print(len(bert_vocab))
print("Writing to", args.out_vocab)
with open(args.out_vocab, "w") as fo:
    for token in bert_vocab:
        fo.write(token + "\n")

