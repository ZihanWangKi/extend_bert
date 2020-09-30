import json
import sys

def load_to_list(file):
    with open(file, "r") as f:
        data = f.readlines()
    return list(map(lambda x: x.strip(), data))

base_vocab = sys.argv[1]
to_extend = sys.argv[2]
output = sys.argv[3]
config = sys.argv[4]

base_vocab = load_to_list(base_vocab)
to_extend = load_to_list(to_extend)
checker = set(base_vocab)
extended = 0
for word in to_extend:
    if word not in checker and "[UNUSED_" not in word: # generates lots of UNUSED_X >= 252 when corpus is small
        base_vocab.append(word)
        checker.add(word) # not necessary though
        extended += 1
print("{}/{}".format(extended, len(to_extend)))
with open(output, "w") as f:
    for word in base_vocab:
        f.write("{}\n".format(word))

bert_config = json.load(open(config, "r"))
bert_config["vocab_size"] = len(base_vocab)
json.dump(bert_config, open(config, "w"), indent=2)

