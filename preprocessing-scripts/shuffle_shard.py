import random
import os
import json

'''
Creates shards, considers sampling probability (exponential 0.7).
# input: 
    - fnames: path to all the text files to process  
    - outdir: path to output all the shards
    - limit: number of lines of each shard, can be off by a few lines
# output:
    - Sharded files inside outdir
'''


class LineCount:
    def __init__(self):
        self.mem = {}

    def line_count(self, filename):
        if filename not in self.mem:
            self.mem[filename] = sum(1 for _ in open(filename))
        return self.mem[filename]


def read_doc(fp):
    """
    Read until the next empty line.
    """
    line = fp.readline()
    doc = []
    while line and (line != '\n' and len(doc) < 4096):  # cutoff document if it is too long.
        doc.append(line)
        line = fp.readline()
    return doc, line == ''


def load_into_memory(fp):  # loads whole file to allow randomization
    documents = []
    end = False
    while not end:
        document, end = read_doc(fp)
        documents.append(document)
    random.shuffle(documents)
    return documents


def json_print(dict_p):
    print(json.dumps(dict_p, indent=2))


def get_sample_probability(filenames, linecount):
    size_dict = {filename: linecount.line_count(filename) for filename in filenames}
    total_size = sum(size_dict.values())
    exp_sample = {filename: ((size + 0.0) / total_size) ** 0.7 for filename, size in size_dict.items()}
    sum_prob = sum(exp_sample.values())
    exp_sample = {filename: prob / sum_prob for filename, prob in exp_sample.items()}
    json_print(exp_sample)
    print(total_size)
    prob_dict = {filename: prob * total_size / size_dict[filename] for filename, prob in exp_sample.items()}
    json_print(prob_dict)
    return prob_dict


def shuffle_shard(fnames, outdir, limit=50000):
    """
    outdir: where we will store shards
    limit: number of lines per shard (roughly)
    fnames: list of text files containing documents.
    """
    num = 0
    prefix = "shard_"
    outfile = "{}{:02}".format(prefix, num)
    out = open(os.path.join(outdir, outfile), "w")
    shard_size = 0
    linecount = LineCount()
    fname_index = {fname: i for i, fname in enumerate(fnames)}
    sample_prob = get_sample_probability([fname for fname in fnames], linecount)
    documents = [load_into_memory(open(fp)) for fp in fnames]
    print("Done loading documents")
    documents_length = {fname: len(documents[i]) for i, fname in enumerate(fnames)}
    fake_files = [
        [(fname_index[fname], i % documents_length[fname]) for i in range(int(documents_length[fname] * prob))]
            for fname, prob in sample_prob.items()
    ] # use (file_id, doc_id) to represent a document
    fake_files = [y for x in fake_files for y in x]  # flatten
    random.shuffle(fake_files)

    for file_id, doc_id in fake_files:
        doc = documents[file_id][doc_id]
        shard_size += len(doc)
        for line in doc:
            out.write(line)
        out.write("\n")

        if shard_size > limit:
            out.close()
            num += 1
            # open a new file
            outfile = "{}{:02}".format(prefix, num)
            out = open(os.path.join(outdir, outfile), "w")
            shard_size = 0

    out.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Shuffle and shard a group of text files.')
    parser.add_argument('--fnames', type=str, nargs='+', help='filenames of text files')
    parser.add_argument('--outdir', type=str, help='directory to write to')
    parser.add_argument('--limit', type=int, help='number of lines per shard', default=50000)

    args = parser.parse_args()
    shuffle_shard(args.fnames, args.outdir, args.limit)
