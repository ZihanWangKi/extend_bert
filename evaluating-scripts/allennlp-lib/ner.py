from typing import Dict, List, Sequence, Iterable
import itertools
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
import os
import ccg_nlpy as ccg

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":  # pylint: disable=simplifiable-if-statement
            return True
        else:
            return False

@DatasetReader.register("ner") # skips files that have ner tags
class NERTranslitDatasetReader(DatasetReader):
    _VALID_LABELS = {'ner'}

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tag_label: str = "ner",
                 feature_labels: Sequence[str] = (),
                 lazy: bool = False,
                 coding_scheme: str = "IOB1",
                 label_namespace: str = "labels",
                 ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        if tag_label is not None and tag_label not in self._VALID_LABELS:
            raise ConfigurationError("unknown tag label type: {}".format(tag_label))
        for label in feature_labels:
            if label not in self._VALID_LABELS:
                raise ConfigurationError("unknown feature label type: {}".format(label))
        if coding_scheme not in ("IOB1", "BIOUL"):
            raise ConfigurationError("unknown coding_scheme: {}".format(coding_scheme))

        self.tag_label = tag_label
        self.feature_labels = set(feature_labels)
        self.coding_scheme = coding_scheme
        self.label_namespace = label_namespace
        self._original_coding_scheme = "IOB1"

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # if `file_path` is a URL, redirect to the cache
        print(file_path)
        for single_file_path in file_path.split(','):
            if os.path.isdir(single_file_path):  # read with text_annotation: json like
                logger.info("Reading instances from files in folder at: %s", single_file_path)
                for fname in os.listdir(single_file_path):
                    doc = ccg.load_document_from_json(os.path.join(single_file_path, fname))
                    label_indices = ["O"] * len(doc.tokens)
                    if "NER_CONLL" in doc.view_dictionary:
                        ner = doc.get_ner_conll
                    else:
                        ner = None
                    doctokens = doc.tokens

                    if ner is not None:
                        if ner.cons_list is not None:
                            for cons in ner:
                                tag = cons['label']
                                # constituent range end is one past the token
                                for i in range(cons['start'], cons['end']):
                                    pref = "I-"
                                    # in IOB1: you can't start a sentence with B
                                    if i not in doc.sentence_end_position and \
                                            i == cons["start"] and \
                                            label_indices[i - 1][2:] == tag:
                                        pref = "B-"
                                    label_indices[i] = pref + tag
                        for start, end in zip([0] + doc.sentence_end_position[:-1], doc.sentence_end_position):
                            sent_toks = doctokens[start:end]
                            ner_tags = label_indices[start:end]
                            tokens = [Token(token) for token in sent_toks]
                            yield self.text_to_instance(single_file_path, tokens, ner_tags=ner_tags)
                    else:
                        print("doc has no ner: ", fname)
            else:  # read with conll format
                with open(single_file_path, "r") as data_file:
                    logger.info("Reading instances from lines in file at: %s", single_file_path)

                    # Group into alternative divider / sentence chunks.
                    for is_divider, lines in itertools.groupby(data_file, _is_divider):
                        # Ignore the divider chunks, so that `lines` corresponds to the words
                        # of a single sentence.
                        if not is_divider:
                            fields = [line.strip().split() for line in lines]
                            # unzipping trick returns tuples, but our Fields need lists
                            fields = [list(field) for field in zip(*fields)]
                            if len(fields) == 2:
                                tokens_, ner_tags = fields
                                weights = None
                            else:
                                assert False, "requires [text tag] format data"
                            # TextField requires ``Token`` objects
                            tokens = [Token(token) for token in tokens_]
                            yield self.text_to_instance(single_file_path, tokens, ner_tags, weights)

    def text_to_instance(self,  # type: ignore
                         filename: str,
                         tokens: List[Token],
                         ner_tags: List[str] = None,
                         weights: List[float] = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {'tokens': sequence,
                                             "metadata": MetadataField({"words": [x.text for x in tokens]})}
        if weights is None:
            weights = [1.0] * len(tokens)
        weight = weights[0]

        instance_fields["dataset"] = MetadataField(filename)

        # Recode the labels if necessary.
        if self.coding_scheme == "BIOUL":
            coded_ner = to_bioul(ner_tags,
                                 encoding=self._original_coding_scheme) if ner_tags is not None else None
        else:
            # the default IOB1
            coded_ner = ner_tags

        # Add "feature labels" to instance
        if 'ner' in self.feature_labels:
            if coded_ner is None:
                raise ConfigurationError("Dataset reader was specified to use NER tags as "
                                         " features. Pass them to text_to_instance.")
            instance_fields['ner_tags'] = SequenceLabelField(coded_ner, sequence, "ner_tags")

        # Add "tag label" to instance
        if self.tag_label == 'ner' and coded_ner is not None:
            instance_fields['tags'] = SequenceLabelField(coded_ner, sequence,
                                                         self.label_namespace)

        return Instance(instance_fields)
