""" attribute value extrtaction fine-tuning: utilities to work  """
import torch
import logging
import os
import copy
import json
from .utils_attr import DataProcessor
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, title, attribute,labels):
        self.guid = guid
        self.title = title
        self.attribute = attribute
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())
    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, t_input_ids, a_input_ids,a_input_mask, a_input_len,
                 a_segment_ids,t_input_mask,t_input_len,t_segment_ids, label_ids):

        self.a_input_ids = a_input_ids
        self.a_input_mask = a_input_mask
        self.a_segment_ids = a_segment_ids
        self.a_input_len = a_input_len
        self.label_ids = label_ids
        self.t_input_len = t_input_len
        self.t_input_ids = t_input_ids
        self.t_input_mask = t_input_mask
        self.t_segment_ids = t_segment_ids
        self.t_input_len = t_input_len

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    t_all_input_ids, t_all_input_mask, t_all_segment_ids, t_all_lens, \
    a_all_input_ids, a_all_input_mask, a_all_segment_ids, a_all_lens, all_label_ids = map(torch.stack, zip(*batch))
    t_max_len = max(t_all_lens).item()
    a_max_len = max(a_all_lens).item()

    t_all_input_ids = t_all_input_ids[:, :t_max_len]
    t_all_input_mask = t_all_input_mask[:, :t_max_len]
    t_all_segment_ids = t_all_segment_ids[:, :t_max_len]
    all_label_ids = all_label_ids[:,:t_max_len]

    a_all_input_ids = a_all_input_ids[:, :a_max_len]
    a_all_input_mask = a_all_input_mask[:, :a_max_len]
    a_all_segment_ids = a_all_segment_ids[:, :a_max_len]

    return t_all_input_ids, t_all_input_mask, t_all_segment_ids, t_all_lens, \
    a_all_input_ids, a_all_input_mask, a_all_segment_ids, a_all_lens, all_label_ids

def convert_examples_to_features(examples,label_list,max_seq_length,tokenizer,max_attr_length,
                                 cls_token_at_end=False,cls_token="[CLS]",cls_token_segment_id=1,
                                 sep_token="[SEP]",pad_on_left=False,pad_token=0,pad_token_segment_id=0,
                                 sequence_a_segment_id=0,mask_padding_with_zero=True,):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        t_tokens = tokenizer.tokenize(example.title)
        a_tokens = tokenizer.tokenize(example.attribute)
        label_ids = [label_map[x] for x in example.labels]
        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(t_tokens) > max_seq_length - special_tokens_count:
            t_tokens = t_tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]
        if len(a_tokens) > max_attr_length - special_tokens_count:
            a_tokens = a_tokens[: (max_attr_length - special_tokens_count)]
        t_tokens += [sep_token]
        a_tokens += [sep_token]
        label_ids += [label_map[sep_token]]
        t_segment_ids = [sequence_a_segment_id] * len(t_tokens)
        a_segment_ids = [sequence_a_segment_id] * len(a_tokens)
        if cls_token_at_end:
            t_tokens += [cls_token]
            label_ids += [label_map[cls_token]]
            t_segment_ids += [cls_token_segment_id]
            a_tokens += [cls_token]
            a_segment_ids += [cls_token_segment_id]
        else:
            t_tokens = [cls_token] + t_tokens
            label_ids = [label_map[cls_token]] + label_ids
            t_segment_ids = [cls_token_segment_id] + t_segment_ids
            a_tokens = [cls_token] + a_tokens
            a_segment_ids = [cls_token_segment_id] + a_segment_ids

        t_input_ids = tokenizer.convert_tokens_to_ids(t_tokens)
        a_input_ids = tokenizer.convert_tokens_to_ids(a_tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        t_input_mask = [1 if mask_padding_with_zero else 0] * len(t_input_ids)
        a_input_mask = [1 if mask_padding_with_zero else 0] * len(a_input_ids)
        t_input_len = len(t_input_ids)
        a_input_len = len(a_input_ids)
        # Zero-pad up to the sequence length.
        t_padding_length = max_seq_length - len(t_input_ids)
        a_padding_length = max_attr_length - len(a_input_ids)
        if pad_on_left:
            t_input_ids = ([pad_token] * t_padding_length) + t_input_ids
            t_input_mask = ([0 if mask_padding_with_zero else 1] * t_padding_length) + t_input_mask
            t_segment_ids = ([pad_token_segment_id] * t_padding_length) + t_segment_ids
            label_ids = ([pad_token] * t_padding_length) + label_ids

            a_input_ids = ([pad_token] * a_padding_length) + a_input_ids
            a_input_mask = ([0 if mask_padding_with_zero else 1] * a_padding_length) + a_input_mask
            a_segment_ids = ([pad_token_segment_id] * a_padding_length) + a_segment_ids
        else:
            t_input_ids += [pad_token] * t_padding_length
            t_input_mask += [0 if mask_padding_with_zero else 1] * t_padding_length
            t_segment_ids += [pad_token_segment_id] * t_padding_length
            label_ids += [pad_token] * t_padding_length

            a_input_ids += [pad_token] * a_padding_length
            a_input_mask += [0 if mask_padding_with_zero else 1] * a_padding_length
            a_segment_ids += [pad_token_segment_id] * a_padding_length
        assert len(t_input_ids) == max_seq_length
        assert len(t_input_mask) == max_seq_length
        assert len(t_segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(a_input_ids) == max_attr_length
        assert len(a_input_mask) == max_attr_length
        assert len(a_segment_ids) == max_attr_length
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens_title: %s", " ".join([str(x) for x in t_tokens]))
            logger.info("input_ids_title: %s", " ".join([str(x) for x in t_input_ids]))
            logger.info("input_mask_title: %s", " ".join([str(x) for x in t_input_mask]))
            logger.info("segment_ids_title: %s", " ".join([str(x) for x in t_segment_ids]))
            logger.info("tokens_attr: %s", " ".join([str(x) for x in a_tokens]))
            logger.info("input_ids_attr: %s", " ".join([str(x) for x in a_input_ids]))
            logger.info("input_mask_attr: %s", " ".join([str(x) for x in a_input_mask]))
            logger.info("segment_ids_attr: %s", " ".join([str(x) for x in a_segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(InputFeatures(t_input_ids=t_input_ids,
                                      t_input_mask=t_input_mask,
                                      t_input_len = t_input_len,
                                      t_segment_ids=t_segment_ids,
                                      a_input_ids=a_input_ids,
                                      a_input_mask=a_input_mask,
                                      a_input_len=a_input_len,
                                      a_segment_ids=a_segment_ids,
                                      label_ids=label_ids))
    return features

class AttrProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "train_sample.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev_sample.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_json(os.path.join(data_dir, "test_sample.json")), "test")

    def get_labels(self):
        """See base class."""
        return ["X", "B-a","I-a",'S-a','O',"[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            title= line['title']
            attribute = line['attr']
            labels = line['labels']
            examples.append(InputExample(guid=guid, title=title, attribute = attribute,labels=labels))
        return examples
ner_processors = {
    'attr':AttrProcessor
}
