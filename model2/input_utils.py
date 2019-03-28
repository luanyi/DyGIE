import numpy as np
import tensorflow as tf
import pdb

def tensorize_labeled_spans(tuples, label_dict):
  if len(tuples) > 0:
    starts, ends, labels = zip(*tuples)
  else:
    starts, ends, labels = [], [], []

  if label_dict:
    return np.array(starts), np.array(ends), np.array([label_dict.get(c, 0) for c in labels])

  return np.array(starts), np.array(ends), np.array(labels)


# FIXME
def tensorize_srl_relations(tuples, label_dict, filter_v_args):
  # Removing V-V self-loop.
  if filter_v_args: 
    filtered_tuples = [t for t in tuples if t[-1] not in ["V", "C-V"]]
  else:
    filtered_tuples = [t for t in tuples if t[-1] != "V"]
    # filtered_tuples = tuples
  if len(filtered_tuples) > 0:
    heads, starts, ends, labels = zip(*filtered_tuples)
  else:
    heads, starts, ends, labels = [], [], [], []
  return (np.array(heads), np.array(starts), np.array(ends),
          np.array([label_dict.get(c, 0) for c in labels]))


def tensorize_entity_relations(tuples, label_dict, filter_reverse):
  # Removing V-V self-loop.
  filtered_tuples = []
  for t in tuples:
    if filter_reverse and "REVERSE" in t[-1]:
      filtered_tuples.append(t[:-1] + [t[-1].split("_REVERSE")[0],])
    else:
      filtered_tuples.append(t)
  if len(filtered_tuples) > 0:
    s1, e1, s2, e2, labels = zip(*filtered_tuples)
  else:
    s1, e1, s2, e2, labels = [], [], [], [], []
  return (np.array(s1), np.array(e1), np.array(s2), np.array(e2),
          np.array([label_dict.get(c, 0) for c in labels]))


def get_all_predicates(tuples):
  if len(tuples) > 0:
    predicates, _, _, _ = zip(*tuples)
  else:
    predicates = []
  return np.unique(predicates)
  

def load_lm_embeddings_for_sentence(lm_file, lm_layers, lm_size, doc_key, sent_key,
                                    transpose):
  """ Load LM embeddings for given sentence.
  """
  if lm_file is None:
    return np.zeros([0, lm_size, lm_layers])  # FIXME: dimensions are wrong? Num. words, ...
  # file_key = doc_key.replace("/", ":")
  file_key = doc_key
  # print file_key
  # pdb.set_trace()
  group = lm_file[file_key]
  if sent_key is not None:
    sentence = group[sent_key][...]
  else:
    sentence = group[...]
  if transpose:
    return sentence.transpose(1, 2, 0)
  else:
    return sentence


def pad_batch_tensors(tensor_dicts, tensor_name):
  """
  Args:
    tensor_dicts: List of dictionary tensor_name: numpy array of length B.
    tensor_name: String name of tensor.
  
  Returns:
    Numpy array of (B, ?)
  """
  batch_size = len(tensor_dicts)
  tensors = [np.expand_dims(td[tensor_name], 0) for td in tensor_dicts]
  shapes = [t.shape for t in tensors]
  # Take max shape along each dimension.
  max_shape = np.max(zip(*shapes), axis=1)
  #print tensor_name, batch_size, tensors[0].shape, max_shape
  zeros = np.zeros_like(max_shape)
  padded_tensors = [np.pad(t, zip(zeros, max_shape - t.shape), "constant") for t in tensors]
  return np.concatenate(padded_tensors, axis=0)


def populate_sentence_offset(examples):
  # Compute sentence offset (that share the same doc key), because of LM embedding formatting difference.
  prev_doc_key = "XXX"
  sent_offset = 0
  for example in examples:
    doc_key = example["doc_key"][:example["doc_key"].rfind("_")]
    if doc_key != prev_doc_key:
      prev_doc_key = doc_key
      sent_offset = 0
    example["sent_offset"] = sent_offset
    sent_offset += len(example["sentences"])


def split_srl_labels(srl_labels, include_c_v):
  adjunct_role_labels = []
  core_role_labels = []
  for label in srl_labels:
    if "AM" in label or "ARGM" in label:
      adjunct_role_labels.append(label)
    elif label != "V" and (include_c_v or label != "C-V"):
      core_role_labels.append(label)
  return adjunct_role_labels, core_role_labels



