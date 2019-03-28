import math
import numpy as np
import tensorflow as tf
import util
import random
import srl_ops

def flatten_emb(emb):
  num_sentences = tf.shape(emb)[0]
  max_sentence_length = tf.shape(emb)[1]
  emb_rank = len(emb.get_shape())
  if emb_rank  == 2:
    flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
  elif emb_rank == 3:
    flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
  else:
    raise ValueError("Unsupported rank: {}".format(emb_rank))
  return flattened_emb


def flatten_emb_by_sentence(emb, text_len_mask):
  num_sentences = tf.shape(emb)[0]
  max_sentence_length = tf.shape(emb)[1]
  flattened_emb = flatten_emb(emb)
  return tf.boolean_mask(flattened_emb,
                         tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))


def batch_gather(emb, indices):
  # TODO: Merge with util.batch_gather.
  """
  Args:
    emb: Shape of [num_sentences, max_sentence_length, (emb)]
    indices: Shape of [num_sentences, k, (l)]
  """
  num_sentences = tf.shape(emb)[0] 
  max_sentence_length = tf.shape(emb)[1] 
  flattened_emb = flatten_emb(emb)  # [num_sentences * max_sentence_length, emb]
  offset = tf.expand_dims(tf.range(num_sentences) * max_sentence_length, 1)  # [num_sentences, 1]
  if len(indices.get_shape()) == 3:
    offset = tf.expand_dims(offset, 2)  # [num_sentences, 1, 1]
  return tf.gather(flattened_emb, indices + offset)

def batch_gather_pyfunc(emb, indices):
  # TODO: Merge with util.batch_gather.
  """
  Args:
    emb: Shape of [num_sentences, max_sentence_length, (emb)]
    indices: Shape of [num_sentences, k, (l)]
  """
  num_sentences = tf.shape(emb)[0] 
  max_sentence_length = tf.shape(emb)[1] 
  flattened_emb = flatten_emb(emb)  # [num_sentences * max_sentence_length, emb]
  offset = tf.expand_dims(tf.range(num_sentences) * max_sentence_length, 1)  # [num_sentences, 1]
  return tf.gather(flattened_emb, indices + offset) 
  

def lstm_contextualize(text_emb, text_len, config, lstm_dropout):
  num_sentences = tf.shape(text_emb)[0]
  current_inputs = text_emb  # [num_sentences, max_sentence_length, emb]
  for layer in xrange(config["contextualization_layers"]):
    with tf.variable_scope("layer_{}".format(layer)):
      with tf.variable_scope("fw_cell"):
        cell_fw = util.CustomLSTMCell(config["contextualization_size"], num_sentences, lstm_dropout)
      with tf.variable_scope("bw_cell"):
        cell_bw = util.CustomLSTMCell(config["contextualization_size"], num_sentences, lstm_dropout)
      state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]),
                                               tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
      state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]),
                                               tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))
      (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
          cell_fw=cell_fw,
          cell_bw=cell_bw,
          inputs=current_inputs,
          sequence_length=text_len,
          initial_state_fw=state_fw,
          initial_state_bw=state_bw)
      text_outputs = tf.concat([fw_outputs, bw_outputs], 2)  # [num_sentences, max_sentence_length, emb]
      text_outputs = tf.nn.dropout(text_outputs, lstm_dropout)
      if layer > 0:
        highway_gates = tf.sigmoid(util.projection(
            text_outputs, util.shape(text_outputs, 2)))  # [num_sentences, max_sentence_length, emb]
        text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
      current_inputs = text_outputs

  return text_outputs  # [num_sentences, max_sentence_length, emb]


def get_span_candidates(text_len, max_sentence_length, max_mention_width):
  """
  Args:
    text_len: Tensor of [num_sentences,]
    max_sentence_length: Integer scalar.
    max_mention_width: Integer.
  """
  num_sentences = util.shape(text_len, 0)
  candidate_starts = tf.tile(
      tf.expand_dims(tf.expand_dims(tf.range(max_sentence_length), 0), 1),
      [num_sentences, max_mention_width, 1])  # [num_sentences, max_mention_width, max_sentence_length]
  candidate_widths = tf.expand_dims(tf.expand_dims(tf.range(max_mention_width), 0), 2)  # [1, max_mention_width, 1]
  candidate_ends = candidate_starts + candidate_widths  # [num_sentences, max_mention_width, max_sentence_length]
  
  candidate_starts = tf.reshape(candidate_starts, [num_sentences, max_mention_width * max_sentence_length])
  candidate_ends = tf.reshape(candidate_ends, [num_sentences, max_mention_width * max_sentence_length])
  candidate_mask = tf.less(
      candidate_ends,
      tf.tile(tf.expand_dims(text_len, 1), [1, max_mention_width * max_sentence_length])
  )  # [num_sentences, max_mention_width * max_sentence_length]

  # Mask to avoid indexing error.
  candidate_starts = tf.multiply(candidate_starts, tf.to_int32(candidate_mask))
  candidate_ends = tf.multiply(candidate_ends, tf.to_int32(candidate_mask))
  return candidate_starts, candidate_ends, candidate_mask  


def get_span_emb(head_emb, context_outputs, span_starts, span_ends, config, dropout):
  """Compute span representation shared across tasks.
   
  Args:
    head_emb: Tensor of [num_words, emb]
    context_outputs: Tensor of [num_words, emb]
    span_starts: [num_spans]
    span_ends: [num_spans]
  """
  text_length = util.shape(context_outputs, 0)
  num_spans = util.shape(span_starts, 0)

  span_start_emb = tf.gather(context_outputs, span_starts)  # [num_words, emb]
  span_end_emb = tf.gather(context_outputs, span_ends)  # [num_words, emb]
  span_emb_list = [span_start_emb, span_end_emb]

  span_width = 1 + span_ends - span_starts # [num_spans]
  max_arg_width = config["max_arg_width"]
  num_heads = config["num_attention_heads"]

  if config["use_features"]:
    span_width_index = span_width - 1  # [num_spans]
    span_width_emb = tf.gather(
        tf.get_variable("span_width_embeddings", [max_arg_width, config["feature_size"]]),
        span_width_index)  # [num_spans, emb]
    span_width_emb = tf.nn.dropout(span_width_emb, dropout)
    span_emb_list.append(span_width_emb)

  head_scores = None
  span_text_emb = None
  span_indices = None
  span_indices_log_mask = None

  if config["model_heads"]:
    span_indices = tf.minimum(
        tf.expand_dims(tf.range(max_arg_width), 0) + tf.expand_dims(span_starts, 1),
        text_length - 1)  # [num_spans, max_span_width]
    span_text_emb = tf.gather(head_emb, span_indices)  # [num_spans, max_arg_width, emb]
    span_indices_log_mask = tf.log(
        tf.sequence_mask(span_width, max_arg_width, dtype=tf.float32)) # [num_spans, max_arg_width]
    with tf.variable_scope("head_scores"):
      head_scores = util.projection(context_outputs, num_heads)  # [num_words, num_heads]
    span_attention = tf.nn.softmax(
      tf.gather(head_scores, span_indices) + tf.expand_dims(span_indices_log_mask, 2),
      dim=1)  # [num_spans, max_arg_width, num_heads]
    span_head_emb = tf.reduce_sum(span_attention * span_text_emb, 1)  # [num_spans, emb]
    span_emb_list.append(span_head_emb)

  span_emb = tf.concat(span_emb_list, 1) # [num_spans, emb]
  return span_emb, head_scores, span_text_emb, span_indices, span_indices_log_mask


def get_unary_scores(span_emb, config, dropout, num_labels = 1, name="span_scores"):
  """Compute span score with FFNN(span embedding).
  Args:
    span_emb: Tensor of [num_sentences, num_spans, emb].
  """
  with tf.variable_scope(name):
    scores = util.ffnn(span_emb, config["ffnn_depth"], config["ffnn_size"], num_labels,
                       dropout)  # [num_sentences, num_spans, num_labels] or [k, num_labels]
  if num_labels == 1:
    scores = tf.squeeze(scores, -1)  # [num_sentences, num_spans] or [k]
  return scores


def get_srl_scores(arg_emb, pred_emb, arg_scores, pred_scores, num_labels, config, dropout):
  num_sentences = util.shape(arg_emb, 0)
  num_args = util.shape(arg_emb, 1)
  num_preds = util.shape(pred_emb, 1)

  arg_emb_expanded = tf.expand_dims(arg_emb, 2)  # [num_sents, num_args, 1, emb]
  pred_emb_expanded = tf.expand_dims(pred_emb, 1)  # [num_sents, 1, num_preds, emb] 
  arg_emb_tiled = tf.tile(arg_emb_expanded, [1, 1, num_preds, 1])  # [num_sentences, num_args, num_preds, emb]
  pred_emb_tiled = tf.tile(pred_emb_expanded, [1, num_args, 1, 1])  # [num_sents, num_args, num_preds, emb]

  pair_emb_list = [arg_emb_tiled, pred_emb_tiled]
  pair_emb = tf.concat(pair_emb_list, 3)  # [num_sentences, num_args, num_preds, emb]
  pair_emb_size = util.shape(pair_emb, 3)
  flat_pair_emb = tf.reshape(pair_emb, [num_sentences * num_args * num_preds, pair_emb_size])

  flat_srl_scores = get_unary_scores(flat_pair_emb, config, dropout, num_labels - 1,
      "predicate_argument_scores")  # [num_sentences * num_args * num_predicates, 1]
  srl_scores = tf.reshape(flat_srl_scores, [num_sentences, num_args, num_preds, num_labels - 1])
  srl_scores += tf.expand_dims(tf.expand_dims(arg_scores, 2), 3) + tf.expand_dims(
      tf.expand_dims(pred_scores, 1), 3)  # [num_sentences, 1, max_num_preds, num_labels-1]
  
  dummy_scores = tf.zeros([num_sentences, num_args, num_preds, 1], tf.float32)
  srl_scores = tf.concat([dummy_scores, srl_scores], 3)  # [num_sentences, max_num_args, max_num_preds, num_labels] 
  return srl_scores  # [num_sentences, num_args, num_predicates, num_labels]


def get_rel_scores(entity_emb, entity_scores, num_labels, config, dropout, num_predicted_entities):
  num_sentences = util.shape(entity_emb, 0)
  num_entities = util.shape(entity_emb, 1)
  entities_mask = tf.sequence_mask(num_predicted_entities, num_entities) #[num_sentences, num_entities]
  flat_entities_mask = tf.reshape(entities_mask, [-1]) 
  rel_mask = tf.logical_and(tf.expand_dims(entities_mask, 2),  # [num_sentences, max_num_entities, 1]
                                            tf.expand_dims(entities_mask, 1)  # [num_sentences, 1, max_num_entities]
  )
  e1_emb_expanded = tf.expand_dims(entity_emb, 2)  # [num_sents, num_ents, 1, emb]
  e2_emb_expanded = tf.expand_dims(entity_emb, 1)  # [num_sents, 1, num_ents, emb]
  e1_emb_tiled = tf.tile(e1_emb_expanded, [1, 1, num_entities, 1])  # [num_sents, num_ents, num_ents, emb]
  e2_emb_tiled = tf.tile(e2_emb_expanded, [1, num_entities, 1, 1])  # [num_sents, num_ents, num_ents, emb]
  

  similarity_emb = e1_emb_expanded * e2_emb_expanded  # [num_sents, num_ents, num_ents, emb]

  # if config['add_ner_emb']:
  #   pair_emb_list = [ner1_emb_tiled, ner2_emb_tiled, e1_emb_tiled, e2_emb_tiled, similarity_emb]
  # else:
  pair_emb_list = [e1_emb_tiled, e2_emb_tiled, similarity_emb]
  # pair_emb_list = [e1_emb_tiled, e2_emb_tiled, similarity_emb]
  pair_emb = tf.concat(pair_emb_list, 3)  # [num_sentences, num_ents, num_ents, emb]
  pair_emb_size = util.shape(pair_emb, 3)
  flat_pair_emb = tf.reshape(pair_emb, [num_sentences * num_entities * num_entities, pair_emb_size])

  flat_rel_scores = get_unary_scores(flat_pair_emb, config, dropout, num_labels - 1,
      "relation_scores")  # [num_sentences * num_ents * num_ents, num_labels-1]
  rel_scores = tf.reshape(flat_rel_scores, [num_sentences, num_entities, num_entities, num_labels - 1])
  rel_scores += tf.expand_dims(tf.expand_dims(entity_scores, 2), 3) + tf.expand_dims(
      tf.expand_dims(entity_scores, 1), 3)  # [num_sentences, ents, max_num_ents, num_labels-1]
  if config['rel_prop']:
    flat_rel_scores = tf.reshape(rel_scores, [num_sentences * num_entities* num_entities, num_labels - 1])
    with tf.variable_scope("rel_W"):
      entity_emb_size = util.shape(entity_emb, -1)
      relation_transition = util.projection(tf.nn.relu(flat_rel_scores), entity_emb_size)
      e2_emb_tiled = tf.reshape(e2_emb_tiled, [num_sentences * num_entities * num_entities, entity_emb_size])
      rel_mask = tf.reshape(rel_mask, [-1])
      tranformed_embeddings = tf.multiply(tf.transpose(relation_transition * e2_emb_tiled), tf.to_float(rel_mask)) #[entity_emb_size, num_sents * num_ents * num_ents]
      tranformed_embeddings = tf.transpose(tranformed_embeddings) # [entity_emb_size, num_sents * num_ents * num_ents]
      tranformed_embeddings = tf.reshape(tranformed_embeddings, [num_sentences, num_entities, num_entities, entity_emb_size]) #[num_sents, num_ents, num_ents, entity_emb_size] 
      tranformed_embeddings = tf.reduce_sum(tranformed_embeddings, 2) #[num_sents, num_ents, entity_emb_size]
      tranformed_embeddings = tf.reshape(tranformed_embeddings, [num_sentences * num_entities, entity_emb_size])
      entity_emb = tf.reshape(entity_emb, [num_sentences * num_entities, entity_emb_size]) 
      with tf.variable_scope("f"):
        f = tf.sigmoid(util.projection(tf.concat([tranformed_embeddings, entity_emb], 1), entity_emb_size)) # [num_sents * num_ents, entity_emb_size]
        entity_emb = f * tranformed_embeddings + (1 - f) * entity_emb # [num_sents * num_ents, entity_emb_size]
      entity_emb = tf.reshape(entity_emb, [num_sentences, num_entities, entity_emb_size])
      
  dummy_scores = tf.zeros([num_sentences, num_entities, num_entities, 1], tf.float32)
  rel_scores = tf.concat([dummy_scores, rel_scores], 3)  # [num_sentences, max_num_ents, max_num_ents, num_labels]
  if config['rel_prop']:
    return rel_scores, entity_emb, flat_entities_mask
  else:
    return rel_scores  # [num_sentences, num_entities, num_entities, num_labels]

def get_rel_nonzero_scores(entity_emb, entity_scores, num_labels, config, dropout):
  num_sentences = util.shape(entity_emb, 0)
  num_entities = util.shape(entity_emb, 1)

    
  e1_emb_expanded = tf.expand_dims(entity_emb, 2)  # [num_sents, num_ents, 1, emb]
  e2_emb_expanded = tf.expand_dims(entity_emb, 1)  # [num_sents, 1, num_ents, emb]
  e1_emb_tiled = tf.tile(e1_emb_expanded, [1, 1, num_entities, 1])  # [num_sents, num_ents, num_ents, emb]
  e2_emb_tiled = tf.tile(e2_emb_expanded, [1, num_entities, 1, 1])  # [num_sents, num_ents, num_ents, emb]
  

  similarity_emb = e1_emb_expanded * e2_emb_expanded  # [num_sents, num_ents, num_ents, emb]

  # if config['add_ner_emb']:
  #   pair_emb_list = [ner1_emb_tiled, ner2_emb_tiled, e1_emb_tiled, e2_emb_tiled, similarity_emb]
  # else:
  pair_emb_list = [e1_emb_tiled, e2_emb_tiled, similarity_emb]
  # pair_emb_list = [e1_emb_tiled, e2_emb_tiled, similarity_emb]
  pair_emb = tf.concat(pair_emb_list, 3)  # [num_sentences, num_ents, num_ents, emb]
  pair_emb_size = util.shape(pair_emb, 3)
  flat_pair_emb = tf.reshape(pair_emb, [num_sentences * num_entities * num_entities, pair_emb_size])

  flat_rel_scores = get_unary_scores(flat_pair_emb, config, dropout, num_labels,
      "relation_scores")  # [num_sentences * num_ents * num_ents, 1]
  rel_scores = tf.reshape(flat_rel_scores, [num_sentences, num_entities, num_entities, num_labels])
  rel_scores += tf.expand_dims(tf.expand_dims(entity_scores, 2), 3) + tf.expand_dims(
      tf.expand_dims(entity_scores, 1), 3)  # [num_sentences, ents, max_num_ents, num_labels-1]

  return rel_scores  # [num_sentences, num_entities, num_entities, num_labels]

def get_rel_bilinear_scores(entity_emb, entity_scores, num_labels, config, dropout):
  num_sentences = util.shape(entity_emb, 0)
  num_entities = util.shape(entity_emb, 1)

    
  e1_emb_expanded = tf.expand_dims(entity_emb, 2)  # [num_sents, num_ents, 1, emb]
  e2_emb_expanded = tf.expand_dims(entity_emb, 1)  # [num_sents, 1, num_ents, emb]
  e1_emb_tiled = tf.tile(e1_emb_expanded, [1, 1, num_entities, 1])  # [num_sents, num_ents, num_ents, emb]
  e2_emb_tiled = tf.tile(e2_emb_expanded, [1, num_entities, 1, 1])  # [num_sents, num_ents, num_ents, emb]
  
  bilinear_score = util.bilinear(e1_emb_tiled, e2_emb_tiled, num_labels - 1) # [num_sents, num_ents, num_ents, bilinear_dim]


  pair_emb_list = [e1_emb_tiled, e2_emb_tiled]
  pair_emb = tf.concat(pair_emb_list, 3)  # [num_sentences, num_ents, num_ents, emb]
  pair_emb_size = util.shape(pair_emb, 3)
  # flat_pair_emb = tf.reshape(pair_emb, [num_sentences * num_entities * num_entities, pair_emb_size])
  flat_rel_scores = bilinear_score
  # flat_rel_scores = get_unary_scores(flat_pair_emb, config, dropout, num_labels - 1,
  #     "relation_scores")  # [num_sentences * num_ents * num_ents, 1]

  # flat_rel_scores += bilinear_score
  rel_scores = tf.reshape(flat_rel_scores, [num_sentences, num_entities, num_entities, num_labels - 1])
  rel_scores += tf.expand_dims(tf.expand_dims(entity_scores, 2), 3) + tf.expand_dims(
      tf.expand_dims(entity_scores, 1), 3)  # [num_sentences, ents, max_num_ents, num_labels-1]
  
  dummy_scores = tf.zeros([num_sentences, num_entities, num_entities, 1], tf.float32)
  rel_scores = tf.concat([dummy_scores, rel_scores], 3)  # [num_sentences, max_num_ents, max_num_ents, num_labels] 
  return rel_scores  # [num_sentences, num_entities, num_entities, num_labels]



def get_batch_topk(candidate_starts, candidate_ends, candidate_scores, topk_ratio, text_len,
                   max_sentence_length, sort_spans=False, enforce_non_crossing=True):
  """
  Args:
    candidate_starts: [num_sentences, max_num_candidates]
    candidate_mask: [num_sentences, max_num_candidates]
    topk_ratio: A float number.
    text_len: [num_sentences,]
    max_sentence_length:
    enforce_non_crossing: Use regular top-k op if set to False.
 """
  num_sentences = util.shape(candidate_starts, 0)
  max_num_candidates = util.shape(candidate_starts, 1)

  topk = tf.maximum(tf.to_int32(tf.floor(tf.to_float(text_len) * topk_ratio)),
                    tf.ones([num_sentences,], dtype=tf.int32))  # [num_sentences]

  predicted_indices = srl_ops.extract_spans(
      candidate_scores, candidate_starts, candidate_ends, topk, max_sentence_length,
      sort_spans, enforce_non_crossing)  # [num_sentences, max_num_predictions]
  predicted_indices.set_shape([None, None])

  predicted_starts = batch_gather(candidate_starts, predicted_indices)  # [num_sentences, max_num_predictions]
  predicted_ends = batch_gather(candidate_ends, predicted_indices)  # [num_sentences, max_num_predictions]
  predicted_scores = batch_gather(candidate_scores, predicted_indices)  # [num_sentences, max_num_predictions]

  return predicted_starts, predicted_ends, predicted_scores, topk, predicted_indices

  
def get_ner_spans_func(sorted_indices, candidate_labels, top_num_entities):
  num_sentences, num_candidates = candidate_labels.shape
  max_num_entities = max(top_num_entities)
  predicted_tensor_batch = np.zeros([num_sentences, max_num_entities], dtype=np.int32)
  num_entities = np.zeros([num_sentences], dtype=np.int32)
  for i in range(num_sentences):
    idx = 0
    for j in range(num_candidates):
      cur_idx = sorted_indices[i,j]
      if idx < max_num_entities and candidate_labels[i,cur_idx] != 0:
        predicted_tensor_batch[i,idx] = cur_idx
        idx += 1
    lastidx = idx
    num_entities[i] = lastidx
    for idx in range(lastidx, max_num_entities):
      predicted_tensor_batch[i,idx] = predicted_tensor_batch[i,lastidx-1]
  return predicted_tensor_batch, num_entities
      
def get_ner_spans(sorted_indices, candidate_labels, max_num_entities):
  predicted_indices, num_entities = tf.py_func(
    get_ner_spans_func, [sorted_indices, candidate_labels, max_num_entities], (tf.int32, tf.int32)
  ) #[num_sentences, max_num_entities]
  predicted_indices.set_shape([None, None])
  return predicted_indices, num_entities
# def get_ner_spans_func(predicted_indices, num_sentences):
#   num_entities, _ = predicted_indices.shape
#   max_entities = np.zeros([num_sentences], dtype=np.int32)
#   for i in range(num_entities):
#     max_entities[num_entities[i, 0]] += 1
#   max_num_entities = np.max(max_entities) + 1
#   predicted_tensor_batch = np.zeros([num_sentences, max_num_entities], dtype=np.int32)
#   last_indices = np.zeros([num_sentences], dtype=np.int32)
  
#   for i in range(num_entities):
#     sentid = num_entities[i, 0]
#     idx = num_entities[i, 1]
#     predicted_tensor_batch[sentid, last_indices[sentid]] = idx
#     last_indices[sentid] += 1
    
#   for i in range(num_sentences):
#     while last_indices[sentid] < max_entities:
#       predicted_tensor_batch[sentid, last_indices[sentid]] = max_entities[sentid]-1
#       last_indices[sentid] += 1 
#   return predicted_tensor_batch, max_num_entities # [num_sentences, max_num_entities]
      
    
def get_ner_candidates(candidate_starts, candidate_ends, candidate_scores, candidate_mask, text_len, topk_ratio):
  """
  Args:
    candidate_starts: [num_sentences, max_num_candidates]
    candidate_mask: [num_sentences, max_num_candidates]
    candidate_scores: [num_sentences, max_num_candidates, num_labels]
 """
  candidate_scores = tf.nn.softmax(candidate_scores, axis = -1)
  num_sentences, max_num_ents, num_labels = candidate_scores.shape
  num_sentences = util.shape(candidate_starts, 0)
  topk = tf.maximum(tf.to_int32(tf.floor(tf.to_float(text_len) * topk_ratio)), tf.ones([num_sentences,], dtype=tf.int32))  # [num_sentences]                                                          
  
  candidate_labels = tf.argmax(candidate_scores, -1, output_type=tf.int32) #[num_sentences, max_num_candidates]
  candidate_labels = tf.multiply(candidate_labels, tf.to_int32(candidate_mask)) # [num_sentences, max_num_candidates]
  candidate_maxscores = tf.reduce_max(candidate_scores, reduction_indices=[-1]) # [num_sentences, max_num_candidates]
  # flat_candidate_labels = tf.reshape(candidate_labels, [-1]) # [num_sentences * max_num_candidates]
  # flat_candidate_maxscores = tf.reshape(candidate_maxscores, [-1])
  sorted_indices = tf.contrib.framework.argsort(candidate_maxscores, direction='DESCENDING', axis=-1)
  
  # zero = tf.constant(0, dtype=tf.int32)
  # where = tf.not_equal(candidate_labels, zero)
  # num_entities = tf.reduce_sum(tf.to_int32(where), reduction_indices=[-1])
  # max_entities = tf.reduce_max(num_entities)
  # predicted_indices = tf.where(where, out_type=tf.int32) #[num_entities]
  predicted_indices, num_entities = get_ner_spans(sorted_indices, candidate_labels, topk) #[num_sentences, max_num_entities]
  # predicted_indices = get_ner_spans(candidate_labels, topk) #[num_sentences, max_num_entities]
  predicted_starts = batch_gather(candidate_starts, predicted_indices)
  predicted_ends = batch_gather(candidate_ends, predicted_indices)
  predicted_scores = batch_gather(candidate_maxscores, predicted_indices)
  
  return predicted_starts, predicted_ends, predicted_scores, num_entities, predicted_indices


def get_srl_labels(arg_starts, arg_ends, predicates, labels, max_sentence_length):
  """
  Args:
    arg_starts: [num_sentences, max_num_args]
    arg_ends: [num_sentences, max_num_args]
    predicates: [num_sentences, max_num_predicates]
    labels: Dictionary of label tensors.
    max_sentence_length: An integer scalar.
  """
  num_sentences = util.shape(arg_starts, 0)
  max_num_args = util.shape(arg_starts, 1)
  max_num_preds = util.shape(predicates, 1)
  sentence_indices_2d = tf.tile(
      tf.expand_dims(tf.expand_dims(tf.range(num_sentences), 1), 2),
      [1, max_num_args, max_num_preds])  # [num_sentences, max_num_args, max_num_preds]
  tiled_arg_starts = tf.tile(
      tf.expand_dims(arg_starts, 2),
      [1, 1, max_num_preds])  # [num_sentences, max_num_args, max_num_preds]
  tiled_arg_ends = tf.tile(
      tf.expand_dims(arg_ends, 2),
      [1, 1, max_num_preds])  # [num_sentences, max_num_args, max_num_preds]
  tiled_predicates = tf.tile(
      tf.expand_dims(predicates, 1),
      [1, max_num_args, 1])  # [num_sentences, max_num_args, max_num_preds]
  pred_indices = tf.concat([
      tf.expand_dims(sentence_indices_2d, 3),
      tf.expand_dims(tiled_arg_starts, 3),
      tf.expand_dims(tiled_arg_ends, 3),
      tf.expand_dims(tiled_predicates, 3)], axis=3)  # [num_sentences, max_num_args, max_num_preds, 4]
 
  dense_srl_labels = get_dense_span_labels(
      labels["arg_starts"], labels["arg_ends"], labels["arg_labels"], labels["srl_len"], max_sentence_length,
      span_parents=labels["predicates"])  # [num_sentences, max_sent_len, max_sent_len, max_sent_len]
 
  srl_labels = tf.gather_nd(params=dense_srl_labels, indices=pred_indices)  # [num_sentences, max_num_args]
  return srl_labels


def get_relation_labels_func(entity_starts, entity_ends, num_entities, max_sentence_length,
                             gold_e1_starts, gold_e1_ends, gold_e2_starts, gold_e2_ends,
                             gold_labels, num_gold_rels):
  """Might be slow ..
  """
  num_sentences, max_num_ents = entity_starts.shape
  rel_labels = np.zeros([num_sentences, max_num_ents + 1, max_num_ents + 1], dtype=np.int32)
  entity_ids = np.zeros([num_sentences, max_sentence_length, max_sentence_length], dtype=np.int32)
  for i in range(num_sentences):
    for j in range(num_entities[i]):
      entity_ids[i, entity_starts[i,j], entity_ends[i,j]] = j + 1
    for j in range(num_gold_rels[i]):
      rel_labels[i, entity_ids[i, gold_e1_starts[i,j], gold_e1_ends[i,j]],
                 entity_ids[i, gold_e2_starts[i,j], gold_e2_ends[i,j]]] = gold_labels[i,j]
  return rel_labels[:,1:,1:]  # Remove "dummy" entities.
    

def get_relation_labels(entity_starts, entity_ends, num_entities, labels, max_sentence_length):
  #e2_starts = tf.Print(labels["rel_e2_starts"],
  #                     [labels["doc_id"], labels["rel_e2_starts"], labels["rel_e2_ends"], max_sentence_length], summarize=50)
  rel_labels = tf.py_func(
      get_relation_labels_func,
      [entity_starts, entity_ends, num_entities, max_sentence_length, 
       labels["rel_e1_starts"], labels["rel_e1_ends"], labels["rel_e2_starts"], labels["rel_e2_ends"],
       labels["rel_labels"], labels["rel_len"]], tf.int32)
  rel_labels.set_shape([None, None])
  return rel_labels


def get_span_task_labels(arg_starts, arg_ends, labels, max_sentence_length):
  """Get dense labels for NER/Constituents (unary span prediction tasks).
  """
  num_sentences = util.shape(arg_starts, 0)
  max_num_args = util.shape(arg_starts, 1)
  sentence_indices = tf.tile(
      tf.expand_dims(tf.range(num_sentences), 1),
      [1, max_num_args])  # [num_sentences, max_num_args]
  pred_indices = tf.concat([
      tf.expand_dims(sentence_indices, 2),
      tf.expand_dims(arg_starts, 2),
      tf.expand_dims(arg_ends, 2)], axis=2)  # [num_sentences, max_num_args, 3]
  
  dense_ner_labels = get_dense_span_labels(
      labels["ner_starts"], labels["ner_ends"], labels["ner_labels"], labels["ner_len"],
      max_sentence_length)  # [num_sentences, max_sent_len, max_sent_len]
  dense_coref_labels = get_dense_span_labels(
      labels["coref_starts"], labels["coref_ends"], labels["coref_cluster_ids"], labels["coref_len"],
      max_sentence_length)  # [num_sentences, max_sent_len, max_sent_len]

  ner_labels = tf.gather_nd(params=dense_ner_labels, indices=pred_indices)  # [num_sentences, max_num_args]
  coref_cluster_ids = tf.gather_nd(params=dense_coref_labels, indices=pred_indices)  # [num_sentences, max_num_args]
  return ner_labels, coref_cluster_ids
 

def get_dense_span_labels(span_starts, span_ends, span_labels, num_spans, max_sentence_length, span_parents=None):
  """Utility function to get dense span or span-head labels.
  Args:
    span_starts: [num_sentences, max_num_spans]
    span_ends: [num_sentences, max_num_spans]
    span_labels: [num_sentences, max_num_spans]
    num_spans: [num_sentences,]
    max_sentence_length:
    span_parents: [num_sentences, max_num_spans]. Predicates in SRL.
  """
  num_sentences = util.shape(span_starts, 0)
  max_num_spans = util.shape(span_starts, 1)
 
  # For padded spans, we have starts = 1, and ends = 0, so they don't collide with any existing spans.
  span_starts += (1 - tf.sequence_mask(num_spans, dtype=tf.int32))  # [num_sentences, max_num_spans]
  sentence_indices = tf.tile(
      tf.expand_dims(tf.range(num_sentences), 1),
      [1, max_num_spans])  # [num_sentences, max_num_spans]
  sparse_indices = tf.concat([
      tf.expand_dims(sentence_indices, 2),
      tf.expand_dims(span_starts, 2),
      tf.expand_dims(span_ends, 2)], axis=2)  # [num_sentences, max_num_spans, 3]
  if span_parents is not None:
    sparse_indices = tf.concat([
      sparse_indices, tf.expand_dims(span_parents, 2)], axis=2)  # [num_sentenes, max_num_spans, 4]

  rank = 3 if (span_parents is None) else 4
  # (sent_id, span_start, span_end) -> span_label
  dense_labels = tf.sparse_to_dense(
      sparse_indices = tf.reshape(sparse_indices, [num_sentences * max_num_spans, rank]),
      output_shape = [num_sentences] + [max_sentence_length] * (rank - 1),
      sparse_values = tf.reshape(span_labels, [-1]),
      default_value = 0,
      validate_indices = False)  # [num_sentences, max_sent_len, max_sent_len]
  return dense_labels

def get_negative_sample_mask_func(rel_labels, rel_scores, num_predicted_entities, randp):
  """Might be slow ..
  """
  num_sentences, max_num_ents, max_num_ents = rel_labels.shape
  rel_mask = np.full([num_sentences, max_num_ents, max_num_ents], False)

  for i in range(num_sentences):
    for j in range(max_num_ents):
      for m in range(max_num_ents):
        if j >= num_predicted_entities[i] or m >= num_predicted_entities[i]: #need to confirm
          continue
        if rel_mask[i,j,m] != 0:
          rel_mask[i,j,m] = True
        else:
          randnum = random.uniform(0, 1)
          if randnum <= randp:
            rel_mask[i,j,m] = True
        

  return rel_mask


def get_rel_softmax_loss(rel_scores, rel_labels, num_predicted_entities, config):
  """Softmax loss with 2-D masking.
  Args:
    rel_scores: [num_sentences, max_num_entities, max_num_entities, num_labels]
    rel_labels: [num_sentences, max_num_entities, max_num_entities]
    num_predicted_entities: [num_sentences]
  """
  max_num_entities = util.shape(rel_scores, 1)
  num_labels = util.shape(rel_scores, 3)
  entities_mask = tf.sequence_mask(num_predicted_entities, max_num_entities)  # [num_sentences, max_num_entities]
  randp = config['ns_randp']
  print "Negative sample rate: " + str(randp)
  negative_sample_mask = tf.py_func(get_negative_sample_mask_func, [rel_labels, rel_scores, num_predicted_entities, randp], tf.bool)
  rel_loss_mask = tf.logical_and(
      tf.expand_dims(entities_mask, 2),  # [num_sentences, max_num_entities, 1]
      tf.expand_dims(entities_mask, 1)  # [num_sentences, 1, max_num_entities]
  )  # [num_sentences, max_num_entities, max_num_entities]
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.reshape(rel_labels, [-1]),
      logits=tf.reshape(rel_scores, [-1, num_labels]),
      name="srl_softmax_loss")  # [num_sentences * max_num_args * max_num_preds]
  # loss = tf.boolean_mask(loss, tf.reshape(rel_loss_mask, [-1]))
  loss = tf.boolean_mask(loss, tf.reshape(negative_sample_mask, [-1]))
  loss.set_shape([None])
  loss = tf.reduce_sum(loss)
  return loss



def get_softmax_loss(scores, labels, candidate_mask):
  """Softmax loss with 1-D masking. (on Unary factors)
  Args:
    scores: [num_sentences, max_num_candidates, num_labels]
    labels: [num_sentences, max_num_candidates]
    candidate_mask: [num_sentences, max_num_candidates]
  """
  max_num_candidates = util.shape(scores, 1)
  num_labels = util.shape(scores, 2)
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.reshape(labels, [-1]), 
      logits=tf.reshape(scores, [-1, num_labels]),
      name="softmax_loss")  # [num_sentences, max_num_candidates]
  loss = tf.boolean_mask(loss, tf.reshape(candidate_mask, [-1]))
  loss.set_shape([None])
  return loss


def get_coref_softmax_loss(antecedent_scores, antecedent_labels):
  gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels))  # [k, max_ant + 1]
  marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1])  # [k]
  log_norm = tf.reduce_logsumexp(antecedent_scores, [1])  # [k]
  return log_norm - marginalized_gold_scores  # [k]


def get_antecedent_scores(top_span_emb, top_span_mention_scores, antecedents, config, dropout, top_fast_antecedent_scores, top_antecedent_offsets):
  k = util.shape(top_span_emb, 0)
  max_antecedents = util.shape(antecedents, 1)
  feature_emb_list = []


  if config["use_features"]:
    # target_indices = tf.range(k)  # [k]
    # antecedent_distance = tf.expand_dims(target_indices, 1) - antecedents  # [k, max_ant]
    # antecedent_distance_buckets = bucket_distance(antecedent_distance)  # [k, max_ant]
    antecedent_distance_buckets = bucket_distance(top_antecedent_offsets)
    with tf.variable_scope("features"):
      antecedent_distance_emb = tf.gather(
          tf.get_variable("antecedent_distance_emb", [10, config["feature_size"]]),
          antecedent_distance_buckets)  # [k, max_ant]
    feature_emb_list.append(antecedent_distance_emb)

  feature_emb = tf.concat(feature_emb_list, 2)  # [k, max_ant, emb]
  feature_emb = tf.nn.dropout(feature_emb, dropout)  # [k, max_ant, emb]

  antecedent_emb = tf.gather(top_span_emb, antecedents)  # [k, max_ant, emb]
  target_emb = tf.expand_dims(top_span_emb, 1)  # [k, 1, emb]
  similarity_emb = antecedent_emb * target_emb  # [k, max_ant, emb]
  target_emb = tf.tile(target_emb, [1, max_antecedents, 1])  # [k, max_ant, emb]
  pair_emb = tf.concat([target_emb, antecedent_emb, similarity_emb, feature_emb], 2)  # [k, max_ant, emb]
  with tf.variable_scope("antecedent_scores"):
    antecedent_scores = util.ffnn(pair_emb, config["ffnn_depth"], config["ffnn_size"], 1,
                                  dropout)  # [k, max_ant, 1]
    antecedent_scores = tf.squeeze(antecedent_scores, 2)  # [k, max_ant]
  # antecedent_scores += tf.expand_dims(top_span_mention_scores, 1) + tf.gather(
  #     top_span_mention_scores, antecedents)  # [k, max_ant]
  antecedent_scores += top_fast_antecedent_scores
  return antecedent_scores, antecedent_emb, pair_emb  # [k, max_ant]


def bucket_distance(distances):
  """
  Places the given values (designed for distances) into 10 semi-logscale buckets:
  [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
  """
  logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances))/math.log(2))) + 3
  use_identity = tf.to_int32(distances <= 4)
  combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
  return tf.clip_by_value(combined_idx, 0, 9)
  # return tf.minimum(combined_idx, 9)

def get_fast_antecedent_scores(top_span_emb, dropout):
    with tf.variable_scope("src_projection"):
      source_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, util.shape(top_span_emb, -1)), dropout) # [k, emb]
    target_top_span_emb = tf.nn.dropout(top_span_emb, dropout) # [k, emb]
    return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True) # [k, k]

def coarse_to_fine_pruning(top_span_emb, top_span_mention_scores, c, mention_doc_ids, dropout):
    k = util.shape(top_span_emb, 0)
    top_span_range = tf.range(k) # [k]
    antecedent_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0) # [k, k]
    antecedents_mask = antecedent_offsets >= 1 # [k, k]
    antecedents = tf.maximum(antecedent_offsets, 0)  # [k, k]
    target_doc_ids = tf.expand_dims(mention_doc_ids, 1)  # [k, k]
    antecedent_doc_ids = tf.gather(mention_doc_ids, antecedents)  # [k, k]
    antecedents_mask = tf.logical_and(tf.equal(target_doc_ids, antecedent_doc_ids), antecedents_mask) # [k,k]
    fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.expand_dims(top_span_mention_scores, 0) # [k, k]
    fast_antecedent_scores += tf.log(tf.to_float(antecedents_mask)) # [k, k] can not do masking at the end, need to sort
    fast_antecedent_scores += get_fast_antecedent_scores(top_span_emb, dropout) # [k, k]

    _, top_antecedents = tf.nn.top_k(fast_antecedent_scores, c, sorted=False) # [k, c]
    top_antecedents_mask = util.batch_gather(antecedents_mask, top_antecedents) # [k, c]
    top_antecedents_mask = tf.squeeze(top_antecedents_mask, -1)
    top_fast_antecedent_scores = util.batch_gather(fast_antecedent_scores, top_antecedents) # [k, c]
    top_fast_antecedent_scores = tf.squeeze(top_fast_antecedent_scores, -1)
    top_antecedent_offsets = util.batch_gather(antecedent_offsets, top_antecedents) # [k, c]
    top_antecedent_offsets = tf.squeeze(top_antecedent_offsets, -1)
    return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets


def distance_pruning(top_span_emb, top_span_mention_scores, c, mention_doc_ids):
    k = util.shape(top_span_emb, 0)
    top_antecedent_offsets = tf.tile(tf.expand_dims(tf.range(c) + 1, 0), [k, 1]) # [k, c]
    raw_top_antecedents = tf.expand_dims(tf.range(k), 1) - top_antecedent_offsets # [k, c]
    top_antecedents_mask = raw_top_antecedents >= 0 # [k, c]
    top_antecedents = tf.maximum(raw_top_antecedents, 0)  # [k, c]
    target_doc_ids = tf.expand_dims(mention_doc_ids, 1)  # [k, 1]
    antecedent_doc_ids = tf.gather(mention_doc_ids, top_antecedents)  # [k, c]
    top_antecedent_mask = tf.logical_and(tf.equal(target_doc_ids, antecedent_doc_ids), top_antecedents_mask)
    top_antecedents = tf.maximum(raw_top_antecedents, 0) # [k, c]

    top_fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.gather(top_span_mention_scores, top_antecedents) # [k, c]
    top_fast_antecedent_scores += tf.log(tf.to_float(top_antecedents_mask)) # [k, c]
    return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets
