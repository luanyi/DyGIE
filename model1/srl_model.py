# Multi-predicate span-based SRL based on the e2e-coref model.

import math
import numpy as np
import operator
import os
import random
import tensorflow as tf

import util
import conll
from lsgn_data import LSGNData

from embedding_helper import get_embeddings
from input_utils import *
from model_utils import *


class SRLModel(object):
  def __init__(self, lsgn_data, config):
    self.config = config
    self.data = lsgn_data 
    
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
      self.predictions, self.loss = self.get_predictions_and_loss(
        self.data.input_dict, self.data.labels_dict, self.config)
      
    self.global_step = tf.Variable(0, name="global_step", trainable=False)
    self.reset_global_step = tf.assign(self.global_step, 0)
    learning_rate = tf.train.exponential_decay(
        self.config["learning_rate"], self.global_step, self.config["decay_frequency"],
        self.config["decay_rate"], staircase=True)
    trainable_params = tf.trainable_variables()
    gradients = tf.gradients(self.loss, trainable_params)
    gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
    optimizers = {
      "adam" : tf.train.AdamOptimizer,
      "sgd" : tf.train.GradientDescentOptimizer
    }
    optimizer = optimizers[self.config["optimizer"]](learning_rate)
    self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)

  def get_predictions_and_loss(self, inputs, labels, config):
    is_training = inputs["is_training"][0]
    self.dropout = 1 - (tf.to_float(is_training) * config["dropout_rate"])
    self.lexical_dropout = 1 - (tf.to_float(is_training) * config["lexical_dropout_rate"])
    self.lstm_dropout = 1 - (tf.to_float(is_training) * config["lstm_dropout_rate"])
  
    sentences = inputs["tokens"] 
    text_len = inputs["text_len"]  # [num_sentences]
    context_word_emb = inputs["context_word_emb"]  # [num_sentences, max_sentence_length, emb]
    head_word_emb = inputs["head_word_emb"]  # [num_sentences, max_sentence_length, emb]
    num_sentences = tf.shape(context_word_emb)[0]
    max_sentence_length = tf.shape(context_word_emb)[1]
    context_emb, head_emb, self.lm_weights, self.lm_scaling = get_embeddings(
        self.data, sentences, text_len, context_word_emb, head_word_emb, inputs["char_idx"],
        inputs["lm_emb"], self.lexical_dropout)  # [num_sentences, max_sentence_length, emb]
    
    context_outputs = lstm_contextualize(
        context_emb, text_len, config, self.lstm_dropout)  # [num_sentences, max_sentence_length, emb]

    # [num_sentences, max_num_candidates], ...
    candidate_starts, candidate_ends, candidate_mask = get_span_candidates(
        text_len, max_sentence_length, config["max_arg_width"])
    flat_candidate_mask = tf.reshape(candidate_mask, [-1])  # [num_sentences * max_num_candidates]
    batch_word_offset = tf.expand_dims(tf.cumsum(text_len, exclusive=True), 1)  # [num_sentences, 1]
    flat_candidate_starts = tf.boolean_mask(
        tf.reshape(candidate_starts + batch_word_offset, [-1]), flat_candidate_mask)  # [num_candidates]
    flat_candidate_ends = tf.boolean_mask(
        tf.reshape(candidate_ends + batch_word_offset, [-1]), flat_candidate_mask)  # [num_candidates]

    text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length)  # [num_sentences, max_sentence_length]
    flat_context_outputs = flatten_emb_by_sentence(context_outputs, text_len_mask)  # [num_doc_words]
    flat_head_emb = flatten_emb_by_sentence(head_emb, text_len_mask)  # [num_doc_words]
    doc_len = util.shape(flat_context_outputs, 0)

    candidate_span_emb, head_scores, span_head_emb, head_indices, head_indices_log_mask = get_span_emb(
        flat_head_emb, flat_context_outputs, flat_candidate_starts, flat_candidate_ends,
        config, self.dropout
    )  # [num_candidates, emb], [num_candidates, max_span_width, emb], [num_candidates, max_span_width]
    num_candidates = util.shape(candidate_span_emb, 0)
    max_num_candidates_per_sentence = util.shape(candidate_mask, 1)
    candidate_span_ids = tf.sparse_to_dense(
        sparse_indices=tf.where(tf.equal(candidate_mask, True)),
        output_shape=tf.cast(tf.stack([num_sentences, max_num_candidates_per_sentence]), tf.int64),
        sparse_values=tf.range(num_candidates, dtype=tf.int32),
        default_value=0,
        validate_indices=True)  # [num_sentences, max_num_candidates]

    predict_dict = {"candidate_starts": candidate_starts, "candidate_ends": candidate_ends}

    if config["coref_depth"]:
      candidate_mention_scores = get_unary_scores(
          candidate_span_emb, config, self.dropout, 1, "mention_scores")  # [num_candidates]
      #if self.config["span_score_weight"] > 0:
      #  candidate_mention_scores += self.config["span_score_weight"] * flat_span_scores

      doc_ids = tf.expand_dims(inputs["doc_id"], 1)  # [num_sentences, 1]
      candidate_doc_ids = tf.boolean_mask(
          tf.reshape(tf.tile(doc_ids, [1, max_num_candidates_per_sentence]), [-1]),
          flat_candidate_mask)  # [num_candidates]
 
      k = tf.to_int32(tf.floor(tf.to_float(doc_len) * config["mention_ratio"]))
      top_mention_indices = srl_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                                  tf.expand_dims(flat_candidate_starts, 0),
                                                  tf.expand_dims(flat_candidate_ends, 0),
                                                  tf.expand_dims(k, 0), doc_len,
                                                  True, True)  # [1, k]
      top_mention_indices.set_shape([1, None])
      top_mention_indices = tf.squeeze(top_mention_indices, 0)  # [k]
      mention_starts = tf.gather(flat_candidate_starts, top_mention_indices)  # [k]
      mention_ends = tf.gather(flat_candidate_ends, top_mention_indices)  #[k]
      mention_scores = tf.gather(candidate_mention_scores, top_mention_indices)  #[k]
      mention_emb = tf.gather(candidate_span_emb, top_mention_indices)  # [k, emb]
      mention_doc_ids = tf.gather(candidate_doc_ids, top_mention_indices)  # [k]



      max_mentions_per_doc = tf.reduce_max(
          #tf.segment_sum(data=tf.ones_like(mention_doc_ids, dtype=tf.int32),
          tf.unsorted_segment_sum(data=tf.ones_like(mention_doc_ids, dtype=tf.int32),
          segment_ids=mention_doc_ids,
          num_segments=tf.reduce_max(mention_doc_ids) + 1))  # []

      k_Print = tf.Print(k,
          [num_sentences, doc_len, k, max_mentions_per_doc],
          "Num sents, num tokens, num_mentions, max_antecedents")

      max_antecedents = tf.minimum(
          tf.minimum(config["max_antecedents"], k - 1), max_mentions_per_doc - 1)

      if self.config["coarse_to_fine"]:
        antecedents, antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = coarse_to_fine_pruning(mention_emb, mention_scores, max_antecedents, mention_doc_ids, self.dropout)
      else:
        antecedents, antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = distance_pruning(mention_emb, mention_scores, max_antecedents, mention_doc_ids)

      dummy_scores = tf.zeros([k, 1]) # [k, 1]
      
      for i in range(self.config["coref_depth"]):
        top_antecedent_emb = tf.gather(mention_emb, antecedents) # [k, c, emb]
        top_antecedent_scores, top_antecedent_emb, _ = get_antecedent_scores(mention_emb, mention_scores, antecedents, config, self.dropout, top_fast_antecedent_scores, top_antecedent_offsets)  # [k, max_ant]
        top_antecedent_weights = tf.nn.softmax(tf.concat([dummy_scores, top_antecedent_scores], 1)) # [k, c + 1]
        top_antecedent_emb = tf.concat([tf.expand_dims(mention_emb, 1), top_antecedent_emb], 1) # [k, c + 1, emb]
        attended_span_emb = tf.reduce_sum(tf.expand_dims(top_antecedent_weights, 2) * top_antecedent_emb, 1) # [k, emb]
        mention_emb = attended_span_emb
        with tf.variable_scope("f"):
          f = tf.sigmoid(util.projection(tf.concat([mention_emb, attended_span_emb], 1), util.shape(mention_emb, -1))) # [k, emb]
          mention_emb = f * attended_span_emb + (1 - f) * mention_emb # [k, emb]
      old_mention_emb = tf.gather(candidate_span_emb, top_mention_indices)
      top_mention_indices = tf.expand_dims(top_mention_indices, 1)
      old_mention_emb_padded = tf.scatter_nd(top_mention_indices, old_mention_emb, tf.shape(candidate_span_emb))
      new_mention_emb_padded = tf.scatter_nd(top_mention_indices, mention_emb, tf.shape(candidate_span_emb))
      candidate_span_emb = candidate_span_emb - old_mention_emb_padded + new_mention_emb_padded
      top_antecedent_scores = tf.concat([
          tf.zeros([k, 1]), top_antecedent_scores], 1)  # [k, max_ant+1]
      predict_dict.update({
          "candidate_mention_starts": flat_candidate_starts,  # [num_candidates]
          "candidate_mention_ends": flat_candidate_ends,  # [num_candidates]
          "candidate_mention_scores": candidate_mention_scores,  # [num_candidates]
          "mention_starts": mention_starts,  # [k]
          "mention_ends": mention_ends,  # [k]
          "antecedents": antecedents,  # [k, max_ant]
          "antecedent_scores": top_antecedent_scores,  # [k, max_ant+1]
      })

    spans_log_mask = tf.log(tf.to_float(candidate_mask))  # [num_sentences, max_num_candidates]
    if head_scores is not None:
      predict_dict["head_scores"] = head_scores

    dummy_scores = tf.expand_dims(tf.zeros_like(candidate_span_ids, dtype=tf.float32), 2)

    if config["ner_weight"] + config["coref_weight"] > 0:
      gold_ner_labels, gold_coref_cluster_ids = get_span_task_labels(
          candidate_starts, candidate_ends, labels, max_sentence_length)  # [num_sentences, max_num_candidates]
      

    if config["relation_weight"] > 0:
      if config['entity_beam']:
        flat_ner_scores = get_unary_scores(
          candidate_span_emb, config, self.dropout, len(self.data.ner_labels) - 1,
          "ner_scores")  # [num_candidates, num_labels-1]

        ner_scores = tf.gather(
          flat_ner_scores, candidate_span_ids
        ) + tf.expand_dims(spans_log_mask, 2)  # [num_sentences, max_num_candidates, num_labels-1]
        ner_scores = tf.concat([dummy_scores, ner_scores], 2)  # [num_sentences, max_num_candidates, num_labels]
        entity_starts, entity_ends, entity_scores, num_entities, top_entity_indices = get_ner_candidates(
          candidate_starts, candidate_ends, ner_scores, candidate_mask, text_len, config["entity_ratio"])  # Do we need to sort spans?
      else:
        flat_candidate_entity_scores = get_unary_scores(
          candidate_span_emb, config, self.dropout, 1, "entity_scores")  # [num_candidates,]
        candidate_entity_scores = tf.gather(
          flat_candidate_entity_scores, candidate_span_ids) + spans_log_mask  # [num_sentences, max_num_candidates] 
        entity_starts, entity_ends, entity_scores, num_entities, top_entity_indices = get_batch_topk(
          candidate_starts, candidate_ends, candidate_entity_scores, config["entity_ratio"], text_len,
          max_sentence_length, sort_spans=True, enforce_non_crossing=False)  # Do we need to sort spans?
      entity_span_indices = batch_gather(candidate_span_ids, top_entity_indices)  # [num_sentences, max_num_ents]
      entity_emb = tf.gather(candidate_span_emb, entity_span_indices)  # [num_sentences, max_num_ents, emb]
      max_num_entities = util.shape(entity_scores, 1)

    if config["relation_weight"] > 0:
      if config['add_ner_emb']:
        ner_emb = tf.gather(flat_ner_scores, entity_span_indices)
        entity_emb = tf.concat([entity_emb, ner_emb], 2)
      rel_labels = get_relation_labels(
          entity_starts, entity_ends, num_entities, labels, max_sentence_length
      )  # [num_sentences, max_num_ents, max_num_ents]
      if config['bilinear']:
        rel_scores = get_rel_bilinear_scores(
        entity_emb, entity_scores, len(self.data.rel_labels), config, self.dropout
      )  # [num_sentences, max_num_ents, max_num_ents, num_labels]
      else:
        if config['rel_prop']:
          for i in range(config['rel_prop']):
            rel_scores, entity_emb, flat_entities_mask = get_rel_scores(
              entity_emb, entity_scores, len(self.data.rel_labels), config, self.dropout, num_entities
            )  # [num_sentences, max_num_ents, max_num_ents, num_labels]
          if config['rel_prop_emb']:
            entity_emb_size = util.shape(entity_emb, -1)
            flat_entity_emb = tf.reshape(entity_emb, [num_sentences * max_num_entities, entity_emb_size])
            flat_entity_emb = tf.boolean_mask(flat_entity_emb, flat_entities_mask)
            entity_indices = tf.boolean_mask(tf.reshape(entity_span_indices, [-1]), flat_entities_mask)
            old_entity_emb = tf.gather(candidate_span_emb, entity_indices)
            entity_indices = tf.expand_dims(entity_indices, 1)
            old_entity_emb_padded = tf.scatter_nd(entity_indices, old_entity_emb, tf.shape(candidate_span_emb))
            new_entity_emb_padded = tf.scatter_nd(entity_indices, flat_entity_emb, tf.shape(candidate_span_emb))
            candidate_span_emb = candidate_span_emb - old_entity_emb_padded + new_entity_emb_padded
          
        else:
          rel_scores = get_rel_scores(
            entity_emb, entity_scores, len(self.data.rel_labels), config, self.dropout, num_entities
          )  # [num_sentences, max_num_ents, max_num_ents, num_labels]
          
    if config["relation_weight"] > 0:
      rel_loss = get_rel_softmax_loss(
          rel_scores, rel_labels, num_entities, config)  # [num_sentences, max_num_ents, max_num_ents]
      predict_dict.update({
        "entity_starts": entity_starts,
        "entity_ends": entity_ends,
        "entitiy_scores": entity_scores,
        "num_entities": num_entities,
        "rel_labels": tf.argmax(rel_scores, -1), # [num_sentences, num_ents, num_ents]
        "rel_scores": rel_scores
      })
    else:
      rel_loss = 0


    if config["ner_weight"] > 0:
      flat_ner_scores = get_unary_scores(
          candidate_span_emb, config, self.dropout, len(self.data.ner_labels) - 1,
          "ner_scores")  # [num_candidates, num_labels-1]

      ner_scores = tf.gather(
          flat_ner_scores, candidate_span_ids
      ) + tf.expand_dims(spans_log_mask, 2)  # [num_sentences, max_num_candidates, num_labels-1]
      ner_scores = tf.concat([dummy_scores, ner_scores], 2)  # [num_sentences, max_num_candidates, num_labels]
      
      
      ner_loss = get_softmax_loss(ner_scores, gold_ner_labels, candidate_mask)  # [num_sentences]
      ner_loss = tf.reduce_sum(ner_loss) # / tf.to_float(num_sentences)  # []
      predict_dict["ner_scores"] = ner_scores
    else:
      ner_loss = 0


    # Get coref representations.
    if config["coref_weight"] > 0:
      candidate_mention_scores = get_unary_scores(
          candidate_span_emb, config, self.dropout, 1, "mention_scores")  # [num_candidates]
      doc_ids = tf.expand_dims(inputs["doc_id"], 1)  # [num_sentences, 1]
      candidate_doc_ids = tf.boolean_mask(
          tf.reshape(tf.tile(doc_ids, [1, max_num_candidates_per_sentence]), [-1]),
          flat_candidate_mask)  # [num_candidates]
 
      k = tf.to_int32(tf.floor(tf.to_float(doc_len) * config["mention_ratio"]))
      top_mention_indices = srl_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                                  tf.expand_dims(flat_candidate_starts, 0),
                                                  tf.expand_dims(flat_candidate_ends, 0),
                                                  tf.expand_dims(k, 0), doc_len,
                                                  True, True)  # [1, k]
      top_mention_indices.set_shape([1, None])
      top_mention_indices = tf.squeeze(top_mention_indices, 0)  # [k]
      mention_starts = tf.gather(flat_candidate_starts, top_mention_indices)  # [k]
      mention_ends = tf.gather(flat_candidate_ends, top_mention_indices)  #[k]
      mention_scores = tf.gather(candidate_mention_scores, top_mention_indices)  #[k]
      mention_emb = tf.gather(candidate_span_emb, top_mention_indices)  # [k, emb]
      mention_doc_ids = tf.gather(candidate_doc_ids, top_mention_indices)  # [k]

      if head_scores is not None:
        predict_dict["coref_head_scores"] = head_scores

      max_mentions_per_doc = tf.reduce_max(
          #tf.segment_sum(data=tf.ones_like(mention_doc_ids, dtype=tf.int32),
          tf.unsorted_segment_sum(data=tf.ones_like(mention_doc_ids, dtype=tf.int32),
          segment_ids=mention_doc_ids,
          num_segments=tf.reduce_max(mention_doc_ids) + 1))  # []

      k_Print = tf.Print(k,
          [num_sentences, doc_len, k, max_mentions_per_doc],
          "Num sents, num tokens, num_mentions, max_antecedents")

      max_antecedents = tf.minimum(
          tf.minimum(config["max_antecedents"], k - 1), max_mentions_per_doc - 1)
      if self.config["coarse_to_fine"]:
        antecedents, antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = coarse_to_fine_pruning(mention_emb, mention_scores, max_antecedents, mention_doc_ids, self.dropout)
      else:
        antecedents, antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = distance_pruning(mention_emb, mention_scores, max_antecedents, mention_doc_ids)
              
      antecedent_log_mask = tf.log(tf.to_float(antecedents_mask))  # [k, max_ant]


      # [k, max_ant], [k, max_ant, emb], [k, max_ant, emb2]
      antecedent_scores, antecedent_emb, pair_emb = get_antecedent_scores(
          mention_emb, mention_scores, antecedents, config, self.dropout, top_fast_antecedent_scores, top_antecedent_offsets)  # [k, max_ant]
      antecedent_scores = tf.concat([
          tf.zeros([k, 1]), antecedent_scores], 1)  # [k, max_ant+1]
      



    # Compute Coref loss.
    if config["coref_weight"] > 0:
      flat_cluster_ids = tf.boolean_mask(
          tf.reshape(gold_coref_cluster_ids, [-1]), flat_candidate_mask)  # [num_candidates]
      mention_cluster_ids = tf.gather(flat_cluster_ids, top_mention_indices)  # [k]

      antecedent_cluster_ids = tf.gather(mention_cluster_ids, antecedents)  # [k, max_ant]
      antecedent_cluster_ids += tf.to_int32(antecedent_log_mask)  # [k, max_ant]
      same_cluster_indicator = tf.equal(
          antecedent_cluster_ids, tf.expand_dims(mention_cluster_ids, 1))  # [k, max_ant]
      non_dummy_indicator = tf.expand_dims(mention_cluster_ids > 0, 1)  # [k, 1]
      pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator)  # [k, max_ant]

      dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keep_dims=True))  # [k, 1]
      antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1)  # [k, max_ant+1]
      coref_loss = get_coref_softmax_loss(antecedent_scores, antecedent_labels)  # [k]
      coref_loss = tf.reduce_sum(coref_loss) # / tf.to_float(num_sentences)  # []
      predict_dict.update({
          "candidate_mention_starts": flat_candidate_starts,  # [num_candidates]
          "candidate_mention_ends": flat_candidate_ends,  # [num_candidates]
          "candidate_mention_scores": candidate_mention_scores,  # [num_candidates]
          "mention_starts": mention_starts,  # [k]
          "mention_ends": mention_ends,  # [k]
          "antecedents": antecedents,  # [k, max_ant]
          "antecedent_scores": antecedent_scores,  # [k, max_ant+1]
      })
    else:
      coref_loss = 0

    
    tf.summary.scalar("REL_loss", rel_loss)
    tf.summary.scalar("NER_loss", ner_loss)
    tf.summary.scalar("Coref_loss", coref_loss)
    #srl_loss_Print = tf.Print(srl_loss, [srl_loss, ner_loss, coref_loss], "Loss")
    loss = config["ner_weight"] * ner_loss + (config["coref_weight"] * coref_loss +
        config["relation_weight"] * rel_loss)

    return predict_dict, loss


       



