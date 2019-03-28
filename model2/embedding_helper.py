import tensorflow as tf

import util


def get_embeddings(data, sentences, text_len, context_word_emb, head_word_emb, char_index, lm_emb,
                   lexical_dropout):
  """Build word-level representations.
  Args:
    data: LSGNData object.
    sentences: string tokens. [batch_size, max_len]
    text_len: Length of each sentence. [batch_size]
    context_word_embeddings:
    head_word_embedding:
    char_index: Characters
    lm_emb: Cached contextualized embeddings.
    lexical_dropout: Tensor scalar
  """
  num_sentences = tf.shape(context_word_emb)[0]
  max_sentence_length = tf.shape(context_word_emb)[1]
  context_emb_list = [context_word_emb]
  head_emb_list = [head_word_emb]
  
  if data.char_embedding_size > 0:
    char_emb = tf.gather(
        tf.get_variable("char_embeddings", [len(data.char_dict), data.char_embedding_size]),
        char_index)  # [num_sentences, max_sentence_length, max_word_length, emb]
    flattened_char_emb = tf.reshape(
        char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2),
        util.shape(char_emb, 3)])  # [num_sentences * max_sentence_length, max_word_length, emb]
    flattened_aggregated_char_emb = util.cnn(
        flattened_char_emb, data.config["filter_widths"],
        data.config["filter_size"])  # [num_sentences * max_sentence_length, emb]
    aggregated_char_emb = tf.reshape(
        flattened_aggregated_char_emb, [num_sentences, max_sentence_length,
        util.shape(flattened_aggregated_char_emb, 1)]) # [num_sentences, max_sentence_length, emb]
    context_emb_list.append(aggregated_char_emb)
    head_emb_list.append(aggregated_char_emb)

  if data.lm_file or data.lm_hub:
    # Alternatively, we could initialize module/aggregation/* from here. 
    with tf.variable_scope("lm_aggregation"):
      lm_weights = tf.nn.softmax(tf.get_variable("lm_scores", [data.lm_layers],
                                 initializer=tf.constant_initializer(0.0)))
      lm_scaling = tf.get_variable("lm_scaling", [], initializer=tf.constant_initializer(1.0))

    # Load lm_embeddings from hub.
    if data.lm_hub:
      lm_embeddings = data.lm_hub(
        inputs={"tokens": sentences, "sequence_len": text_len},
        signature="tokens", as_dict=True)
      word_emb = tf.expand_dims(lm_embeddings["word_emb"], 3)  # [B, slen, 512]
      lm_emb = tf.concat([
          tf.concat([word_emb, word_emb], 2),  # [B, slen, 1024, 1]
          tf.expand_dims(lm_embeddings["lstm_outputs1"], 3),
          tf.expand_dims(lm_embeddings["lstm_outputs2"], 3)], 3)  # [B, slen, 1024, 3]

    lm_emb_size = util.shape(lm_emb, 2)  # TODO: Might not need this.
    lm_num_layers = util.shape(lm_emb, 3)
    flattened_lm_emb = tf.reshape(
        lm_emb, [num_sentences * max_sentence_length * lm_emb_size, lm_num_layers]
    )  # [num_sentences * max_sentence_length * emb, layers]
    flattened_aggregated_lm_emb = tf.matmul(
        flattened_lm_emb, tf.expand_dims(lm_weights, 1)
    )  # [num_sentences * max_sentence_length * emb, 1]
    aggregated_lm_emb = tf.reshape(
        flattened_aggregated_lm_emb, [num_sentences, max_sentence_length, lm_emb_size])
    aggregated_lm_emb *= lm_scaling
    context_emb_list.append(aggregated_lm_emb)
  else:
    lm_weights = None
    lm_scaling = None

  # Concatenate and apply dropout.
  context_emb = tf.concat(context_emb_list, 2)  # [num_sentences, max_sentence_length, emb]
  head_emb = tf.concat(head_emb_list, 2)  # [num_sentences, max_sentence_length, emb]
  context_emb = tf.nn.dropout(context_emb, lexical_dropout)
  head_emb = tf.nn.dropout(head_emb, lexical_dropout)
  return context_emb, head_emb, lm_weights, lm_scaling

