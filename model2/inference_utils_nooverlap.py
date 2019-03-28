# Inference functions for the SRL model.
import numpy as np
import pdb

# TODO: Pass arg 
def decode_spans(span_starts, span_ends, span_scores, labels_inv):
  """
  Args:
    span_starts: [num_candidates,]
    span_scores: [num_candidates, num_labels]
  """
  pred_spans = []
  span_labels = np.argmax(span_scores, axis=1)  # [num_candidates]
  spans_list = zip(span_starts, span_ends, span_labels, span_scores)
  spans_list = sorted(spans_list, key=lambda x: x[3][x[2]], reverse=True)
  predicted_spans = {}
  for start, end, label, _ in spans_list:
    # Skip invalid span.
    if label == 0 or (start, end) in predicted_spans:
      continue
    pred_spans.append((start, end, labels_inv[label]))
    predicted_spans[(start, end)] = label
  return pred_spans
     

def greedy_decode(predict_dict, srl_labels_inv):
  """Greedy decoding for SRL predicate-argument structures.

  Args:
    predict_dict: Dictionary of name to numpy arrays.
    srl_labels_inv: SRL label id to string name.
    suppress_overlap: Whether to greedily suppress overlapping arguments for the same predicate.

  Returns:
    A dictionary from predicate ids to lists of argument spans.
  """
  arg_starts = predict_dict["arg_starts"]
  arg_ends = predict_dict["arg_ends"]
  predicates = predict_dict["predicates"]
  arg_labels = predict_dict["arg_labels"]
  scores = predict_dict["srl_scores"]

  num_suppressed_args = 0
 
  # Map from predicates to a list of labeled spans.
  pred_to_args = {}
  if len(arg_ends) > 0 and len(predicates) > 0:
    max_len = max(np.max(arg_ends), np.max(predicates)) + 1
  else:
    max_len = 1

  for j, pred_id in enumerate(predicates):
    args_list = []
    for i, (arg_start, arg_end) in enumerate(zip(arg_starts, arg_ends)):
      # If label is not null.
      if arg_labels[i][j] == 0:
        continue
      label = srl_labels_inv[arg_labels[i][j]]
      #if label not in ["V", "C-V"]:
      args_list.append((arg_start, arg_end, label, scores[i][j][arg_labels[i][j]]))

    # Sort arguments by highest score first.
    args_list = sorted(args_list, key=lambda x: x[3], reverse=True)
    new_args_list = []

    flags = [False for _ in range(max_len)]
    # Predicate will not overlap with arguments either. 
    flags[pred_id] = True

    for (arg_start, arg_end, label, score) in args_list:
      # If none of the tokens has been covered:
      if not max(flags[arg_start:arg_end+1]):
        new_args_list.append((arg_start, arg_end, label))
        for k in range(arg_start, arg_end+1):
          flags[k] = True

    # Only add predicate if it has any argument.
    if new_args_list:
      pred_to_args[pred_id] = new_args_list

    num_suppressed_args += len(args_list) - len(new_args_list)

  return pred_to_args, num_suppressed_args 
  

_CORE_ARGS = { "ARG0": 1, "ARG1": 2, "ARG2": 4, "ARG3": 8, "ARG4": 16, "ARG5": 32, "ARGA": 64,
               "A0": 1, "A1": 2, "A2": 4, "A3": 8, "A4": 16, "A5": 32, "AA": 64 }


def dp_decode(predict_dict, srl_labels_inv):
  """Decode arguments with dynamic programming. Enforce two constraints:
  1. Non-overlapping constraint.
  2. Unique core arg constraint.
  """
  arg_starts = predict_dict["arg_starts"]
  arg_ends = predict_dict["arg_ends"]
  predicates = predict_dict["predicates"]
  # Greedy labels.
  arg_labels = predict_dict["arg_labels"]  # [args,predicates]
  scores = predict_dict["srl_scores"]  # [args, predicates, roles] 

  pred_to_args = {}
  num_roles = scores.shape[2]

  for j, pred_id in enumerate(predicates):
    num_args = len(arg_starts)
    args = zip(arg_starts, arg_ends, range(num_args))
    args = sorted(args, key=lambda x: (x[0], x[1]))
    #print args

    text_len = max(max(arg_ends), pred_id) + 2
    f = np.log(np.zeros([text_len, 128], dtype=float))
    f[0,0] = .0
    states = { 0: set([0]) }  # A dictionary from id to list of binary core-arg states.
    pointers = {}  # A dictionary from states to (arg_id, role, prev_t, prev_rs) 
    best_state = [(0, 0)]

    def _update_state(t0, rs0, t1, rs1, delta, arg_id, role):
      if f[t0][rs0] + delta > f[t1][rs1]:
        f[t1][rs1] = f[t0][rs0] + delta
        if t1 not in states:
          states[t1] = set()
        states[t1].update([rs1])
        pointers[(t1, rs1)] = (arg_id, role, t0, rs0)
        if f[t1][rs1] > f[best_state[0][0]][best_state[0][1]]:
          best_state[0] = (t1, rs1)

    for start, end, i in args:
      assert scores[i][j][0] == 0
      # The extra dummy score should be same for all states, so we can safely skip arguments overlap
      # with the predicate.
      if start <= pred_id and pred_id <= end:
        continue
      # Locally best role assignment.
      r0 = arg_labels[i][j]
      # Strictly better to incorporate a dummy span if it has the highest local score.
      if r0 == 0:
        continue
      r0_str = srl_labels_inv[r0]
      # Enumerate explored states.
      t_states = [t for t in states.keys() if t <= start]
      for t in t_states:
        role_states = states[t]
        # Update states if best role is not a core arg.
        if not r0_str in _CORE_ARGS:
          for rs in role_states:
            _update_state(t, rs, end+1, rs, scores[i][j][r0], i, r0)
        else:
          core_state = _CORE_ARGS[r0_str]
          # Get highest-scored non-core arg.
          r1 = 0
          for r in range(1, num_roles):
            if scores[i][j][r] > scores[i][j][r1] and srl_labels_inv[r] not in _CORE_ARGS:
              r1 = r
          for rs in role_states:
            #print i, t, rs, core_state, r0, r0_str, r1
            if core_state & rs == 0:
              _update_state(t, rs, end+1, rs|core_state, scores[i][j][r0], i, r0)
            elif r1 > 0: 
              _update_state(t, rs, end+1, rs, scores[i][j][r0], i, r1)
    '''print f
    print states
    print pointers'''
    # Backtrack to decode.
    args_list = []
    t, rs = best_state[0]
    while (t, rs) in pointers:
      i, r, t0, rs0 = pointers[(t, rs)]
      args_list.append((arg_starts[i], arg_ends[i], srl_labels_inv[r]))
      t = t0
      rs = rs0
    if args_list:
      pred_to_args[pred_id] = args_list[::-1]
       
  return pred_to_args, 0


# Coref decoding.
def get_predicted_antecedents(antecedents, antecedent_scores):
  predicted_antecedents = []
  for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
    if index < 0:
      predicted_antecedents.append(-1)
    else:
      predicted_antecedents.append(antecedents[i, index])
  return predicted_antecedents


def get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents):
  mention_to_predicted = {}
  predicted_clusters = []
  for i, predicted_index in enumerate(predicted_antecedents):
    if predicted_index < 0:
      continue
    assert i > predicted_index
    predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
    if predicted_antecedent in mention_to_predicted:
      predicted_cluster = mention_to_predicted[predicted_antecedent]
    else:
      predicted_cluster = len(predicted_clusters)
      predicted_clusters.append([predicted_antecedent])
      mention_to_predicted[predicted_antecedent] = predicted_cluster

    mention = (int(top_span_starts[i]), int(top_span_ends[i]))
    predicted_clusters[predicted_cluster].append(mention)
    mention_to_predicted[mention] = predicted_cluster

  predicted_clusters = [tuple(pc) for pc in predicted_clusters]
  mention_to_predicted = { m:predicted_clusters[i] for m,i in mention_to_predicted.items() }

  return predicted_clusters, mention_to_predicted


def _decode_non_overlapping_spans(starts, ends, scores, max_len, labels_inv, pred_id):
  labels = np.argmax(scores, axis=1)
  spans = []
  for i, (start, end, label) in enumerate(zip(starts, ends, labels)):
    if label <= 0:
      continue
    label_str = labels_inv[label]
    if pred_id is not None and label_str == "V":
      continue
    spans.append((start, end, label_str, scores[i][label]))
  spans = sorted(spans, key=lambda x: x[3], reverse=True)
  flags = np.zeros([max_len], dtype=bool)
  if pred_id is not None:
    flags[pred_id] = True
  new_spans = []
  for start, end, label_str, score in spans:
    if not max(flags[start:end+1]):
      new_spans.append((start, end, label_str)) #, score))
      for k in range(start, end+1):
        flags[k] = True
  return new_spans


def _dp_decode_non_overlapping_spans(starts, ends, scores, max_len, labels_inv, pred_id,
                                     u_constraint=False):
  num_roles = scores.shape[1]
  labels = np.argmax(scores, axis=1)
  spans = zip(starts, ends, range(len(starts)))
  spans = sorted(spans, key=lambda x: (x[0], x[1]))

  if u_constraint:
    f = np.zeros([max_len + 1, 128], dtype=float) - 0.1
  else:
    f = np.zeros([max_len + 1, 1], dtype=float) - 0.1
  f[0, 0] = 0
  states = { 0: set([0]) }  # A dictionary from id to list of binary core-arg states.
  pointers = {}  # A dictionary from states to (arg_id, role, prev_t, prev_rs) 
  best_state = [(0, 0)]

  def _update_state(t0, rs0, t1, rs1, delta, arg_id, role):
    if f[t0][rs0] + delta > f[t1][rs1]:
      f[t1][rs1] = f[t0][rs0] + delta
      if t1 not in states:
        states[t1] = set()
      states[t1].update([rs1])
      pointers[(t1, rs1)] = (arg_id, role, t0, rs0)
      if f[t1][rs1] > f[best_state[0][0]][best_state[0][1]]:
        best_state[0] = (t1, rs1)
  new_spans = []
  for start, end, i in spans:
    assert scores[i][0] == 0
    # The extra dummy score should be same for all states, so we can safely skip arguments overlap
    # with the predicate.
    if pred_id is not None and start <= pred_id and pred_id <= end:
      continue
    r0 = labels[i]  # Locally best role assignment.
    # Strictly better to incorporate a dummy span if it has the highest local score.
    if r0 == 0:
      continue
    new_spans.append((starts[i], ends[i], labels_inv[r0]))
  return new_spans
  #   r0_str = labels_inv[r0]
  #   # Enumerate explored states.
  #   t_states = [t for t in states.keys() if t <= start]
  #   for t in t_states:
  #     role_states = states[t]
  #     # Update states if best role is not a core arg.
  #     if not u_constraint or not r0_str in _CORE_ARGS:
  #       for rs in role_states:
  #         _update_state(t, rs, end+1, rs, scores[i][r0], i, r0)
  #     else:
  #       for rs in role_states:
  #         for r in range(1, num_roles):
  #           if scores[i][r] > 0:
  #             r_str = labels_inv[r]
  #             core_state = _CORE_ARGS.get(r_str, 0)
  #             #print start, end, i, r_str, core_state, rs
  #             if core_state & rs == 0:            
  #               _update_state(t, rs, end+1, rs|core_state, scores[i][r], i, r)
  # # Backtrack to decode.
  # new_spans = []
  # t, rs = best_state[0]
  # while (t, rs) in pointers:
  #   i, r, t0, rs0 = pointers[(t, rs)]
  #   new_spans.append((starts[i], ends[i], labels_inv[r], scores[i,r]))
  #   t = t0
  #   rs = rs0

  #print spans
  #print new_spans[::-1]
  # return new_spans[::-1]
  

# One-stop decoder for all the tasks.
def mtl_decode(sentences, predict_dict, srl_labels_inv, ner_labels_inv, rel_labels_inv, config):
  predictions = {}
  
  # Decode sentence-level tasks.
  num_sentences = len(sentences)
  if "srl_scores" in predict_dict:
    predictions["srl"] = [{} for i in range(num_sentences)]
  if "ner_scores" in predict_dict:
    predictions["ner"] = [{} for i in range(num_sentences)]
  if "rel_scores" in predict_dict:
    predictions["rel"] = [[] for i in range(num_sentences)]
      
  # Sentence-level predictions.
  for i in range(num_sentences):
    if "srl" in predictions:
      num_args = predict_dict["num_args"][i]
      num_preds = predict_dict["num_preds"][i]
      for j, pred_id in enumerate(predict_dict["predicates"][i][:num_preds]):
        #arg_spans = _decode_non_overlapping_spans(
        arg_spans = _dp_decode_non_overlapping_spans(
            predict_dict["arg_starts"][i][:num_args],
            predict_dict["arg_ends"][i][:num_args],
            predict_dict["srl_scores"][i,:num_args,j,:],
            len(sentences[i]), srl_labels_inv, pred_id, config["enforce_srl_constraint"])
        # To avoid warnings in the eval script.
        if config["use_gold_predicates"]:
          arg_spans.append((pred_id, pred_id, "V"))
        if arg_spans:
          predictions["srl"][i][pred_id] = sorted(arg_spans, key=lambda x: (x[0], x[1]))
    if "rel" in predictions:
      num_ents = predict_dict["num_entities"][i]
      ent_starts = predict_dict["entity_starts"][i]
      ent_ends = predict_dict["entity_ends"][i]
      for j in range(num_ents):
        for k in range(num_ents):
          pred = predict_dict["rel_labels"][i,j,k]
          if pred > 0:
            predictions["rel"][i].append([
                ent_starts[j], ent_ends[j], ent_starts[k], ent_ends[k],
                rel_labels_inv[pred]])
    if "ner" in predictions:
      ner_spans = _dp_decode_non_overlapping_spans(
          predict_dict["candidate_starts"][i],
          predict_dict["candidate_ends"][i],
          predict_dict["ner_scores"][i],
          len(sentences[i]), ner_labels_inv, None, False)
      predictions["ner"][i] = ner_spans
  
  # Document-level predictions. -1 means null antecedent.
  if "antecedent_scores" in predict_dict:
    mention_spans = zip(predict_dict["mention_starts"], predict_dict["mention_ends"])
    mention_to_predicted = {}
    predicted_clusters = []

    def _link_mentions(curr_span, ant_span):
      if ant_span not in mention_to_predicted:
        new_cluster_id = len(predicted_clusters)
        mention_to_predicted[ant_span] = new_cluster_id
        predicted_clusters.append([ant_span,])
      cluster_id = mention_to_predicted[ant_span]
      if not curr_span in mention_to_predicted:
        mention_to_predicted[curr_span] = cluster_id
        predicted_clusters[cluster_id].append(curr_span)
      '''else:
        cluster_id2 = mention_to_predicted[curr_span]
        # Merge clusters.
        if cluster_id != cluster_id2:
          print "Merging clusters:", predicted_clusters[cluster_id], predicted_clusters[cluster_id2]
          for span in predicted_clusters[cluster_id2]:
            mention_to_predicted[span] = cluster_id
            predicted_clusters[cluster_id].append(span)
          predicted_clusters[cluster_id2] = []'''

    scores = predict_dict["antecedent_scores"]
    antecedents = predict_dict["antecedents"]
    #if config["coref_loss"] == "mention_rank":
    for i, ant_label in enumerate(np.argmax(scores, axis=1)):
      if ant_label <= 0:
        continue
      ant_id = antecedents[i][ant_label - 1]
      assert i > ant_id
      _link_mentions(mention_spans[i], mention_spans[ant_id])
    '''else:
      for i, curr_span in enumerate(mention_spans):
        for j in range(1, scores.shape[1]):
          if scores[i][j] > 0:
            _link_mentions(curr_span, mention_spans[antecedents[i][j-1]])'''

    predicted_clusters = [tuple(sorted(pc)) for pc in predicted_clusters]
    predictions["predicted_clusters"] = predicted_clusters
    predictions["mention_to_predicted"] = { m:predicted_clusters[i] for m,i in mention_to_predicted.items() }

  #print predictions["srl"]
  return predictions  

  
   
  
          
 

