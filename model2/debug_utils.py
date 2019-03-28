import codecs
import numpy as np
import os

_CORE_ARGS = { "ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5", "ARGA",
               "A0", "A1", "A2", "A3", "A4", "A5", "AA" }


def logsumexp(arr):
  maxv = np.max(arr)
  lognorm = maxv + np.log(np.sum(np.exp(arr - maxv)))
  arr2 = np.exp(arr - lognorm)
  #print maxv, lognorm, arr, arr2
  return arr2


def srl_constraint_tracker(pred_to_args):
  unique_core_role_violations = 0
  continuation_role_violations = 0
  reference_role_violations = 0
  for pred_ids, args in pred_to_args.iteritems():
    # Sort by span start, assuming they are not overlapping.
    sorted_args = sorted(args, key=lambda x: x[0], reverse=True)
    core_args = set()
    base_args = set()
    for start, end, role in sorted_args:
      if role in _CORE_ARGS:
        if role in core_args:
          unique_core_role_violations += 1
        core_args.update([role])
      elif role.startswith("C-") and not role[2:] in base_args:
        continuation_role_violations += 1
      if not role.startswith("C-") and not role.startswith("R-"):
        base_args.update(role)
    for start, end, role in sorted_args:
      if role.startswith("R-") and not role[2:] in base_args:
        reference_role_violations += 1
  return unique_core_role_violations, continuation_role_violations, reference_role_violations
    
  
def print_sentence_to_conll(fout, tokens, labels, head_scores, raw_head_scores=None):
  """token_info: Unnormalized head scores, etc.
  """
  for label_column in labels:
    assert len(label_column) == len(tokens)
  for i in range(len(tokens)):
    fout.write(tokens[i].ljust(10) + "\t")
    if raw_head_scores:
      for hs in raw_head_scores[i]:
        fout.write(str(round(hs, 3)).rjust(4) + "\t")
    for label_column, score_column in zip(labels, head_scores):
      fout.write(label_column[i].rjust(10) + "\t")
      if score_column[i] > 0:
        fout.write(str(round(score_column[i], 2)).rjust(4) + "\t")
      else:
        fout.write(" ".rjust(4) + "\t")
    fout.write("\n")
  fout.write("\n")


class DebugPrinter():
  def __init__(self):
    debug_filename = "/tmp/srl_debug_%d" % os.getpid()
    print ("Writing debugging info to: {}".format(debug_filename))
    self.fout = codecs.open(debug_filename, "w", "utf-8")

  def print_sentence(self, gold, pred_to_args, ner, constituency, head_scores, coref_head_scores=None):
    words, gold_srl, gold_ner = gold
    col_labels = [["*" for _ in words] for _ in range(len(pred_to_args))]
    span_head_scores = [[0.0 for _ in words] for _ in range(len(pred_to_args))]

    if coref_head_scores is not None:
      raw_head_scores = zip(head_scores, coref_head_scores)
    else:
      raw_head_scores = zip(head_scores)

    # Write predicted SRL.
    for i, pred_id in enumerate(sorted(pred_to_args.keys())):
      for start, end, label in pred_to_args[pred_id]:
        col_labels[i][start] = "(" + label + col_labels[i][start]
        col_labels[i][end] = col_labels[i][end] + ")"
        hs = logsumexp(head_scores[start:end+1])
        for j in range(start, end+1):
          span_head_scores[i][j] = hs[j - start]
      col_labels[i][pred_id] = "(V*)"

    # Write predicted NER.
    if ner:
      col_labels.append(["*" for _ in words]) 
      span_head_scores.append([0.0 for _ in words])
      for start, end, label in ner:
        col_labels[-1][start] = "(" + label + col_labels[-1][start]
        col_labels[-1][end] = col_labels[-1][end] + ")"
        hs = logsumexp(head_scores[start:end+1])
        for j in range(start, end+1):
         span_head_scores[-1][j] = hs[j - start]

    # Write predicted Const
    if constituency:
      col_labels.append(["*" for _ in words])
      span_head_scores.append([0.0 for _ in words])
      for start, end, label in constituency:
        col_labels[-1][start] = "(" + label + col_labels[-1][start]
        col_labels[-1][end] = col_labels[-1][end] + ")"
        hs = logsumexp(head_scores[start:end+1])
        for j in range(start, end+1):
         span_head_scores[-1][j] = hs[j - start]

    # Write gold SRL and NER.
    for _, pred_id in enumerate(sorted(gold_srl.keys())):
      col_labels.append(["*" for _ in words]) 
      span_head_scores.append([0.0 for _ in words])
      for start, end, label in gold_srl[pred_id]:
        col_labels[-1][start] = "(" + label + col_labels[-1][start]
        col_labels[-1][end] = col_labels[-1][end] + ")"
        hs = logsumexp(head_scores[start:end+1])
        for j in range(start, end+1):
          span_head_scores[-1][j] = hs[j - start]
      col_labels[-1][pred_id] = "(V*)"

    # Write predicted NER.
    if gold_ner:
      col_labels.append(["*" for _ in words]) 
      span_head_scores.append([0.0 for _ in words])
      for start, end, label in gold_ner:
        col_labels[-1][start] = "(" + label + col_labels[-1][start]
        col_labels[-1][end] = col_labels[-1][end] + ")"
        hs = logsumexp(head_scores[start:end+1])
        for j in range(start, end+1):
         span_head_scores[-1][j] = hs[j - start]

    print_sentence_to_conll(self.fout, words, col_labels, span_head_scores, raw_head_scores)

  def print_document(self, doc_example, sentence_examples, gold_ner,
                     srl_predictions, ner_predictions, coref_predictions,
                     mention_spans, antecedents, entity_gate, antecedent_attn):
    word_offset = 0
    mention_span_to_id = {}
    for i, span in enumerate(mention_spans):
      mention_span_to_id[span] = i
    
    doc_words = []

    #print len(sentence_examples), len(srl_predictions)
    for i, sent_example in enumerate(sentence_examples):
      words, gold_srl = sent_example
      doc_words.extend(words)

      self.fout.write(" ".join(words) + "\n")
      # Print SRL information.
      for pred, args in srl_predictions[i].iteritems():
        #print pred, args
        self.fout.write("{}:".format(words[pred]) + "\n")
        for arg in args:
          arg_tokens = " ".join(words[arg[0]:arg[1]+1])
          self.fout.write("\t" + arg_tokens + "\t" + arg[2])

          arg_span = (arg[0] + word_offset, arg[1] + word_offset)
          if arg_span in mention_span_to_id:
            mention_id = mention_span_to_id[arg_span]
            best_ant_id = np.argmax(antecedent_attn[mention_id])
            best_ant_span = mention_spans[antecedents[mention_id][best_ant_id]]
            try:
              self.fout.write("\t{}\t{}\t{}\n".format(
                  entity_gate[mention_id],
                  antecedent_attn[mention_id][best_ant_id],
                  " ".join(doc_words[best_ant_span[0]:best_ant_span[1]+1]))) 
            except UnicodeEncodeError:
              self.fout.write("\t{}\t{}\t{}\n".format(
                  entity_gate[mention_id], antecedent_attn[mention_id][best_ant_id], "???"))
          else:
            self.fout.write("\t-\n")

      self.fout.write("\n")
      word_offset += len(words)

  def print_sentence_and_beam(self, words, arg_starts, arg_ends, arg_scores,
                              predicates, pred_scores, srl_scores, pred_to_args):
    self.fout.write(" ".join(words) + "\n")
    args_to_preds = {}
    for pred, args in pred_to_args.iteritems():
      for start, end, label in args:
        arg = (start, end)
        if not arg in args_to_preds: args_to_preds[arg] = []
        args_to_preds[arg].append((words[pred], label))
    for start, end, score in zip(arg_starts, arg_ends, arg_scores):
      self.fout.write(
          " ".join(words[start:end+1]) + "\t" + str(score) + "\t" + str(args_to_preds.get((start, end), "-")) + "\n")
    self.fout.write("\n")
    for start, score in zip(predicates, pred_scores):
      self.fout.write(words[start] + "\t" + str(score) + "\n")
    self.fout.write("\n")
    
  def close(self):
    self.fout.close()
 
