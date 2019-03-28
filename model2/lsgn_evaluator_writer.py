import datetime
import time


import coref_metrics
import debug_utils
import inference_utils
from input_utils import pad_batch_tensors
import operator
import srl_eval_utils
import util
import json
import pdb
from JsonSerializer import MyEncoder

class LSGNEvaluator(object):
  def __init__(self, config):
    self.config = config
    self.eval_data = None

  # TODO: Split to multiple functions.
  def evaluate(self, session, data, predictions, loss, official_stdout=False):
    if self.eval_data is None:
      self.eval_data, self.eval_tensors, self.coref_eval_data = data.load_eval_data()

    def _k_to_tag(k):
      if k == -3:
        return "oracle"
      elif k == -2:
        return "actual"
      elif k == -1:
        return "exact"
      elif k == 0:
        return "threshold"
      else:
        return "{}%".format(k)

    # Retrieval evaluators.
    arg_evaluators = { k:util.RetrievalEvaluator() for k in [-3, -2, -1, 30, 40, 50, 80, 100, 120, 150] }
    predicate_evaluators = { k:util.RetrievalEvaluator() for k in [-3, -2, -1, 10, 20, 30, 40, 50, 70] }
    mention_evaluators = { k:util.RetrievalEvaluator() for k in [-3, -2, -1, 10, 20, 30, 40, 50] }
    entity_evaluators = { k:util.RetrievalEvaluator() for k in [-3, -2, -1, 10, 20, 30, 40, 50, 70] }

    total_loss = 0
    total_num_predicates = 0
    total_gold_predicates = 0

    srl_comp_sents = 0
    srl_predictions = []
    ner_predictions = []
    rel_predictions = []
    coref_predictions = {}
    coref_evaluator = coref_metrics.CorefEvaluator()
    all_gold_predicates = []
    all_guessed_predicates = []

    start_time = time.time()
    debug_printer = debug_utils.DebugPrinter()

    # Simple analysis.
    unique_core_role_violations = 0
    continuation_role_violations = 0
    reference_role_violations = 0
    gold_u_violations = 0
    gold_c_violations = 0
    gold_r_violations = 0
    json_data = []
    # Global sentence ID.
    rel_sent_id = 0
    srl_sent_id = 0

    for i, doc_tensors in enumerate(self.eval_tensors):
      feed_dict = dict(zip(
          data.input_tensors,
          [pad_batch_tensors(doc_tensors, tn) for tn in data.input_names + data.label_names]))
      predict_names = []
      for tn in data.predict_names:
        if tn in predictions:
          predict_names.append(tn)
      predict_tensors = [predictions[tn] for tn in predict_names] + [loss]
      predict_tensors = session.run(predict_tensors, feed_dict=feed_dict)
      predict_dict = dict(zip(predict_names + ["loss"], predict_tensors))
      doc_key = doc_tensors[0]['doc_key']
      json_output = {'doc_key':doc_key}
      doc_size = len(doc_tensors)
      doc_example = self.coref_eval_data[i]
      sentences = doc_example["sentences"]
      decoded_predictions = inference_utils.mtl_decode(
          sentences, predict_dict, data.ner_labels_inv, data.rel_labels_inv,
          self.config)

      # Relation extraction.
      if "rel" in decoded_predictions:
        rel_predictions.extend(decoded_predictions["rel"])
        json_output['relation'] = decoded_predictions["rel"]
        for j in range(len(sentences)):
          sent_example = self.eval_data[rel_sent_id][3]  # relations
          text_length = len(sentences[j])
          ne = predict_dict["num_entities"][j]
          gold_entities = set([])
          for rel in sent_example:
            gold_entities.update([rel[:2], rel[2:4]])
          # srl_eval_utils.evaluate_retrieval(
          #     predict_dict["candidate_starts"][j], predict_dict["candidate_ends"][j],
          #     predict_dict["candidate_entity_scores"][j], predict_dict["entity_starts"][j][:ne],
          #     predict_dict["entity_ends"][j][:ne], gold_entities, text_length, entity_evaluators)
          rel_sent_id += 1


      if "ner" in decoded_predictions:
        ner_predictions.extend(decoded_predictions["ner"])
        json_output['ner'] = decoded_predictions["ner"]

      if "predicted_clusters" in decoded_predictions:
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in doc_example["clusters"]]
        gold_mentions = set([])
        mention_to_gold = {}
        for gc in gold_clusters:
          for mention in gc:
            mention_to_gold[mention] = gc
            gold_mentions.add(mention)
        coref_evaluator.update(decoded_predictions["predicted_clusters"], gold_clusters, decoded_predictions["mention_to_predicted"],
                               mention_to_gold)
        coref_predictions[doc_example["doc_key"]] = decoded_predictions["predicted_clusters"]
        json_output['coref'] = decoded_predictions["predicted_clusters"]
        
        # Evaluate retrieval.
        doc_text_length = sum([len(s) for s in sentences])
        srl_eval_utils.evaluate_retrieval(
            predict_dict["candidate_mention_starts"], predict_dict["candidate_mention_ends"],
            predict_dict["candidate_mention_scores"], predict_dict["mention_starts"], predict_dict["mention_ends"],
            gold_mentions, doc_text_length, mention_evaluators)

      total_loss += predict_dict["loss"]
      if (i + 1) % 50 == 0:
        print ("Evaluated {}/{} documents.".format(i + 1, len(self.coref_eval_data)))
      json_data.append(json_output)
    debug_printer.close()
    outfn = self.config["output_path"]
    print 'writing to ' + outfn
    with open(outfn,'w') as f:
      for json_line in json_data:
        f.write(json.dumps(json_line, cls=MyEncoder))
        f.write('\n')
