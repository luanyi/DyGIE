import datetime
import time


import coref_metrics
import debug_utils
import inference_utils
from input_utils import pad_batch_tensors
import operator
import srl_eval_utils
import util


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

      doc_size = len(doc_tensors)
      doc_example = self.coref_eval_data[i]
      sentences = doc_example["sentences"]
      decoded_predictions = inference_utils.mtl_decode(
          sentences, predict_dict, data.ner_labels_inv, data.rel_labels_inv,
          self.config)

      # Relation extraction.
      if "rel" in decoded_predictions:
        rel_predictions.extend(decoded_predictions["rel"])
        for j in range(len(sentences)):
          sent_example = self.eval_data[rel_sent_id][3]  # sentence, srl, ner, relations
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
        
        # Evaluate retrieval.
        doc_text_length = sum([len(s) for s in sentences])
        srl_eval_utils.evaluate_retrieval(
            predict_dict["candidate_mention_starts"], predict_dict["candidate_mention_ends"],
            predict_dict["candidate_mention_scores"], predict_dict["mention_starts"], predict_dict["mention_ends"],
            gold_mentions, doc_text_length, mention_evaluators)

      total_loss += predict_dict["loss"]
      if (i + 1) % 50 == 0:
        print ("Evaluated {}/{} documents.".format(i + 1, len(self.coref_eval_data)))

    debug_printer.close()
    summary_dict = {}
    task_to_f1 = {}  # From task name to F1.
    elapsed_time = time.time() - start_time

    sentences, gold_srl, gold_ner, gold_relations = zip(*self.eval_data)

    # Summarize results.
    if self.config["relation_weight"] > 0:
      precision, recall, f1 = (
          srl_eval_utils.compute_relation_f1(sentences, gold_relations, rel_predictions))
      task_to_f1["relations"] = f1
      summary_dict["Relation F1"] = f1
      summary_dict["Relation precision"] = precision
      summary_dict["Relation recall"] = recall
      for k, evaluator in sorted(entity_evaluators.items(), key=operator.itemgetter(0)):
        tags = ["{} {} @ {}".format("Entities", t, _k_to_tag(k)) for t in ("R", "P", "F")]
        results_to_print = []
        for t, v in zip(tags, evaluator.metrics()):
          results_to_print.append("{:<10}: {:.4f}".format(t, v))
          summary_dict[t] = v
        print ", ".join(results_to_print)
  

    if self.config["ner_weight"] > 0:
      ner_precision, ner_recall, ner_f1, ul_ner_prec, ul_ner_recall, ul_ner_f1, ner_label_mat = (
          srl_eval_utils.compute_span_f1(gold_ner, ner_predictions, "NER"))
      summary_dict["NER F1"] = ner_f1
      summary_dict["NER precision"] = ner_precision
      summary_dict["NER recall"] = ner_recall
      summary_dict["Unlabeled NER F1"] = ul_ner_f1
      summary_dict["Unlabeled NER precision"] = ul_ner_prec
      summary_dict["Unlabeled NER recall"] = ul_ner_recall

      # Write NER prediction to IOB format and run official eval script.
      srl_eval_utils.print_to_iob2(sentences, gold_ner, ner_predictions, self.config["ner_conll_eval_path"])
      task_to_f1["ner"] = ner_f1
      #for label_pair, freq in ner_label_mat.most_common():
      #  if label_pair[0] != label_pair[1] and freq > 10:
      #    print ("{}\t{}\t{}".format(label_pair[0], label_pair[1], freq))


    if self.config["coref_weight"] > 0:
      #conll_results = conll.evaluate_conll(self.config["conll_eval_path"], coref_predictions, official_stdout)
      #coref_conll_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
      #summary_dict["Average F1 (conll)"] = coref_conll_f1
      #print "Average F1 (conll): {:.2f}%".format(coref_conll_f1)

      p,r,f = coref_evaluator.get_prf()
      summary_dict["Average Coref F1 (py)"] = f
      print "Average F1 (py): {:.2f}%".format(f * 100)
      summary_dict["Average Coref precision (py)"] = p
      print "Average precision (py): {:.2f}%".format(p * 100)
      summary_dict["Average Coref recall (py)"] = r
      print "Average recall (py): {:.2f}%".format(r * 100)

      task_to_f1["coref"] = f * 100  # coref_conll_f1
      for k, evaluator in sorted(mention_evaluators.items(), key=operator.itemgetter(0)):
        tags = ["{} {} @ {}".format("Mentions", t, _k_to_tag(k)) for t in ("R", "P", "F")]
        results_to_print = []
        for t, v in zip(tags, evaluator.metrics()):
          results_to_print.append("{:<10}: {:.4f}".format(t, v))
          summary_dict[t] = v
        print ", ".join(results_to_print)

    summary_dict["Dev Loss"] = total_loss / len(self.coref_eval_data)

    print "Decoding took {}.".format(str(datetime.timedelta(seconds=int(elapsed_time))))
    print "Decoding speed: {}/document, or {}/sentence.".format(
        str(datetime.timedelta(seconds=int(elapsed_time / len(self.coref_eval_data)))),
        str(datetime.timedelta(seconds=int(elapsed_time / len(self.eval_data))))
    )

    metric_names = self.config["main_metrics"].split("_")
    main_metric = sum([task_to_f1[t] for t in metric_names]) / len(metric_names)
    print "Combined metric ({}): {}".format(self.config["main_metrics"], main_metric)

    return util.make_summary(summary_dict), main_metric, task_to_f1

