#!/usr/bin/env python

import os
import sys
import time
import json
import numpy as np

import tensorflow as tf

import coref_model as cm
import inference_utils
import input_utils
import srl_model as srl
import util

if __name__ == "__main__":
  util.set_gpus()

  name = sys.argv[1]
  output_filename = sys.argv[2]

  print "Running experiment: {}.".format(name)
  config = util.get_config("experiments.conf")[name]
  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

  util.print_config(config)
  #model = cm.CorefModel(config)
  model = srl.SRLModel(config)

  model.load_eval_data()

  saver = tf.train.Saver()
  log_dir = config["log_dir"]

  with tf.Session() as session:
    checkpoint_path = os.path.join(log_dir, "model.max.ckpt")
    saver.restore(session, checkpoint_path)

    with open(output_filename, "w") as f:
      #for example_num, (tensorized_example, example) in enumerate(model.eval_data):
      for i, doc_tensors in enumerate(model.eval_tensors):
        #feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
        feed_dict = dict(zip(
            model.input_tensors,
            [input_utils.pad_batch_tensors(doc_tensors, tn) for tn in model.input_names + model.label_names]))

        predict_names = []
        for tn in model.predict_names:
           if tn in model.predictions:
            predict_names.append(tn)

        predict_tensors = [model.predictions[tn] for tn in predict_names] + [model.loss]
        predict_tensors = session.run(predict_tensors, feed_dict=feed_dict)
        predict_dict = dict(zip(predict_names + ["loss"], predict_tensors))

        #_, _, _, mention_starts, mention_ends, antecedents, antecedent_scores, head_scores = session.run(model.predictions + [model.head_scores], feed_dict=feed_dict)
        doc_example = model.coref_eval_data[i]
        sentences = doc_example["sentences"]
        predictions = inference_utils.mtl_decode(sentences, predict_dict, model.srl_labels_inv, model.ner_labels_inv, config)

        #predicted_antecedents = model_utils.get_predicted_antecedents(antecedents, antecedent_scores)
        #example["predicted_clusters"], _ = model.get_predicted_clusters(mention_starts, mention_ends, predicted_antecedents)

        doc_example["predicted_clusters"] = []
        for cluster in predictions["predicted_clusters"]:
          doc_example["predicted_clusters"].append(tuple([(int(m[0]), int(m[1])) for m in cluster]))
          
        mention_starts = predict_dict["mention_starts"]
        mention_ends = predict_dict["mention_ends"]
        doc_example["top_spans"] = zip((int(i) for i in mention_starts), (int(i) for i in mention_ends))
        doc_example["head_scores"] = predict_dict["coref_head_scores"].tolist()

        # SRL and NER stuff. Maybe flatten to doc level.
        '''doc_example["predicted_srl"] = []
        doc_example["predicted_ner"] = []
        word_offset = 0
        for j, sentence in enumerate(sentences):
          for pred, args in predictions["srl"][j].iteritems():
            doc_example["predicted_srl"].extend([
                [int(pred + word_offset), int(a[0] + word_offset), int(a[1] + word_offset), a[2]] for a in args])
          for s in predictions["ner"][j]:
            doc_example["predicted_ner"].append([int(s[0] + word_offset), int(s[1] + word_offset), s[2]])
          word_offset += len(sentence)'''

        #print doc_example
        f.write(json.dumps(doc_example))
        f.write("\n")

        if (i + 1) % 10 == 0:
          print "Decoded {} examples.".format(i + 1)
          #break
