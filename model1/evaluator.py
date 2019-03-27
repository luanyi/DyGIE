#!/usr/bin/env python

import os
import re
import sys
sys.path.append(os.getcwd())
import time
import random
import shutil

import numpy as np
import tensorflow as tf
import copy
from lsgn_data import LSGNData
from lsgn_evaluator import LSGNEvaluator
from srl_model import SRLModel
import util


def copy_checkpoint(source, target):
  for ext in (".index", ".data-00000-of-00001"):
    shutil.copyfile(source + ext, target + ext)


if __name__ == "__main__":
  util.set_gpus()

  if len(sys.argv) > 1:
    name = sys.argv[1]
    print "Running experiment: {} (from command-line argument).".format(name)
  else:
    name = os.environ["EXP"]
    print "Running experiment: {} (from environment variable).".format(name)

  config = util.get_config("experiments.conf")[name]
  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

  # Dynamic batch size.
  config["batch_size"] = -1
  config["max_tokens_per_batch"] = -1
  coref_config = copy.deepcopy(config)
  coref_config['train_path'] = config['train_path_coref']
  coref_config['lm_path'] = config['lm_path_coref']
  coref_config['eval_path'] = config['eval_path_coref']
  coref_config['lm_path_dev'] = config['lm_path_dev_coref']
  coref_config['ner_weight'] = 0
  coref_config['relation_weight'] = 0
  coref_config['coref_weight'] = 1
  coref_config['main_metrics'] = 'coref'
  coref_config['coref_depth'] = 0
  
  # Use dev lm, if provided.
  if config["lm_path"] and "lm_path_dev" in config and config["lm_path_dev"]:
    config["lm_path"] = config["lm_path_dev"]
  if coref_config["lm_path"] and "lm_path_dev" in coref_config and coref_config["lm_path_dev"]:
    coref_config["lm_path"] = coref_config["lm_path_dev"]

  util.print_config(config)
  data = LSGNData(config)
  coref_data = LSGNData(coref_config)
  model = SRLModel(data, config)
  coref_model = SRLModel(coref_data, coref_config)
  evaluator = LSGNEvaluator(config)
  coref_evaluator = LSGNEvaluator(coref_config)
  variables_to_restore = []
  for var in tf.global_variables():
    if "module/" not in var.name:
      variables_to_restore.append(var)
    else:
      print "Not restoring from checkpoint:", var.name

  saver = tf.train.Saver(variables_to_restore)
  log_dir = config["log_dir"]
  assert not ("final" in name)  # Make sure we don't override a finalized checkpoint.

  writer = tf.summary.FileWriter(log_dir, flush_secs=20)
  evaluated_checkpoints = set()
  max_f1 = 0
  best_task_f1 = {}
  checkpoint_pattern = re.compile(".*model.ckpt-([0-9]*)\Z")

  with tf.Session() as session:
    while True:
      ckpt = tf.train.get_checkpoint_state(log_dir)
      if ckpt and ckpt.model_checkpoint_path and ckpt.model_checkpoint_path not in evaluated_checkpoints:
        print "Evaluating {}".format(ckpt.model_checkpoint_path)
        tf.global_variables_initializer().run()
        # Move it to a temporary location to avoid being deleted by the training supervisor.
        tmp_checkpoint_path = os.path.join(log_dir, "model.tmp.ckpt")
        copy_checkpoint(ckpt.model_checkpoint_path, tmp_checkpoint_path)

        global_step = int(checkpoint_pattern.match(ckpt.model_checkpoint_path).group(1))
        saver.restore(session, ckpt.model_checkpoint_path)

        print "Start evaluating ..."
        if config['coref_only']:
          eval_summary, f1, task_to_f1 = coref_evaluator.evaluate(
            session, coref_data, coref_model.predictions, coref_model.loss)
        else:
          if config['coref_freq'] and global_step % 8 == 1:
            print "Start Conll Eval..."
            coref_summary, coref_f1, _ = coref_evaluator.evaluate(
              session, coref_data, coref_model.predictions, coref_model.loss)
            print "End Conll Eval..."
          eval_summary, f1, task_to_f1 = evaluator.evaluate(
            session, data, model.predictions, model.loss)

        if f1 > max_f1:
          max_f1 = f1
          for task, f1 in task_to_f1.iteritems():
            best_task_f1[task] = f1

          copy_checkpoint(tmp_checkpoint_path, os.path.join(log_dir, "model.max.ckpt"))

        print "Current max combined F1: {:.2f}".format(max_f1)
        for task, f1 in best_task_f1.iteritems():
          print "Max {} F1: {:.2f}".format(task, f1) 
        
        writer.add_summary(eval_summary, global_step)
        print "Evaluation written to {} at step {}".format(log_dir, global_step)

        evaluated_checkpoints.add(ckpt.model_checkpoint_path)
      time.sleep(config["eval_sleep_secs"])
