#!/usr/bin/env python

import os
import re
import sys
sys.path.append(os.getcwd())
import time
import random

import numpy as np
import tensorflow as tf

from lsgn_data import LSGNData
from lsgn_evaluator_writer import LSGNEvaluator
from srl_model import SRLModel
import util

if __name__ == "__main__":
  #if "GPU" in os.environ:
  #  util.set_gpus(int(os.environ["GPU"]))
  #else:
  #  util.set_gpus()

  if len(sys.argv) > 1:
    name = sys.argv[1]
    print "Running experiment: {} (from command-line argument).".format(name)
  else:
    name = os.environ["EXP"]
    print "Running experiment: {} (from environment variable).".format(name)

  config = util.get_config("experiments.conf")[name]
  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

  config["batch_size"] = -1
  config["max_tokens_per_batch"] = -1

  # Use dev lm, if provided.
  if config["lm_path"] and "lm_path_dev" in config and config["lm_path_dev"]:
    config["lm_path"] = config["lm_path_dev"]

  util.print_config(config)
  data = LSGNData(config)
  model = SRLModel(data, config)
  evaluator = LSGNEvaluator(config)

  variables_to_restore = []
  for var in tf.global_variables():
    print var.name
    if "module/" not in var.name:
      variables_to_restore.append(var)

  saver = tf.train.Saver(variables_to_restore)
  log_dir = config["log_dir"]

  with tf.Session() as session:
    checkpoint_path = os.path.join(log_dir, "model.max.ckpt")
    print "Evaluating {}".format(checkpoint_path)
    tf.global_variables_initializer().run()
    saver.restore(session, checkpoint_path)
    evaluator.evaluate(session, data, model.predictions, model.loss)
