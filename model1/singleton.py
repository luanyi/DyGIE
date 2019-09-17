#!/usr/bin/env python

import os
import sys
sys.path.append(os.getcwd())
import json
import time
import random

import numpy as np
import tensorflow as tf
import pdb
from lsgn_data import LSGNData
from srl_model import SRLModel
import util
import copy

if __name__ == "__main__":
  print 'start'
  if len(sys.argv) > 1:
    name = sys.argv[1]
  else:
    name = os.environ["EXP"]
  config = util.get_config("experiments.conf")[name]
  print 'config'
  report_frequency = config["report_frequency"]

  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))
  util.print_config(config)
  print os.environ
  # if "GPU" in os.environ:
  #   gpus = [int(g) for g in os.environ["GPU"].split(",")]
  #   util.set_gpus(*gpus)
  # else:
  util.set_gpus(config['gpu'])

  data = LSGNData(config)
  coref_config = copy.deepcopy(config)
  coref_config['train_path'] = config['train_path_coref']
  coref_config['lm_path'] = config['lm_path_coref']
  coref_config['eval_path'] = config['eval_path_coref']
  coref_config['lm_path_dev'] = config['lm_path_dev_coref']
  coref_config['ner_weight'] = 0
  coref_config['coref_weight'] = 1
  coref_config['relation_weight'] = 0
  # coref_config['batch_size'] = 30
  coref_config['coref_depth'] = 0
  model = SRLModel(data, config)
  if config['coref_freq']:
    coref_data = LSGNData(coref_config)
    coref_model = SRLModel(coref_data, coref_config)
  saver = tf.train.Saver()
  init_op = tf.global_variables_initializer()

  log_dir = config["log_dir"]
  assert not ("final" in name)  # Make sure we don't override a finalized checkpoint.
  writer = tf.summary.FileWriter(log_dir, flush_secs=20)

  # Create a "supervisor", which oversees the training process.
  if not config['coref_only']:
    sv = tf.train.Supervisor(logdir=log_dir,
                             init_op=init_op,
                             saver=saver,
                             global_step=model.global_step,
                             save_model_secs=120)
  else:
    sv = tf.train.Supervisor(logdir=log_dir,
                             init_op=init_op,
                             saver=saver,
                             global_step=coref_model.global_step,
                             save_model_secs=120)
    

  # The supervisor takes care of session initialization, restoring from
  # a checkpoint, and closing when done or an error occurs.
  with sv.managed_session() as session:
    data.start_enqueue_thread(session)
    if config['coref_freq']:
      coref_data.start_enqueue_thread(session)
    accumulated_loss = 0.0
    initial_time = time.time()
    while not sv.should_stop():
      if config['coref_only']:
        loss, tf_global_step, _ = session.run([coref_model.loss, coref_model.global_step, coref_model.train_op])
        accumulated_loss += loss
        
      else:
        tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op])
        accumulated_loss += tf_loss
        if config['coref_freq'] and tf_global_step % config['coref_freq'] == 0: 
          coref_loss, coref_global_step, _ = session.run([coref_model.loss, coref_model.global_step, coref_model.train_op])
          accumulated_loss += coref_loss
      if tf_global_step % report_frequency == 0:
        total_time = time.time() - initial_time
        steps_per_second = tf_global_step / total_time

        average_loss = accumulated_loss / report_frequency
        print "[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second)
        writer.add_summary(util.make_summary({"loss": average_loss}), tf_global_step)
        accumulated_loss = 0.0

  # Ask for all the services to stop.
  sv.stop()
