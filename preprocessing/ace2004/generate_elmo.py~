import tensorflow as tf
import tensorflow_hub as hub
import h5py
import numpy as np
import json
import sys
print tf.__version__
from util import set_gpus
def Elmo(fn, outfn):
      with open(fn) as f:
            dev_examples = [json.loads(jsonline) for jsonline in f.readlines()]

      sents = [example["sentences"] for example in dev_examples]
      docids = [example["doc_key"] for example in dev_examples]
      

      config = tf.ConfigProto()
      with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            with h5py.File(outfn) as fout:
                  #for line in fin:
                  for i in range(len(sents)):
                        if i % 100 == 0:
                              print 'Finished ' + str(i)
                        doc = sents[i]
                        docid = docids[i]
                        for j in range(len(doc)):
                              sent = [doc[j]]
                              slen = [len(doc[j])]
                              lm_emb = sess.run(
                                    lm_emb_op, feed_dict={
                                          sentences: sent,
                                          text_len: slen
                                    }
                              )
                              sentence_id = docid + '_' + str(j)
                              print sentence_id
                              ds = fout.create_dataset(
                                    sentence_id, lm_emb.shape[1:], dtype='float32',
                                    data=lm_emb[0, :, :, :]  # [slen, lm_size, lm_layers]
                              )
                  fout.close  


#### Model #####

set_gpus(sys.argv[1]) # set the gpu id
elmo = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
sentences = tf.placeholder('string', shape=(None, None))
text_len = tf.placeholder('int32', shape=(None))

lm_embeddings = elmo(
    inputs={
        "tokens": sentences,
        "sequence_len": text_len
    },
        signature="tokens", as_dict=True)

word_emb = tf.expand_dims(lm_embeddings["word_emb"], 3)  # [B, slen, 512]
lm_emb_op = tf.concat([
        tf.concat([word_emb, word_emb], 2),  # [B, slen, 1024, 1]
        tf.expand_dims(lm_embeddings["lstm_outputs1"], 3),
        tf.expand_dims(lm_embeddings["lstm_outputs2"], 3)], 3)  # [B, slen, 1024, 3]
fn = '../../data/ace05/json/test.json'
outfn = '../../data/ace05/elmo/test.hdf5'
Elmo(fn, outfn)
fn = '../../data/ace05/json/dev.json'
outfn = '../../data/ace05/elmo/dev.hdf5'
Elmo(fn, outfn)
fn = '../../data/ace05/json/train.json'
outfn = '../../data/ace05/elmo/train.hdf5'
Elmo(fn, outfn)
          

