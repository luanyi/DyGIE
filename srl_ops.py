import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

srl_op_library = tf.load_op_library("./srl_kernels.so")

extract_spans = srl_op_library.extract_spans
tf.NotDifferentiable("ExtractSpans")

