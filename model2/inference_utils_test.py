import numpy as np
from inference_utils import dp_decode

def dp_decode_test():
  srl_labels_inv = ["", "ARG0", "ARG1", "AM-TMP"]
  arg_starts = np.array([0, 0, 1, 3])
  arg_ends = np.array([0, 2, 2, 4])
  predicates = np.array([5])
  arg_labels = np.array([[1], [1], [2], [2]])
  scores = np.array([[[ 0.,  0.4,  0.,   0.]],
                     [[ 0.,  1.,   0.,   0.]],
                     [[ 0.,  0.,   0.9,  0.]],
                     [[ 0.,  0.,   0.5,  0.1]]])
  predict_dict = {
      "arg_starts": arg_starts,
      "arg_ends": arg_ends,
      "predicates": predicates,
      "arg_labels": arg_labels,
      "srl_scores": scores
  }

  pred_to_args, _ = dp_decode(predict_dict, srl_labels_inv)
  print pred_to_args


if __name__ == "__main__":
  dp_decode_test()


