# SciIE (Under Construction)

This repository contains code and models for replicating results from the following publication:
* [Multi-Task Identification of Entities, Relations, and Coreference for  Scientific Knowledge Graph Construction](https://arxiv.org/pdf/1808.09602.pdf)(EMNLP, 2018)
* [Yi Luan](http://ssli.ee.washington.edu/~luanyi/), [Luheng He](https://homes.cs.washington.edu/~luheng), [Mari Ostendorf](https://ssli.ee.washington.edu/people/mo/), [Hannaneh Hajishirzi](https://homes.cs.washington.edu/~hannaneh/)
* In EMNLP 2018

Part of the codebase is extended from [lsgn](https://github.com/luheng/lsgn) and [e2e-coref](https://github.com/kentonl/e2e-coref). 

### Requirements
* Python 2.7
  * TensorFlow 1.8.0
  * pyhocon (for parsing the configurations)
  * [tensorflow_hub](https://www.tensorflow.org/hub/) (for loading ELMo)

## Getting Started
* Python 2.7
* TensorFlow 1.8.0
* pyhocon (for parsing the configurations)
* tensorflow_hub (for ELMo)

* [GloVe](https://nlp.stanford.edu/projects/glove/) embeddings and downloading data:  
`./scripts/fetch_required_data.sh` 
* Build kernels: `./scripts/build_custom_kernels.sh` (Please make adjustments to the script based on your OS/gcc version)

## Setting up for ELMo (if you use your own data)
* Some of our models are trained with the [ELMo embeddings](https://allennlp.org/elmo). We use the ELMo model loaded by [tensorflow_hub](https://www.tensorflow.org/hub/modules/google/elmo/1).
* It is recommended to cache ELMo embeddings for training and validating efficiency. Please modify the corresponding filenames and run
`python generate_elmo.py ` to generate ELMo embeddings.


## Training Instructions

* Experiment configurations are found in `experiments.conf`
* The parameter `main_metrics` can be selected from coref, ner, relation or any combination of the three, such as coref_ner_relation which indicates the F1 score for averaged F1 score for coref, ner and relation. The model is tuned and saved based on the resulting averaged F1 score.
* The parameters `ner_weight`, `coref_weight` and `relation_weight` are weights for the multi-task objective. If set the weight to 0 then the task is not trained.
* Choose an experiment that you would like to run, e.g. `scientific_best_ner`

* For a single-machine experiment, run the following two commands in parallel:
	* `python singleton.py <experiment>`
	* `python evaluator.py <experiment>`

* Results are stored in the `logs` directory and can be viewed via TensorBoard.
* For final evaluation of the checkpoint with the maximum dev F1:
	* `python test_single.py <experiment>`

## Other Quirks

* It does not use GPUs by default. Instead, it looks for the `GPU` environment variable, which the code treats as shorthand for `CUDA_VISIBLE_DEVICES`.
* The evaluator should not be run on GPUs, since evaluating full documents does not fit within GPU memory constraints.
* The training runs indefinitely and needs to be terminated manually. 

## Making Predictions with Pretrained Models
* Define the output path in experiments.conf as output_path, the system will output the results of eval_path to output_path. The output file is also a json file, which has thesame format as eval_path. Then run
`python write_single.py <experiment>`

## Best Models ##
* Best models can be downloaded from here ([Best NER Model](http://nlp.cs.washington.edu/sciIE/models/scientific_best_ner.zip),[Best Coref Model](http://nlp.cs.washington.edu/sciIE/models/scientific_best_coref.zip),[Best Relation Model](http://nlp.cs.washington.edu/sciIE/models/scientific_best_relation.zip)), unzip and put the model under ./logs . For making predictings or testing results use the same command as the previous steps.