# DyGIE (Under Construction)

This repository contains code and models for replicating results from the following publication:
* [A General Framework for Information Extraction using Dynamic Span Graphs](https://arxiv.org/pdf/1808.09602.pdf)(NAACL, 2019)
* [Yi Luan](http://ssli.ee.washington.edu/~luanyi/), Dave Wadden, [Luheng He](https://homes.cs.washington.edu/~luheng), [Amy Shah](https://www.linkedin.com/in/amy-shah14), [Mari Ostendorf](https://ssli.ee.washington.edu/people/mo/), [Hannaneh Hajishirzi](https://homes.cs.washington.edu/~hannaneh/)
* In NAACL 2019

Part of the codebase is extended from [SciIE](https://bitbucket.org/luanyi/scierc/src/master/), [lsgn](https://github.com/luheng/lsgn) and [e2e-coref](https://github.com/kentonl/e2e-coref). 

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
which will download the GLoVe Embeddings, pre-processed dataset (in jason format) and cached ELMo embeddings. The link to SciERC, WLP and GENIA datasets are included. Processed ACE05 and ACE04 datasets will be provided if proved to have LDC License. 

* Build kernels: `./model1/build_custom_kernels.sh` (Please make adjustments to the script based on your OS/gcc version)

## Setting up for ELMo (if you use your own data)
* Some of our models are trained with the [ELMo embeddings](https://allennlp.org/elmo). We use the ELMo model loaded by [tensorflow_hub](https://www.tensorflow.org/hub/modules/google/elmo/1).
* It is recommended to cache ELMo embeddings for training and validating efficiency. Please modify the corresponding filenames and run
`python generate_elmo.py ` to generate ELMo embeddings for your own data.

## Two Variations of DyGIE
* model1: Coreference propagation layer at the bottom and relation propagation layer at the top (same as described in Figure 2 in the paper). This model architecture is used for all datasets all tasks except for ACE05 NER task.
* model2: Relation propagation layer at the bottom and coreference propagation layer at the top (swap CorefProp and RelProp in Figure 2). This model architecture is used for ACE05 NER task.

## Training Instructions

* Experiment configurations are found in `model*/experiments.conf`
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
* model1: Best models can be downloaded from here [WLP entity model](http://ssli.ee.washington.edu/tial/projects/DyGIE/models/wlp_best_ner.tar.gz),[WLP relation model](http://ssli.ee.washington.edu/tial/projects/DyGIE/models/wlp_best_relation.tar.gz),[GENIA entity model](http://ssli.ee.washington.edu/tial/projects/DyGIE/models/genia_best_ner.tar.gz), [SciERC entity model](http://ssli.ee.washington.edu/tial/projects/DyGIE/models/scierc_best_ner.tar.gz),[SciERC relation model](http://ssli.ee.washington.edu/tial/projects/DyGIE/models/scierc_best_relation.tar.gz), [ACE05 relation Model](http://ssli.ee.washington.edu/tial/projects/DyGIE/models/ace05_best_relation.tar.gz), [ACE05 nested NER](http://ssli.ee.washington.edu/tial/projects/DyGIE/models/ace2005_best_nested_ner.tar.gz), [ACE04 nested NER](http://ssli.ee.washington.edu/tial/projects/DyGIE/models/ace2004_best_nested_ner.tar.gz). Unzip and put the models under `./model1/logs` . For making predictings or testing results use the same command as the previous steps.
* model2: Best models can be downloaded from here [ACE05 entity model](http://ssli.ee.washington.edu/tial/projects/DyGIE/models/ace05_best_ner.tar.gz). Unzip and put the models under `./model2/logs`