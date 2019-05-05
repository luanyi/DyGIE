# Requirements

* python3
* perl
* nltk (for stanford pos tagger)
* java (for stanford tools)
* zsh
* task datasets (see below)

# Links to tasks/data sets

* ACE 2004 (https://catalog.ldc.upenn.edu/LDC2005T09)
* ACE 2005 (https://catalog.ldc.upenn.edu/LDC2006T06)

# Usage
Code adapted from [LSTM-ER](https://github.com/tticoin/LSTM-ER/tree/master/data)

## download Stanford Core NLP & POS tagger

```
cd common
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-04-20.zip
wget http://nlp.stanford.edu/software/stanford-postagger-2015-04-20.zip
unzip stanford-corenlp-full-2015-04-20.zip
unzip stanford-postagger-2015-04-20.zip
cd ..
```

## copy and convert each corpus 

Please set the environment variables for the directories, or directly put the directories in the following commands beforehand.

### ACE 2004

```
cp -r ${ACE2004_DIR}/*/english ace2004/
cd ace2004
zsh run.zsh
mkdir -p ../../data/ace04/json/train
mkdir -p ../../data/ace04/json/test
mkdir -p ../../data/ace04/elmo/train
mkdir -p ../../data/ace04/elmo/test
python ace2json.py
python generate_elmo.py 0 # set gpu id to 0 or any other available gpu ids
cd ..
```

### ACE 2005

```
cp -r ${ACE2005_DIR}/*/English ace2005/
cd ace2005
zsh run.zsh
mkdir -p ../../data/ace05/json/
mkdir -p ../../data/ace05/elmo/
python ace2json.py
python generate_elmo.py 0 # set gpu id to 0 or any other available gpu ids
cd ..
```


### Ontonotes

```
cd ontonotes
./setup_training.sh # Adapted from e2e-coref code. Assumes access to OntoNotes 5.0. Please edit the ontonotes_path variable.
mkdir -p ../../data/conll/json
mkdir -p ../../data/conll/elmo
mv train.english.jsonlines ../../data/conll/json/train.english.json
mv dev.english.jsonlines ../../data/conll/json/dev.english.json
mv test.english.jsonlines ../../data/conll/json/test.english.json
python generate_elmo.py 0 # set gpu id to 0 or any other available gpu ids
cd ..