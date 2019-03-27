#!/bin/bash


EMB_PATH="../embeddings"
if [ ! -d $EMB_PATH ]; then
  mkdir -p $EMB_PATH
fi

cd $EMB_PATH
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
wget https://dada.cs.washington.edu/qasrl/data/glove_50_300_2.zip
unzip glove_50_300_2.zip
rm glove_50_300_2.zip
wget http://ssli.ee.washington.edu/tial/projects/DyGIE/emb/glove.840B.300d.txt.filtered.genia
wget http://ssli.ee.washington.edu/tial/projects/DyGIE/emb/glove.840B.300d.txt.filtered.wlp
wget http://ssli.ee.washington.edu/tial/projects/DyGIE/emb/char_vocab_old.english.txt
cd $OLDPWD

DATA_PATH="../data/"
if [ ! -d $DATA_PATH ]; then
  mkdir -p $DATA_PATH
fi

#/g/tial/html/web-pages/tial/projects/DyGIE
# Get WLP package.
wget -O "${DATA_PATH}/wlp.tar.gz" http://ssli.ee.washington.edu/tial/projects/DyGIE/data/wlp.tar.gz
tar xf "${DATA_PATH}/wlp.tar.gz" -C "${DATA_PATH}"
rm "${DATA_PATH}/wlp.tar.gz"

#to get filtered wlp embeddings
# python scripts/filter_embeddings.py ${EMB_PATH}/glove.840B.300d.txt ${EMB_PATH}/glove.840B.300d.txt.filtered ${DATA_PATH}/wlp/json/train_3.json ${DATA_PATH}/processed_data/json/dev_3.json

# Get SciERC package.
wget -O "${DATA_PATH}/sciERC.tar.gz" http://ssli.ee.washington.edu/tial/projects/sciIE/data/sciERC_processed.tar.gz
tar xf "${DATA_PATH}/sciERC.tar.gz" -C "${DATA_PATH}"
rm "${DATA_PATH}/sciERC.tar.gz"


# Get GENIA package.
wget -O "${DATA_PATH}/genia.tar.gz" http://ssli.ee.washington.edu/tial/projects/DyGIE/data/genia.tar.gz
tar xf "${DATA_PATH}/genia.tar.gz" -C "${DATA_PATH}"
rm "${DATA_PATH}/genia.tar.gz"


python scripts/get_char_vocab.py 
