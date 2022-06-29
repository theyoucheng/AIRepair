#!/bin/bash

mkdir -p sst_data
mkdir -p toxic_data
mkdir -p imdb_data

echo "process sst"
##############prepare the sst 
#'''
#''' https://nlp.stanford.edu/sentiment/
#'''

data=trainDevTestTrees_PTB.zip
curl    --output sst_data/$data -O https://nlp.stanford.edu/sentiment/$data
unzip -o  sst_data/$data -d sst_data/
rm -f sst_data/$data


echo "process toxic"
##############prepare the toxic 
#'''
#''' https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
#'''
data=jigsaw-toxic-comment-classification-challenge.zip

mkdir -p toxic_data/data/
curl    --output toxic_data/data/$data -O "https://raw.githubusercontent.com/alm0st907/ursotoxic/master/jigsaw-toxic-comment-classification-challenge.zip"
unzip -o  toxic_data/data/$data -d toxic_data/data/
#unzip -o  toxic_data/test.csv.zip -d toxic_data/
#unzip -o toxic_data/train.csv.zip -d toxic_data/
rm -fr toxic_data/data/$data


echo "process imdb"
##############prepare the imdb 
#'''
#''' https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
#'''

#data=aclImdb_v1.tar.gz
#curl    --output imdb_data/$data -O https://ai.stanford.edu/~amaas/data/sentiment/$data
#tar -xf  imdb_data/$data -C imdb_data/
#rm -f imdb_data/$data



