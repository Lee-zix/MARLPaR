#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/WN18RR/"
vocab_dir="datasets/data_preprocessed/WN18RR/vocab"
total_iterations=1000
path_length=3
hidden_size=50
embedding_size=50
batch_size=256
beta=0.05
Lambda=0.05
use_entity_embeddings=1
train_entity_embeddings=1
train_relation_embeddings=1
base_output_dir="output/WN18RR/"
load_model=1
#model_load_dir="/home/sdhuliawala/logs/RL-Path-RNN/wn18rrr/edb6_3_0.05_10_0.05/model/model.ckpt"
#model_load_dir="/home/jinxiaolong/lizixuan/python-project/MARLPaR/old/MINERVA-Agent-master/saved_models/WN18RR/model.ckpt"
model_load_dir="/home/jinxiaolong/lizixuan/python-project/MARLPaR/old/MINERVA-Agent-master/output/WN18RR/f389_3_0.05_100_0.05/model/model.ckpt"
nell_evaluation=0
