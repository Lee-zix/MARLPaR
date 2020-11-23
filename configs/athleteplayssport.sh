#!/usr/bin/env bash

data_input_dir="datasets/data_preprocessed/athleteplayssport/"
vocab_dir="datasets/data_preprocessed/athleteplayssport/vocab"
total_iterations=100
path_length=3
hidden_size=50
embedding_size=50
batch_size=128
beta=0.05
Lambda=0.05
use_entity_embeddings=1
train_entity_embeddings=1
train_relation_embeddings=1
base_output_dir="output/athleteplayssport/"
load_model=0
model_load_dir="/home/sdhuliawala/logs/RL-Path-RNN/wn18rrr/edb6_3_0.05_10_0.05/model/model.ckpt"
nell_evaluation=1