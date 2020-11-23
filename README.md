# MARLPaR

Code and models for the paper [Path Reasoning over Knowledge Graph: A Multi-Agent and Reinforcement Learning Based Method](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8637433)

We development the code based on the code of MINERVA [Go for a Walk and Arrive at the Answer - Reasoning over Paths in Knowledge Bases using Reinforcement Learning] (https://github.com/shehzaadzd/MINERVA)

MARLPaR is a multi-agent system which answers queries in a knowledge graph of entities and relations. Starting from an entity node, MARLPaR has
two agents to carry out relation selection and entity selection, respectively, in an iterative manner.


## Requirements
To install the various python dependences (including tensorflow)
```
pip install -r requirements.txt
```

## Training
The hyperparam configs for each experiments are in the **configs** directory. To start a particular experiment, just do
```
sh run.sh configs/${dataset}.sh
```
where the ${dataset}.sh is the name of the config file. For example, 
```
sh run.sh configs/WN18RR.sh
```

## Testing
make
```
load_model=1
model_load_dir="saved_models/WN18RR/model.ckpt"
```


## Citation
If you use this code, please cite our paper
```
@inproceedings{li2018path,
  title={Path reasoning over knowledge graph: A multi-agent and reinforcement learning based method},
  author={Li, Zixuan and Jin, Xiaolong and Guan, Saiping and Wang, Yuanzhuo and Cheng, Xueqi},
  booktitle={2018 IEEE International Conference on Data Mining Workshops (ICDMW)},
  pages={929--936},
  year={2018},
  organization={IEEE}
}
```
