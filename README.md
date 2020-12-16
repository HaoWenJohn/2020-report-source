## 文件说明

tool-wear-gru-pred.ipynb   请使用jupyter notebook打开。内容包括实验数据集处理、模型构建和训练步骤。数据集请自行下载：<https://www.phmsociety.org/competition/phm/10>

train_log 请使用命令 tensorboard --logdir=train_log打开，内容是刀具磨损预测模型训练的loss图

Actor-Critic-Demo是使用tensorflow2搭建的深度强化学习demo。修改loop.py文件中的内容

env = gym.make("Acrobot-v1")可以更换游戏类型，可选值请参考https://gym.openai.com/envs/#classic_control

