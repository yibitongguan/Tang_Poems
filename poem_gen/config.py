# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 08:37:08 2019

@author: Dcm
"""
"""训练相关参数配置"""
import tensorflow as tf
import numpy as np
import argparse
import os
import random
import time
import collections

batchSize = 64

learningRateBase = 0.001
learningRateDecayStep = 1000
learningRateDecayRate = 0.95

epochNum = 10                    # 迭代次数
generateNum = 1                   # 每次生成诗的数目

type = "poetrySong"                   # 数据集名称
trainPoems = "../dataset/" + type + "/" + type + ".txt" 
checkpointsPath = "./checkpoints/" + type 

saveStep = 1000                   # 每saveStep轮保存模型

trainRatio = 0.8                    # 训练比例
evaluateCheckpointsPath = "./checkpoints/evaluate"
