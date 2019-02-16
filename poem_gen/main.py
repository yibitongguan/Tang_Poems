# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 21:52:57 2019

@author: Dcm
"""
"""唐诗生成器"""
from config import *
import data
import model

def defineArgs():
    """define args"""
    parser = argparse.ArgumentParser(description = "Chinese_poem_generator.")
    parser.add_argument("-m", "--mode", help = "select mode by 'train' or test or head",
                        choices = ["train", "test", "head"], default = "head")
    return parser.parse_args()

if __name__ == "__main__":
    tf.reset_default_graph()
    args = defineArgs()
    trainData = data.Poems(trainPoems)
    gen = model.Model(trainData)
    if args.mode == "train":
        gen.train()
    else:
        if args.mode == "test":
            poems = gen.genPoem()
        else:
            characters = input("please input chinese character:")
            poems = gen.genHeadPoem(characters)
