# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 14:14:40 2019

@author: Dcm
"""
"""唐诗文本挖掘"""
import os
import pickle
from collections import Counter, OrderedDict
from jieba import posseg as pseg

class CutResult(object):
    """
    分词结果
    char_counter：字频统计
    author_counter：作者计数
    word_set：词汇表
    word_counter：词频统计
    word_property_counter_dict：词汇词性
    author_poem_dict：解析后的结果，作者与他对应的诗
    """   
    def __init__(self):
        self.char_counter = Counter()
        self.author_counter = Counter()
        self.word_set = set()
        self.word_counter = Counter()
        self.word_property_counter_dict = {}
        self.author_poem_dict = OrderedDict()
        
    def add_cut_poem(self, author, word):
        """为author_poem_dict添加对象"""
        ctp = self.author_poem_dict.get(author)
        if ctp is None:
            self.author_poem_dict[author] = ""
        self.author_poem_dict[author] += ' '.join(word)
     
def wordCut(text):
    # 采取逐字切分
    tex = [i.strip('（|）|{|}|“|”|□|.|\|:|?|，|。|\n|\r|《|》') for i in text]
    tex = list(filter(None, tex))  # 去除空字符
    return tex
        
def poemCut(text):
    """分词函数"""
    # 虚词停用词库
    stopwords = '而|何|乎|乃|其|且|若|所|为|焉|以|因|与|也|则|者|之|的|兮|不|自' \
                +'|得|一|来|去|可|是|已|此|上|中|三|无|有|上|下'
    tex = [i.strip('(|)|{|}|“|”|□|.|\|:|?|，|。|\n|\r|') \
            for i in wordCut(text) if i not in stopwords]
    tex = list(filter(None, tex))  # 去除空字符
    return tex
        
def cut_poem(filename, saved_dir):
    """
    分词
    :param: filename: 全唐诗输入文件位置
            saved_location: 结果存储位置
    :return:分词结果
    """
    target_file_path = os.path.join(saved_dir, 'cut_result.pkl')
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)
    if os.path.exists(target_file_path):
        print('load existed cut result.')
        with open(target_file_path, 'rb') as f:
            result = pickle.load(f)
    else:
        result = CutResult()
        line_count = 0
        # 语料库
        with open(filename, "r", encoding='utf-8') as f:  
            for line in f: 
                line_count += 1
                if line_count % 5000 == 0:
                    print('%d lines processed.' % line_count)
                #if line_count > 1000:
                    #break
                try:
                    # 得到题目 作者 内容
                    title, author, poem = line.strip().split("::")
                    # 作者
                    if author != '不详' or '无名氏':  # 去除作者不详的诗
                        result.author_counter[author] += 1
                    # 诗句
                    chars = poemCut(poem)
                    for char in chars:
                        result.char_counter[char] += 1
                    cut_line = pseg.cut(poem)
                    for word, property in cut_line:
                        if result.word_property_counter_dict.get(property) is None:
                            result.word_property_counter_dict[property] = Counter()
                        if len(word) > 1:
                            result.word_property_counter_dict[property][word] += 1
                            result.word_set.add(word)
                            result.word_counter[word] += 1
                            result.add_cut_poem(author, word)
                except Exception as e:
                    print("%d-解析文件异常 %s" % (line_count, line))
                    raise e
        with open(target_file_path, 'wb') as f:
            pickle.dump(result, f)
    
    return result

if __name__ == '__main__':
    text = "云横秦岭家何在，雪拥蓝关马不前。"
    print(wordCut(text))
    print(poemCut(text))
