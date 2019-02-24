# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:15:19 2019

@author: Dcm
"""
"""自然语言处理"""
import multiprocessing
import os
import numpy as np
from gensim.models.word2vec import LineSentence, Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class Processer(object):
    """
    cut_result:分词结果
    authors: 作者列表
    tfidf_word_vector: 用tf-idf为标准得到的词向量
    w2v_word_vector: 用word2vector得到的词向量
    w2v_model: 用word2vector得到的model
    """
    def __init__(self, cut_result, saved_dir):
        self.cut_result = cut_result
        self.authors = list(cut_result.author_poem_dict.keys())
        print('begin analyzing cut result...')
        self.cut_result = cut_result
        print("calculating poets' tf-idf word vector...")
        self.tfidf_word_vector = self._author_word_vector(cut_result.author_poem_dict)
        print("calculating poets' w2v word vector...")
        self.w2v_model, self.w2v_word_vector = self._word2vec(cut_result.author_poem_dict)
        print("result saved.")
        
    @staticmethod
    def _author_word_vector(author_poem_dict):
        """用tf-id计算每个作者的词向量"""
        poem = list(author_poem_dict.values())
        vectorizer = CountVectorizer(min_df=15)
        word_matrix = vectorizer.fit_transform(poem)
        transformer = TfidfTransformer()
        tfidf_word_vector = transformer.fit_transform(word_matrix).toarray()
        return tfidf_word_vector
    
    @staticmethod
    def _word2vec(author_poem_dict):
        """用word2vector计算每个作者的词向量"""
        """取每个词的向量求平均值"""
        dimension = 600
        authors = list(author_poem_dict.keys())
        poems = list(author_poem_dict.values())
        with open("cut_poem", 'w', encoding='utf-8') as f:
            f.write("\n".join(poems))
        model = Word2Vec(LineSentence("cut_poem"), size=dimension, min_count=15,
                         workers=multiprocessing.cpu_count())
        word_vector = []
        for i, author in enumerate(authors):
            vec = np.zeros(dimension)
            words = poems[i].split()
            count = 0
            for word in words:
                word = word.strip()
                try:
                    vec += model[word]
                    count += 1
                except KeyError:  # 有的词语不满足min_count则不会被记录在词表中
                    pass
            word_vector.append(np.array([v / count for v in vec]))
        os.remove("cut_poem")
        return model, word_vector
    
    def find_similar_author(self, author, use_w2v=False):
        """
        通过词向量寻找最相似的诗人
        :param: author: 需要寻找的诗人名称
        :return:最匹配的诗人
        """
        word_vector = self.tfidf_word_vector if not use_w2v else self.w2v_word_vector
        author_index = self.authors.index(author)
        x = word_vector[author_index]
        min_angle = np.pi
        min_index = 0
        for i, author in enumerate(self.authors):
            if i == author_index:
                continue
            y = word_vector[i]
            cos = x.dot(y) / (np.sqrt(x.dot(x)) * np.sqrt(y.dot(y)))
            angle = np.arccos(cos)  # 用向量间的夹角计算相似度
            if min_angle > angle:
                min_angle = angle
                min_index = i
        return self.authors[min_index]

    def find_similar_word(self, word):
        """通过w2v向量计算词条相似度"""
        return self.w2v_model.most_similar(word)
