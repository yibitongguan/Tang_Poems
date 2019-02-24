# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 08:37:37 2019

@author: Dcm
"""
import wordcloud
import matplotlib.pyplot as plt
from data_cut import cut_poem
from nlp_process import Processer
from scipy.misc import imread

type = "poetryTang"      # 数据集名称
Poems_file = "../dataset/" + type + "/" + type + ".txt" # 数据集路径
save_dir = './save'
image_path = './image/'
myfront = r'C:\Windows\Fonts\STXINGKA.TTF'

def print_counter(counter):
    """打印数目"""
    for key, value in counter:
        print(key, value)
    print()
    
# 测试文本挖掘
result = cut_poem(Poems_file, save_dir)
process = Processer(result, save_dir)

print("**基于统计的分析")
print("写作数量排名：")
print_counter(result.author_counter.most_common(10))

print("最常用的词：")
print_counter(result.word_counter.most_common(10))

print("最常用的名词：")
print_counter(result.word_property_counter_dict['n'].most_common(10))

print("最常见的地名：")
print_counter(result.word_property_counter_dict['ns'].most_common(10))

print("最常见的形容词：")
print_counter(result.word_property_counter_dict['a'].most_common(10))

print("字频统计词云展示：")
print_counter(result.char_counter.most_common(10))
cloudobj = wordcloud.WordCloud(font_path=myfront,
                               mask = imread(image_path + 'cloud.png'),
                               mode = 'RGBA', background_color=None
                               ).fit_words(result.char_counter)
plt.imshow(cloudobj)
plt.axis('off')
plt.show()
cloudobj.to_file(image_path + '唐诗.png')


print("**基于词向量的分析")
for word in ["春", "花", "海", "酒"]:
    print("与 %s 相关的词：" % word)
    print_counter(process.find_similar_word(word))
 
for author in ["李白", "杜甫", "白居易"]:
    print("与 %s 用词相近的诗人：" % author)
    print("根据tf-idf计算： %s" % process.find_similar_author(author))
    print("根据word2vector计算： %s\n" % process.find_similar_author(author, use_w2v=True))

