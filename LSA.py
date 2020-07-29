"""
@Author: jsq
@date: 2020/7/29
"""

# 对文章《一种基于潜在语义分析的问答系统答案定位方法》的实现

from collections import Counter
import numpy as np
from tqdm import tqdm
import jieba
import math


def load_stopwords(path):    # 加载停用词表
    print("load_stopwords_document")
    # stopwords = []
    with open(path, "r", encoding="utf-8") as f:
        stopwords = f.read().strip().split("\n")
    return stopwords


stopwords = load_stopwords("./hit_stopwords.txt")


def preprocess(content):                  # 去除停用词
    content_list = jieba.lcut(content)
    Content = []    # 去除停用词后的query | document 列表
    for eachword in content_list:
        if eachword not in stopwords:
            Content.append(eachword)
    return Content


def data_generator(filepath):
    document_list = []
    query_list = []
    pre_query_list = []
    pre_document_list = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f.read().split("\n")):
            # query, document = line.strip().split("\t")[:2]
            document = line.strip()
            # query_list.append(preprocess(query))
            document_list.append(preprocess(document))
            # pre_query_list.append(query)
            pre_document_list.append(document)
    # return document_list, query_list, pre_document_list, pre_query_list
    return document_list, pre_document_list


class LSA(object):
    def __init__(self, document_list):
        self.all_words = []
        self.document_list = document_list
        self.bag_of_tag()
        self.num_words = len(self.all_words)         # 包含的所有词数
        self.num_document = len(self.document_list)  # 文章数
        self.matrix = self.build_matrix()
        self.k = 150
        self.Uk, self.matrix_Dk = self.SVD()
        self.n = 5                                   # 前5可能的答案

    def bag_of_tag(self):
        all_words = []
        for i in self.document_list:
            all_words.extend(i)
        self.all_words = list(set(all_words))        # 统计document中出现的所有词构成词表

    def build_matrix(self):
        print("build_matrix_Aij")
        self.matrix = np.zeros([len(self.all_words), len(self.document_list)])
        for i in range(self.num_words):
            for j in range(self.num_document):
                frequency = Counter(self.document_list[j])  # 统计doc j的词频
                # 归一化分母
                normalization = math.sqrt(sum([x**2 for x in frequency.values()]))
                tfij = frequency[self.all_words[i]]   # 词i在doc j中出现的频次
                if not normalization:                 # 防止分母为0报错
                    self.matrix[i, j] = 0.0
                else:
                    self.matrix[i, j] = tfij / normalization   # 词频归一化
        return self.matrix

    def SVD(self):
        U, D, V = np.linalg.svd(self.matrix)   # 奇异值分解
        Uk = U[:, :self.k]  # 前K列
        Dk = D[:self.k]     # 取前K个奇异值(前K行 K列)  D只包含奇异值，需要重新构造成对角矩阵
        Vk = V[:self.k, :]  # 前k行
        matrix_Dk = np.zeros((self.k, self.k))
        # print(Dk)
        for i in range(self.k):
            matrix_Dk[i, i] = Dk[i]
        return Uk, matrix_Dk               # 将query和doc向量映射到新的空间需要用到 上述两个矩阵
        # self.Ak = Uk * matrix_Dk * Vk.T

    def shoot(self, query):
        result = []
        self.query = np.zeros((self.num_words, 1))
        for i in query:
            try:
                self.query[self.all_words.index(i)] += 1
            except:
                continue
        # print(np.linalg.inv(self.matrix_Dk))
        _q_ = np.dot(np.dot(self.query.T, self.Uk), np.linalg.inv(self.matrix_Dk))
        for document in range(self.num_document):
            d = np.zeros((self.num_words, 1))
            d[:, 0] = self.matrix[:, document]
            _d_ = np.dot(np.dot(d.T, self.Uk), np.linalg.inv(self.matrix_Dk))
            # print(_d_)
            # print(np.dot(_q_, _d_.T))
            multi = np.linalg.norm(_q_) * np.linalg.norm(_d_)
            if not multi:
                similarity = [[0]]
            else:
                similarity = np.dot(_q_, _d_.T) / multi
            # print(similarity)
            result.append(similarity[0][0])
        # return self.document_list[result.index(max(result))]
        # print(result)
        best_n = np.array(result).argsort()[::-1][:self.n]
        return best_n


if __name__ == "__main__":
    # document_list, documents = data_generator("./selfmake_document/document1.txt")
    # lsa = LSA(document_list)
    # while True:
    #     query = input("input:")
    #     if query == "exit":
    #         break
    #     ans = lsa.shoot(query)
    #     print(ans)
    #     print([documents[n] for n in ans])
    A = open("./result0729_3.txt", "a", encoding="utf-8")
    document_list, documents = data_generator("./selfmake_document/document1.txt")
    query_list, querys = data_generator("./selfmake_document/query.txt")
    lsa = LSA(document_list)
    for i in tqdm(range(len(query_list))):
        each_query = query_list[i]
        best_n = lsa.shoot(each_query)
        result = []
        for j in best_n:
            result.append(documents[j])
        A.write(querys[i]+ "\t" + "answer======>  " + "\t".join(result) + "\n")
        # if document_list[i] in result:
        #     A.write(querys[i] + "\t" + "answer======>  " + "True" + "\n")
        # else:
        #     A.write(querys[i] + "\t" + "answer======>  " + "False" + "\n")
    A.close()
