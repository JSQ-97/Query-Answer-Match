"""
@Author: jsq
@date: 2020/7/23
"""
import jieba
import math
from collections import Counter
import numpy as np
from tqdm import tqdm
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model


def load_stopwords(path):    # 加载停用词表
    print("load_stopwords_document")
    # stopwords = []
    with open(path, "r", encoding="utf-8") as f:
        stopwords = f.read().strip().split("\n")
    return stopwords


class BM_25(object):
    def __init__(self, document_list, k1=1, k2=1, b=0.75):    # document_list --> 所有答案的集合
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.document_list = document_list
        self.document_number = len(document_list)
        self.avgdl = sum([len(i) for i in document_list]) / len(document_list)   # 语料库 平均文档长度(按词总数计算长度)
        self.idf = {}
        self.idf_value()

    def idf_value(self):
        allwords = []
        for i in self.document_list:
            allwords.extend(i)
        allwords = list(set(allwords))    # 统计已知问答列表中的所有词，分别计算其idf值
        for word in allwords:
            num = 0
            for document in self.document_list:
                if word in document:
                    num+=1
            idf_value = math.log10((self.document_number+0.5)/(num+0.5))
            self.idf[word] = idf_value

    def realation_qi_d(self, query):
        Score_list = []
        for i in range(len(self.document_list)):
            document = document_list[i]
            K = self.k1 * (1 - self.b + self.b * len(document) / self.avgdl)
            Score_q_d = 0
            frequence_d = Counter(document)
            frequence_q = Counter(query)
            for qi in list(set(query)):
                left = frequence_d.get(qi, 0) * (self.k1 + 1) / (frequence_d.get(qi, 0) + K)
                right = frequence_q.get(qi) * (self.k2 + 1) / (frequence_q.get(qi) + self.k2)
                R_qi_d = left * right
                Score_q_d += self.idf.get(qi, 0) * R_qi_d
            Score_list.append(Score_q_d)
        K = 5                                             # 设置K=5,取得分前5的答案
        best_K = np.array(Score_list).argsort()[::-1][:K]
        # best_Score = max(Score_list)
        # best_document = [document_list[i] for i in best_K]
        return best_K


def edit_distance(query_1, query_2):                   # 判断query间相似度   方法一：计算query问题 和 问题类别 的最小编辑距离
    m = len(query_1)
    n = len(query_2)
    dp = [[0 for i in range(n+1)] for j in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for i in range(n+1):
        dp[0][i] = i
    for i in range(1,m+1):
        for j in range(1,n+1):
            if query_1[i-1] == query_2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
    return dp[-1][-1]


config_path = "../bert/chinese_L-12_H-768_A-12/bert_config.json"
dict_path = "../bert/chinese_L-12_H-768_A-12/vocab.txt"
checkpoint_path = "../bert/chinese_L-12_H-768_A-12/bert_model.ckpt"
tokenizer = Tokenizer(dict_path, do_lower_case=True)
model = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path)


def cosine_distance(query_1, query_2):
    token_1, segment_1 = tokenizer.encode(query_1)
    token_2, segment_2 = tokenizer.encode(query_2)
    query_1_matrix = model.predict([np.array(token_1), np.array(segment_1)])
    query_2_matrix = model.predict([np.array(token_2), np.array(segment_2)])
    def Sum(array, query_matrix):          # query bert词向量求和
        for i in range(len(query_matrix[0])):
            if array is None:
                array = query_matrix[0][i]
            else:
                array += query_matrix[0][i]
        return array
    # query_1_sum = None
    # query_2_sum = None
    # query_1_sum = Sum(query_1_sum, query_1_matrix)
    # query_2_sum = Sum(query_2_sum, query_2_matrix)
    # neiji = np.linalg.norm(query_1_sum, query_2_sum)  # 向量内积
    # mo_mul = math.sqrt(np.linalg.norm(query_1_sum*query_1_sum))*math.sqrt(np.linalg.norm(query_2_sum*query_2_sum))
    # 向量的模的乘积
    cosine = neiji / mo_mul
    return cosine


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


if __name__ == "__main__":
    # document_list = [["北京","奥运会","将于","明日","召开"],["29届","奥运会","在","北京","举行"],["民法典","在","2021年","正式","施行"],["小王","去","小李家","吃饭","没给钱"]]
    # bm25_model = BM_25(document_list)
    # print(bm25_model.document_list)
    # print(bm25_model.idf)
    # print(bm25_model.realation_qi_d(["北京","奥运会","何时","召开"]))
    A = open("./result0729.txt", "a", encoding="utf-8")
    document_list, documents = data_generator("./selfmake_document/document1.txt")
    query_list, querys = data_generator("./selfmake_document/query.txt")
    # bm25_model = BM_25(document_list)
    # while True:
    #     query = input("input:")
    #     if query == "exit":
    #         break
    #     best_K = bm25_model.realation_qi_d(query)
    #     result = []
    #     for j in best_K:
    #         result.append(documents[j])
    #     print("answer======>  " + "\t".join(result))
    bm25_model = BM_25(document_list)
    for i in tqdm(range(len(query_list))):
        each_query = query_list[i]
        best_K = bm25_model.realation_qi_d(each_query)
        result = []
        for j in best_K:
            result.append(documents[j])
        A.write(querys[i]+ "\t" + "answer======>" + "\t" + "\t".join(result) + "\n")
        # if document_list[i] in result:
        #     A.write(querys[i] + "\t" + "answer======>  " + "True" + "\n")
        # else:
        #     A.write(querys[i] + "\t" + "answer======>  " + "False" + "\n")
    A.close()
