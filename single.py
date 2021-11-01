import math
import time
import re
import collections

## 文本预处理
# 读入语料文件
file_path = "./199801_clear.txt"
file = open(file_path, "r", encoding="gbk")
file_content = file.read()
file.close()

# 读入中文停用词表
stopwords_file_path = "./cn_stopwords.txt"
stopwords_file = open(stopwords_file_path, "r")
stopwords = stopwords_file.read()
stopwords= stopwords.split('\n')
stopwords_file.close()
# 切分得到不同的文章
essay = file_content.split("\n\n")
all_essay_words = []
all_essay_words_dict = []
for e in essay:
    essay_words = []
    lines = e.split("\n")

    for line in lines:
        # 去除英文和数字
        line = re.sub("[A-Za-z0-9]", '', line)
        # 去除所有的中英文标点
        line = re.sub("[!$%&()*+,-./:;<=>?@[\]^_`{|}~＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､　、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。]", '', line)
        # 用空格切分单词
        words = line.split(' ')
        # 删除所有空白字符
        words = list(filter(None, words))
        for word in words:
            # 删除停用词
            if word in stopwords:
                continue
            essay_words.append(word)
    all_essay_words.append(essay_words)
    essay_words_dict = collections.Counter(essay_words)
    all_essay_words_dict.append(essay_words_dict)

## 计算每篇文章单词的TF-IDF值
# tf
# 某个单词在文章中出现的次数/该文章的总词数
def tf(word, essay_words_dict):
    cnt = essay_words_dict[word]
    sum_of_words = sum(essay_words_dict.values())
    return cnt / sum_of_words

# idf
# idf = log(语料库中文章总数/包含该单词的文章数量 + 1)
def idf(word, all_essay_words_dict):
    sum = len(all_essay_words_dict)
    cnt = 0
    for essay_words_dict in all_essay_words_dict:
        if word in essay_words_dict:
            cnt = cnt + 1
    return math.log2(sum / (cnt + 1))

start  = time.time()

num_of_keywords_to_select = 10
keywords_list = []

# 计算所有单词的TF-IDF值
for essay_words_dict in all_essay_words_dict:
    tf_idf_dict = {}
    for word in essay_words_dict:
        # 单词的TF-IDF = tf*idf
        tf_idf_dict[word] = tf(word, essay_words_dict) * idf(word, all_essay_words_dict)
    # 对每篇文章的单词，根据TF-IDF值进行降序排序
    tf_idf_dict = sorted(tf_idf_dict.items(), key=lambda v:v[1], reverse=True)
    tf_idf_dict = dict(tf_idf_dict)
    # 选取TF-IDF值最大的10个单词作为该文章的关键词
    keywords_of_essay = list(tf_idf_dict)[:num_of_keywords_to_select]
    keywords_list.append(keywords_of_essay)

end = time.time()
print("计算TF_IDF指标用时： ", end - start)

## 计算每篇文章对应的向量
# 统计每篇文章在全部关键词列表上的词频，作为文章对应的向量
start = time.time()
# 将所有关键词存入一个一维列表
# 先存入集合中，去除重复的元素
all_keywords = set()
for keywords_of_essay in keywords_list:
    for word in keywords_of_essay:
        all_keywords.add(word)
all_keywords = list(all_keywords)

# 先将关键词和下标对应关系存储到一个词典中
index_dict = dict()
for keyword in all_keywords:
    index_dict[keyword] = all_keywords.index(keyword)

num_of_essays = len(all_essay_words)
# 初始化向量
vectors = [[0.0] * len(all_keywords) for i in range(num_of_essays)]
for i in range(num_of_essays):
    for word in all_essay_words[i]:
        if word in all_keywords:
            # 统计文章的单词在关键词列表上的词频
            vectors[i][index_dict[word]] += 1

end = time.time()
print("计算每个文章的向量用时： ", end - start)

## 计算两两文章之间的相似度
start = time.time()

# 采用余弦Cosine相似度计算公式
def cal_similarity(vec1, vec2):
    dot_product = 0
    len1 = 0
    len2 = 0
    for i in range(len(vec1)):
        # 计算向量数量积
        dot_product += vec1[i] * vec2[i]
        # 计算向量长度
        len1 += vec1[i] ** 2
        len2 += vec2[i] ** 2
    # 分母不能为0
    if len1 == 0 or len2 == 0:
        return 0
    len1 = len1 ** 0.5
    len2 = len2 ** 0.5
    res = round(dot_product / (len1 * len2), 5)
    return res

# 初始化相似度数组 规格为：文章数量*文章数量
similarity_res = [[0.0] * num_of_essays for i in range(num_of_essays)]
for i in range(num_of_essays):
    for j in range(num_of_essays):
        if i == j:
            similarity_res[i][j] = 1.0
        # 只需计算下三角矩阵
        elif j > i:
            continue
        else:
            similarity_res[i][j] = cal_similarity(vectors[i], vectors[j])

end = time.time()
print("计算两两文章之间相似度用时： ", end - start)
#print(similarityRes)

start = time.time()

# 为了更直观地观察结果，将相似度矩阵保存到txt文件中
res_file = open("resOfCalSimilarity.txt", 'w')
for i in range(len(similarity_res)):
    for j in range(len(similarity_res[i])):
        if j > i: # 只保存下三角部分
            break
        else:
            if similarity_res[i][j] == 0.0:
                res_file.write(str(0.0) + '     ')
            else:
                res_file.write(str(similarity_res[i][j]) + ' ')
    res_file.write('\n')
res_file.close()

end = time.time()
print("保存相似度矩阵用时： ", end - start)
