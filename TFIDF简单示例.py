from sklearn.feature_extraction.text import TfidfVectorizer

# 假设有三个文档
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
]

# 创建一个TfidfVectorizer对象
vectorizer = TfidfVectorizer()

# 对文档进行拟合，得到TF-IDF特征矩阵
tfidf_matrix = vectorizer.fit_transform(documents)

# 输出词汇表
print("Vocabulary:")
print(vectorizer.get_feature_names_out())

# 输出TF-IDF特征矩阵
print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())
