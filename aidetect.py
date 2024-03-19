import sys
import gc


#基于Tfidf

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)   #分词处理

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier  #分类器结果整合


test = pd.read_csv('test_essays.csv')
sub = pd.read_csv('sample_submission.csv')
train = pd.read_csv("train_v2_drcat_02.csv", sep=',')



e_ = pd.read_csv(r'daigt_external_dataset.csv')
d_0 = e_[['text']]
d_0['label'] = 0
d_1 = e_[['source_text']]
d_1['label'] = 1
d_1.rename(columns={'source_text':'text'}, inplace=True)
e_1 = pd.concat([d_0, d_1], ignore_index=True)
data_set = [e_1,]
external_train = pd.concat([e_1,], ignore_index=True) #额外训练




train = train[['text','label']]
print(train)
train = train.drop_duplicates(subset=['text'])
train.reset_index(drop=True, inplace=True)


#训练一下分词器
LOWERCASE = False  #不转换为小写
VOCAB_SIZE = 30522  #默认即可

raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])
dataset = Dataset.from_pandas(test[['text']])
print('dataset:',dataset)

def train_corp_iter():
    for i in range(0, len(dataset), 1000):
        yield dataset[i: i + 1000]["text"]  #训练分词器的时候逐步提供数据而不是一次性


raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer) #训练分词器
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)
tokenized_texts_test = []

for text in tqdm(test['text'].tolist()):
    tokenized_texts_test.append(tokenizer.tokenize(text))


tokenized_texts_train = []

for text in tqdm(train['text'].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))


tokenized_texts_external = []
for text in tqdm(external_train['text'].tolist()):
    tokenized_texts_external.append(tokenizer.tokenize(text))
#print('分词:')
#print(tokenized_texts_test)
#print(tokenized_texts_external)


def func(tokenized_texts_train, tokenized_texts_test):
    '''第一个 TF-IDF 向量化器的目的是为了获取测试数据集中出现的所有词汇，并建立词汇表。这样做是为了确保训练数据和测试数据使用了相同的词汇表，
    从而避免了在测试阶段出现未知词汇的情况。在实际应用中，测试数据通常是未知的，我们不能将测试数据提前放入训练阶段，
    因此需要单独拟合一个 TF-IDF 向量化器来获取测试数据的词汇表。
而第二个 TF-IDF 向量化器则直接使用第一个向量化器获取的词汇表，以确保训练数据和测试数据使用相同的词汇表进行特征表示，
从而保持一致性。
总的来说，创建两个 TF-IDF 向量化器的目的是为了确保训练数据和测试数据在特征表示时使用了相同的词汇表，从而避免了词汇表不一致带来的问题。'''
    def dummy(text):
        return text
    # n-gram 特征的范围设置为3-5，意思就是采取连续的3-5词
    vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer = 'word',
                                tokenizer = dummy,
                                preprocessor = dummy,
                                token_pattern = None, strip_accents='unicode',
                            )

    vectorizer.fit(tokenized_texts_test)
    vocab = vectorizer.vocabulary_  #词汇表
    vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
                                analyzer = 'word',
                                tokenizer = dummy,
                                preprocessor = dummy,
                                token_pattern = None, strip_accents='unicode'
                                )

    tf_train = vectorizer.fit_transform(tokenized_texts_train)
    tf_test = vectorizer.transform(tokenized_texts_test)

    del vectorizer #删除释放内存.
    gc.collect()
    return tf_train, tf_test


y_train = train['label'].values


def get_model():
    #from catboost import CatBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.naive_bayes import ComplementNB
    #基本分类器
    clf = MultinomialNB(alpha=0.02)
    cnb = ComplementNB(alpha=0.02)
    lr_svc = CalibratedClassifierCV(LinearSVC(max_iter=8000, loss='hinge', tol=1e-4))
    sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber")
    para = {
        'n_iter': 1500,
        'verbose': -1,
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05073909898961407,
        'colsample_bytree': 0.726023996436955,
        'colsample_bynode': 0.5803681307354022,
        'lambda_l1': 8.562963348932286,
        'lambda_l2': 4.893256185259296,
        'min_data_in_leaf': 115,
        'max_depth': 23,
        'max_bin': 898
    }#配置lgb的参数
    lgb = LGBMClassifier(**para)

    weights = [0.05, 0.05, 0.25, 0.25, 0.4]

    ensemble = VotingClassifier(estimators=[('mnb', clf),
                                            ('cnb', cnb),
                                            ('sgd', sgd_model),
                                            ('lr_svc', lr_svc),
                                            ('lgb', lgb),
                                            ],
                                weights=weights, voting='soft', n_jobs=-1)
    return ensemble



model = get_model() #分类器

tf_train, tf_test = func(tokenized_texts_train, tokenized_texts_test) #
print('tf_train:',tf_train)
model.fit(tf_train, y_train) #TDFIFD提取的特征进行训练
gc.collect()
final_preds = model.predict_proba(tf_test)[:,1]


y_train = external_train['label'].values
tf_train, tf_test = func(tokenized_texts_external, tokenized_texts_test)
model = get_model()
model.fit(tf_train, y_train)
gc.collect()
final_preds_2 = model.predict_proba(tf_test)[:,1]

print(final_preds)
print(final_preds_2)








final_preds = ((final_preds - final_preds.min()) / (final_preds.max() - final_preds.min()))
final_preds_2 = ((final_preds_2 - final_preds_2.min()) / (final_preds_2.max() - final_preds_2.min()))
sub['generated'] = 0.7 * final_preds + 0.3 * final_preds_2  #主体是v2_drcat  额外训练数据external_data
sub.to_csv('submission.csv', index=False)