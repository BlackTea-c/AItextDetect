import sys
import gc

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier


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
external_train = pd.concat([e_1,], ignore_index=True)




train = train[['text','label']]
train = train.drop_duplicates(subset=['text'])
train.reset_index(drop=True, inplace=True)


LOWERCASE = False
VOCAB_SIZE = 30522

raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
dataset = Dataset.from_pandas(test[['text']])


def train_corp_iter():
    for i in range(0, len(dataset), 1000):
        yield dataset[i: i + 1000]["text"]


raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)
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



def func(tokenized_texts_train, tokenized_texts_test):
    def dummy(text):
        return text
    vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer = 'word',
                                tokenizer = dummy,
                                preprocessor = dummy,
                                token_pattern = None, strip_accents='unicode',
                            )

    vectorizer.fit(tokenized_texts_test)
    vocab = vectorizer.vocabulary_
    vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
                                analyzer = 'word',
                                tokenizer = dummy,
                                preprocessor = dummy,
                                token_pattern = None, strip_accents='unicode'
                                )

    tf_train = vectorizer.fit_transform(tokenized_texts_train)
    tf_test = vectorizer.transform(tokenized_texts_test)

    del vectorizer
    gc.collect()
    return tf_train, tf_test


y_train = train['label'].values


def get_model():
    from catboost import CatBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.naive_bayes import ComplementNB
    clf = MultinomialNB(alpha=0.02)
    cnb = ComplementNB(alpha=0.02)
    lr_svc = CalibratedClassifierCV(LinearSVC(max_iter=8000, loss='hinge', tol=1e-4))
    sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber")
    p6 = {
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
    }
    lgb = LGBMClassifier(**p6)

    weights = [0.05, 0.05, 0.25, 0.25, 0.4]

    ensemble = VotingClassifier(estimators=[('mnb', clf),
                                            ('cnb', cnb),
                                            ('sgd', sgd_model),
                                            ('lr_svc', lr_svc),
                                            ('lgb', lgb),
                                            ],
                                weights=weights, voting='soft', n_jobs=-1)
    return ensemble



model = get_model()
tf_train, tf_test = func(tokenized_texts_train, tokenized_texts_test)
model.fit(tf_train, y_train)
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
sub['generated'] = 0.6 * final_preds + 0.4 * final_preds_2
sub.to_csv('submission.csv', index=False)