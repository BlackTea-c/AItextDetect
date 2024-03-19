from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.normalizers import Lowercase

# 创建一个 BPE 模型的 Tokenizer
tokenizer = Tokenizer(models.BPE())

# 设置正规化器为转换为小写
tokenizer.normalizer = Lowercase()

# 设置预分词器为按空格分词
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# 创建一个训练文本数据
train_texts = ["This is a simple example.", "Another sentence for tokenization."]

# 使用训练器训练 Tokenizer
trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train_from_iterator(train_texts, trainer)

# 使用训练好的 Tokenizer 对文本进行分词
encoded_texts = tokenizer.encode_batch(train_texts)

# 打印分词结果
for encoded_text in encoded_texts:
    print(encoded_text.tokens)
