



# 数据解释
train_v2_drcat_02.csv  拓展的数据集 source表示生成的模型

至于另外一个：RDizzl3_seven feature
RDizzl3_seven feature — This is a boolean value feature present in the DAIGT-V2 dataset that determines whether essays 
were written based on prompts included in the hidden test set of the competition. 
We could either include all of the samples or filter only the True samples.
We have found that including only the True samples increases the score as we are training the model on data that is more
representative of the data present in the hidden test
这是DAIGT-V2数据集中的一个布尔值特征，用于确定作文是否是根据竞赛中隐藏的测试集中包含的提示撰写的。
我们可以选择包含所有样本，或者仅筛选出True样本。
我们发现仅包含True样本会提高分数，因为这样我们在训练模型时使用的数据更加代表竞赛中隐藏的测试集中存在的数据（我们通过这些数据进行评估）





# 大致步骤




使用了多个分类器来对TFIDF特征进行分类。
如下:
·多项式朴素贝叶斯 (MultinomialNB)
·补充朴素贝叶斯 (ComplementNB)，处理不平衡的文本分类问题
·线性支持向量机 (LinearSVC)
·随机梯度下降分类器 (SGDClassifier)
·LightGBM (LGBMClassifier) 这个是关键，之前用的CatBoost太慢了，找了半天找到了它；虽然损失了一些分数，但是大大加快了速度
（而且损失的分数貌似可以通过增加数据来弥补回来..）


这个我也没仔细考究，就是之前的LLM比赛里大家都在用各种组合，我就全拿下来一起用了。主要是效果确实比一两个好

参考了https://www.kaggle.com/code/batprem/llm-daigt-cv-0-9983-lb-0-960的代码 



利用了DAIGT-V2和外部训练数据:e_(我看还有很多,但是当时忙着忙着就忘记添加上去了...再回过来看发现结果都出了)。我分别对这两个数据集运行了相同的特征提取和分类算法，然后将它们以7:3的权重进行集成。
至于为什么这个权重主要是因为前者才是主要的数据集，根据两者数据含量简单做了一个对比，我也试过其它整数比但是最终效果就是7：3最好，至于0.几的调整我就试了一下发现变化不大便放弃这个优化点了。
(还有一个有趣的发现就是在尝试使用不同权重的侯，我发现权重越偏向外部数据集，私有得分就越高，但公开得分就越低...我觉得当时把其它额外的数据添加下来肯定就能拿铜甚至银了，可惜忘了这个事了...呜呜呜)11
1

集成之前对每个预测结果应用了最小-最大归一化，略微提高了一些分数.（参考的是这个的一个讨论内容:https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/468150）



