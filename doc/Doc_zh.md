# WENLING (文灵) 情感分析包文档

WENLING (文灵) 是一个借助 Keras 构建的模型包，提供数种常用的用作情感分类（三分）的模型：LSTM，GRU，BiLSTM，BiGRU，CNN+LSTM。通过简单地配置语料和模型路径，即可开始训练以及使用模型进行预测。

## 使用指南

### 安装
```
$ pip install -r requirements.txt
```
注意：Keras 后端是基于 tensorflow 的。如在有 GPU 的环境，应该安装 tensorflow-gpu 才能调动起显卡。

### 配置参数
在 `parames.py` 这个脚本中配置参数。该脚本的注释中提供了的大部分参数的简单介绍，在此对几个常用的参数及其用途进行详细说明：
- pos_file_path：正面样本的语料的路径。
- neu_file_path: 中性样本的语料的路径。
- neg_file_path: 负面样本的语料的路径。
- w2v_model_path: word2vec 模型的路径。
- random_seed: 随机器的种子，用来保证训练样本中的语料和标签是以同一种方式 shuffle 的，才能一一对应。 Keras 本身提供 shuffle 这一参数，但是在 shuffle 前，keras 会先划分好训练集和验证集，也就是说，如果我们给入的数据是有序的，正面和中性样本在前，负面样本在后，那么在设置 validation_split 后很可能导致验证集全都是负面样本，所以此处采用先对数据集手动 shuffle。（参照：[Keras 使用陷阱](https://keras-cn.readthedocs.io/en/latest/for_beginners/trap/)）
- max_features: 词集的大小。在构建模型时，我们会先预设一个词集，再为此词集构建 embedding layer，超过词集外的词将不被处理。
- max_len: 对句子采样的长度。超过该长度后的句子的词将被去除，而不到此长度的句子将会用 0 来填充。
- dropout: 为输入数据施加 Dropout。Dropout 将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，Dropout 层用于防止过拟合。
- output_size: 代表该层的输出维度，因为完成的是三分的任务，所以设为 3。
- rnn_activation: 激活函数种类。（参照：[深度学习中的激活函数导引](https://zhuanlan.zhihu.com/p/22142013)）
- recurrent_activation: 同上。
- l2_regularization: 正则项系数，越高越不容易过拟合。（我一般是先调 Dropout，再调 l2_regularization）
- CNN Parameters: 不是很熟悉，不多解释。
- loss: 目标函数，或称损失函数，是编译一个模型必须的两个参数之一。损失函数随着要完成任务的不同而不同。（参照[目标函数objectives](https://keras-cn.readthedocs.io/en/latest/other/objectives/) 以及 [wiki](https://en.wikipedia.org/wiki/Loss_function)）
- optimizer: 优化器，优化器是编译Keras模型必要的两个参数之一，是在训练的时候使用的优化算法，对结果影响不大。（参照[优化器](https://keras-cn.readthedocs.io/en/latest/other/optimizers/)）
- model_type: 模型类别。在文灵中提供五种模型：
    - LSTM: RNN 的一种变种，能很好地捕获到序列化的信息。（参照[理解 LSTM 网络](http://blog.csdn.net/ycheng_sjtu/article/details/48792467)）
    - BiLSTM: 双向 LSTM，LSTM 的一种变种，即把句子从尾到头过一遍 LSTM，得到的向量和之前的向量拼起来一起作为句子表征。不过在语料不够大的适合，效果和 LSTM 没有什么差别，但是训练时间要长近一倍。（参照[双向 LSTM](http://blog.csdn.net/aliceyangxi1987/article/details/77094970)）
    - GRU: RNN 的另一种变种，相较 LSTM 其计算量较小，效果与 LSTM 差不多，训练时间较短。（参照[理解 LSTM 网络](http://blog.csdn.net/ycheng_sjtu/article/details/48792467)）
    - BiGRU: 双向 GRU，和双向 LSTM 一个道理。
    - CNNLSTM: 句子被切词，词被表征成词向量后，连接起来可以得到一个长方形的矩阵，可以看出一个图片，用 CNN 在上面先扫一遍，扫完之后在这个序列的基础上用 LSTM 来处理。在语料不足的时候效果与之前没有什么差别，训练时间在 LSTM 和 双向 LSTM 之间。
- batch_size: 在一次梯度下降的适合使用多少样本，一般推荐为 4 的倍数。在 GTX 1060Ti （显存为 4G）上一次最大设置为 256，若设置成 512 则可能会出错。
- num_epoch: 迭代次数。取决于语料大小和使用的 optimizer。一般在 100 左右就足够，但若使用 sgd 作为 optimizer 的话需要大大提高迭代次数。具体还是要在训练过程中，看什么时候收敛（验证集准确率不再上升）来决定一次训练的迭代次数。
- validation_split: 验证集比例。在训练样本中按此比例划分出来作验证集。
- model_path: 储存训练好的模型的路径
- token_path: 储存训练好的 Tokenizer 的路径。注意：对于一类语料来说，Tokenizer 和训练出来的模型必须结对使用，因为句子依靠 Tokenizer 得到 one-hot 的表征，再由此表征在模型的 embedding layer 中转成词向量。
- models_config：此为在做预测的时候的参数。一个预测器需要同时加载其对应的 Keras 模型和对应的 Token。以 Dict 的形式给出是为了方便构造多个预测器。

    
### 训练
```
$ python -m src.train
```

### 预测
```
$ python -m src.predict
```

### 调参指南
- [你有哪些deep learning（rnn、cnn）调参的经验？
](https://www.zhihu.com/question/41631631)
- [Must Know Tips/Tricks in Deep Neural Networks (by Xiu-Shen Wei)](http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html)