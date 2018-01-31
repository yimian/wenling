## WENLING (文灵) 情感分析包文档

WENLING (文灵) 是一个借助 Keras 构建的模型包，提供数种常用的用作情感分类的模型。通过简单地配置语料和模型路径，即可开始训练以及使用模型进行预测。

### 使用指南

#### 安装
```angular2html
$ pip install -r requirements.txt
```
注意：Keras 后端是基于 tensorflow 的。如在有 GPU 的环境，应该安装 tensorflow-gpu 才能调动起显卡。

#### 配置参数



#### 训练
Just run the command in root directory of project:
```
$ python -m src.train
```

#### 预测
```angular2html
$ python -m src.predict
```


