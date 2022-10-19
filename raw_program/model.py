import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)   # 用预训练好的resnet50来做
        for param in resnet.parameters():
            param.requires_grad_(False)
        modules = list(resnet.children())[:-1]      # 只需要编码，不需要完成分类任务，所以不要最后一层全连接层
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)   # 用一个全连接层完成维数变化，变成decoder好处理的维数

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)  # 这里对张量进行了变形，展平
        features = self.embed(features)                 # 通过线性层将特征向量变换为与词嵌入大小相同的大小
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_p=0.1):
        super(DecoderRNN, self).__init__()
        
        # 模型参数
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
    
        # 词嵌入
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

        # LSTM
        # batch_first=True:
        # 输入张量形状: (batch_size, caption_length, in_features/embedding_dim)
        # 输出张量形状: (batch_size, caption_length, out/hidden)
        self.lstm = nn.LSTM(self.embed_size,
                            self.hidden_size,
                            self.num_layers, 
                            dropout=drop_p,
                            batch_first=True)
        
        # Dropout层
        self.dropout = nn.Dropout(drop_p)
        
        # 用全连接层将隐藏层值转换为索引向量作为输出
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        
        # softmax函数，又称归一化指数函数。它是二分类函数sigmoid在多分类上的推广，目的是将多分类的结果以概率的形式展现出来
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, features, captions):
        
        # 取前n-1个token，最后一个<end>不要
        # 在参考文献中是这么搞的
        captions = captions[:, :-1]
        
        # 词嵌入
        # 在词嵌入后可以得到一个嵌入后的向量
        embed = self.embedding(captions)
        
        # 对image features变形: features.size(0) == batch_size
        features = features.view(features.size(0), 1, -1)
        
        # 把features和词嵌入后的向量拼接起来
        # 图像特征具有嵌入大小，应首先传递给LSTM，所以它们位于前面
        # 输入张量的形状:(batch_size, sequence_length, embedding_size)
        inputs = torch.cat((features, embed), dim=1)
        
        # LSTM: 输入的序列: [image, token_1, token_2, ..., token_(n-1)]
        # 如果我们不传递任何隐藏状态，它默认为0，但始终返回一个隐藏状态
        # lstm输出张量的形状: (batch_size, sequence_length, hidden_size)
        # hidden -> (h, c)
        # h size: (1, batch_size, hidden_size)
        # c size: (1, batch_size, hidden_size)
        lstm_out, hidden = self.lstm(inputs)

        # 为了传到线性层我们需要的张量形状为：(-1, hidden_dimension),1 i.e., (batch_size*sequence_length, hidden_dimension)
        out = lstm_out.reshape(lstm_out.size(0)*lstm_out.size(1), lstm_out.size(2))
        
        # 先经过一个dropout层
        out = self.dropout(out)
        
        # 再经过一个全连接层从隐藏维度空间映射到词汇表维度空间
        out = self.fc(out)
        
        # 输出张量形状: (batch_size*sequence_length, vocabulary_size)
        # 我们需要对其变形
        out = out.view(lstm_out.size(0), lstm_out.size(1), -1)
        
        # Log-Softmax / SoftMax: dim=2
        #out = self.softmax(out) # a probability for each token in the vocabulary
        
        # 返回最终输出，不返回隐藏层状态因为后面用不到
        return out
    
    def sample(self, inputs, states=None, max_len=20):
        """接收预处理后的图片张量(inputs)返回预测的句子（词典的索引）)
        
        变量:
            inputs (张量): 预处理的图片张量
            states(张量): 隐藏层状态初始化
            max_len (整形): 返回索引数组的最长长度
            
        返回:
            outputs (列表): 索引列表; length: max_len
        """
        
        # 将输出列表初始化为值均为<end>的列表
        outputs = [1]*max_len
        
        # 初始化隐藏层状态为0
        hidden = states
        
        # 传递图像并获取tokens序列。这类似于前向传播
        with torch.no_grad():
            for i in range(max_len):
                # lstm_out size: (batch_size=1, sequence_length=1, hidden_size)
                lstm_out, hidden = self.lstm(inputs, hidden)
                # out size: (1, hidden_size)
                out = lstm_out.reshape(lstm_out.size(0)*lstm_out.size(1), lstm_out.size(2))
                # fc_out size (1, vocabulary_size)
                fc_out = self.fc(out)
                
                # 计算各个索引的概率
                p = F.softmax(fc_out, dim=1).data

                # 转到cpu
                p = p.cpu() 
                # 用top_k sampling 得到下一个词的指标
                top_k = 5
                p, top_indices = p.topk(top_k)
                top_indices = top_indices.numpy().squeeze()
                # 在一些元素随机的情况下选取最可能的下一个索引
                p = p.numpy().squeeze()
                token_id = int(np.random.choice(top_indices, p=p/p.sum()))
                
                # 将这个索引存到输出列表中
                outputs[i] = token_id
                
                # 从output token建立下一次input
                # inputs size: (1, 1, embedding_size=512)
                input_token = torch.Tensor([[token_id]]).long()
                inputs = self.embedding(input_token)

                if token_id == 1:
                    # <end>
                    break
        
        return outputs
