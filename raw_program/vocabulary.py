import nltk
import pickle
import os.path
from pycocotools.coco import COCO
from collections import Counter


class Vocabulary(object):

    def __init__(self, vocab_threshold, vocab_file='./vocab.pkl', start_word="<start>", end_word="<end>",
                 unk_word="<unk>", annotations_file='../coco2017/annotations_trainval2017/captions_train2017.json',
                 vocab_from_file=False):
        """初始化Vocabulary类
        变量:
            vocab_threshold: 最小词频阈值
            vocab_file: 包含词汇的文件.
            start_word: 标志语句开始的词汇
            end_word: 标志语句结束的词汇
            unk_word: 小于词频阈值的词汇
            annotations_file: 训练annotation.json存储路径
            vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from from existing vocab_file, if it exists.
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()                            # 初始化的时候就调用这个方法处理字典了

    def get_vocab(self):
        """从已有的文件中加载词汇或者建立词汇文件"""
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                # 因为之前把self中的所有东西都存进去了这里才能这么取
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print('从vocab.pkl文件中加载词汇成功')
        else:
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)                    # 把self中的所有东西都存进去了
            print('初始化vocab.pkl文件成功')
        
    def build_vocab(self):
        """填充字典以将标记转换为整数（反之亦然）"""
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()

    def init_vocab(self):
        """初始化一个为了将caption转变为张量而构建的字典"""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """向字典中加词"""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        """读入caption，并将所有达到或超过阈值的标记添加到词汇表中。"""
        coco = COCO(self.annotations_file)
        counter = Counter()     # 形成一个存词的字典
        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            tokens = nltk.tokenize.word_tokenize(caption.lower())   # 将所有输入的caption变成小写并且分词，一个annotation对应一个元素
            counter.update(tokens)                                  # 将这个词在字典中的值也就是词频更新

            if i % 100000 == 0:
                print("\r[%d/%d] 正在读取captions并根据其分词建立词典..." % (i, len(ids)))

        words = [word for word, cnt in counter.items() if cnt >= self.vocab_threshold]  # 只留下符合阈值的

        for i, word in enumerate(words):
            self.add_word(word)

    def __call__(self, word):
        """将单词转变为字典中对应的数值，完成映射"""
        if word not in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)
