import nltk
import os
import torch
import torch.utils.data as data
from raw_program.vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json


def get_loader(transform,
               mode='train',
               batch_size=1,
               vocab_threshold=None,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0,
               cocoapi_loc='/opt'):
    """Returns the data loader.
    Args:
      transform: Image transform.
      mode: One of 'train' or 'test'.
      batch_size: Batch size (if in testing mode, must have batch_size=1).
      vocab_threshold: 最小词频阈值
      vocab_file: 包含词汇的文件
      start_word: 标志语句开始的词汇
      end_word: 标志语句结束的词汇
      unk_word: 小于词频阈值的词汇
      vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from from existing vocab_file, if it exists.
      num_workers: Number of subprocesses to use for data loading 
      cocoapi_loc: The location of the folder containing the COCO API: https://github.com/cocodataset/cocoapi
    """
    
    assert mode in ['train', 'test'], "mode一定要是'train'或者'test'."
    if vocab_from_file==False: assert mode == 'train', "为了从captions文件中建立vocab，mode一定要是训练模式(mode='train')."

    # 基于mode(train, val, test), 设置img_folder和annotations_file，并进行输入检查
    if mode == 'train':
        if vocab_from_file: assert os.path.exists(vocab_file), "vocab_file还不存在，把vocab_from_file变成False来建立vocab_file."
        img_folder = os.path.join(cocoapi_loc, 'coco2017/train2017/')
        annotations_file = os.path.join(cocoapi_loc, 'coco2017/annotations_trainval2017/captions_train2017.json')
    if mode == 'test':
        assert batch_size == 1, "测试模式的batch_size需要设置为1"
        assert os.path.exists(vocab_file), "需要首先在训练模式下生成generate vocab.pkl"
        assert vocab_from_file == True, "把vocab_from_file设成True"
        img_folder = os.path.join(cocoapi_loc, 'coco2017/test2017/')
        annotations_file = os.path.join(cocoapi_loc, 'coco2017/image_info_test2017/annotations/image_info_test2017.json')

    # COCO caption数据集
    dataset = CoCoDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)

    if mode == 'train':
        # 随机采样标题长度，并使用该长度采样索引
        indices = dataset.get_train_indices()
        # 创建并分配一个批次采样器，以检索之前返回的索引对应的元素（就是随便选了一些索引出来，根据索引抓元素）
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        # COCO数据集的data loader
        data_loader = data.DataLoader(dataset=dataset, 
                                      num_workers=num_workers,
                                      batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                              drop_last=False))
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)

    return data_loader


class CoCoDataset(data.Dataset):
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, end_word, unk_word, annotations_file, vocab_from_file, img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        # 初始化一个Vocabulary类，详见vocabulary.py
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word, end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder
        if self.mode == 'train':
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print('正在对caption分词...')
            # 将所有输入的caption变成小写并且分词，一个annotation对应一个元素
            all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))]
            # 得到的是每一个annotation对应的分词个数的列表
            self.caption_lengths = [len(token) for token in all_tokens]
        else:
            test_info = json.loads(open(annotations_file).read())
            # 一个元素就对应一个test中的一个图片文件名
            self.paths = [item['file_name'] for item in test_info['images']]
        
    def __getitem__(self, index):
        # 获取图片及其caption
        if self.mode == 'train':
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            path = self.coco.loadImgs(img_id)[0]['file_name']

            # 把图片变成张量并且预处理
            image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            image = self.transform(image)

            # 把caption变成word ids的张量
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())  # 变成小写并且完成分词
            caption = []
            caption.append(self.vocab(self.vocab.start_word))           # 加开头
            caption.extend([self.vocab(token) for token in tokens])     # 加中间内容
            caption.append(self.vocab(self.vocab.end_word))             # 加结尾
            caption = torch.Tensor(caption).long()                      # 从分词列表变成长整型张量

            # 返回预处理好的图片与caption张量
            return image, caption

        # 测试模式下获取图片
        else:
            path = self.paths[index]

            # 把图片变成张量并且预处理
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            # 返回原始图片和预处理好的图片
            return orig_image, image

    def get_train_indices(self):
        # 从分词长度的列表里随便抓一个出来
        sel_length = np.random.choice(self.caption_lengths)
        # 找到分词列表中长度根这个抓出来的长度一样的的索引(np.where[0]表示行索引，[1]表示列索引)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        # 再从这些索引里选batch_size这么多个索引出来
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        if self.mode == 'train':
            return len(self.ids)
        else:
            return len(self.paths)
