from PIL import Image
from pycocotools.coco import COCO
from raw_program.data_loader import get_loader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import torch
from raw_program.model import EncoderCNN, DecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COCOAPIROOT = r"D:\学习资料\实验室"

# Define a transform to pre-process the testing images.
transform_test = transforms.Compose([
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Create a data loader.
data_loader = get_loader(transform=transform_test,
                         mode='test',
                         cocoapi_loc=COCOAPIROOT)

# TODO #2: Specify the saved models to load.
encoder_file = 'encoder-3.pkl'
decoder_file = 'decoder-3.pkl'

# TODO #3: Select appropriate values for the Python variables below.
embed_size = 512
hidden_size = 512

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the encoder and decoder, and set each to inference mode.
encoder = EncoderCNN(embed_size)
encoder.eval()
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
decoder.eval()

# Load the trained weights.
encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file),  map_location='cpu'))
decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file),  map_location='cpu'))

# Move models to GPU if CUDA is available.
encoder.to(device)
decoder.to(device)


def clean_sentence(output):
    # Look for key tokens
    # 0 = <start>
    # 1 = <end>
    # 18 = .
    start = 0
    end = len(output) - 1
    point = end
    for i in range(len(output)):
        if output[i] == 0:
            start = i + 1
            continue
        if output[i] == 18:
            point = i
            continue
        if output[i] == 1:
            end = i
            break
    if point > end:
        point = end
    sentence = " ".join([data_loader.dataset.vocab.idx2word[x] for x in output[start:point]])
    # sentence += "."

    return sentence


def generator():
    dir_path = input("请输入图片的全局路径：")
    dir_path = dir_path.replace('"', "")
    original_image = Image.open(dir_path)
    plt.imshow(np.squeeze(original_image))
    plt.title('Original Image')
    plt.show()
    image = transform_test(original_image)
    image = image.unsqueeze(1).permute(1, 0, 2, 3)
    image = image.to(device)
    features = encoder(image).unsqueeze(1)
    output = decoder.sample(features)
    sentence = clean_sentence(output)
    print("The caption is:" + sentence)


def main():
    breakjudger = 1
    while True:
        if breakjudger == 0:
            break
        generator()
        breakjudger = input("如果想继续使用请输入1，如果想退出请输入0：")
        assert breakjudger != 1 or breakjudger != 0, "请输入1或者0"


main()
