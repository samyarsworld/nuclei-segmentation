import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


IMG_WIDTH = 128
IMG_HEIGHT = 128
N_CHANNELS = 3

BASE_TRAIN_PATH = 'data/train'
BASE_TEST_PATH = 'data/test'

train_obj = os.scandir(BASE_TRAIN_PATH)
test_obj = os.scandir(BASE_TEST_PATH)
train_ids = [x.name for x in train_obj]
test_ids = [x.name for x in test_obj]

X_train = np.zeros(size=(len(train_ids), IMG_WIDTH, IMG_HEIGHT, N_CHANNELS), dtype=np.float16)
y_train = np.zeros(size=(len(train_ids), IMG_WIDTH, IMG_HEIGHT, 1), dtype=np.bool)



# id = next(obj).name
# img_path = "data/train/" + str(id) + "/images/" + str(id) + ".png"
# img = Image.open(img_path)
# npArr = np.asarray(img)

