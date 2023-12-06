import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize
import random

IMG_WIDTH = 128
IMG_HEIGHT = 128
N_CHANNELS = 3 #4th layer is transparency which we neglect

BASE_TRAIN_PATH = 'data/train/'
BASE_TEST_PATH = 'data/test/'

train_obj = os.scandir(BASE_TRAIN_PATH)
test_obj = os.scandir(BASE_TEST_PATH)
train_ids = [x.name for x in train_obj]
test_ids = [x.name for x in test_obj]

X_train = np.zeros((len(train_ids), IMG_WIDTH, IMG_HEIGHT, N_CHANNELS))
y_train = np.zeros((len(train_ids), IMG_WIDTH, IMG_HEIGHT, 1))


for n, id in tqdm(enumerate(train_ids), total=len(train_ids)):
    image_path = os.path.join(BASE_TRAIN_PATH, id, "images", id, ".png")
    img = Image.open(image_path)
    img = np.asarray(img)[:,:,:,:N_CHANNELS]
    X_train[1, :, :, :] = resize(img, (IMG_WIDTH, IMG_HEIGHT), mode="constant", preserve_range=True)

index = random.randint(0, len(train_ids))


# id = next(obj).name
# img_path = "data/train/" + str(id) + "/images/" + str(id) + ".png"
# img = Image.open(img_path)
# npArr = np.asarray(img)

