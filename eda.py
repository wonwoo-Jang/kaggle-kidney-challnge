from PIL import Image
import os

path = './dataset/train'
# for dir in os.listdir(path):
img = Image.open(os.path.join(path, 'kidney_1_dense/labels/2200.tif'))
print(img.size)