import matplotlib.pyplot as plt
from matplotlib.image import imread # 画像表示のためのメソッド
from os import path

image_path = path.dirname(path.dirname(path.abspath(__file__))) + '/dataset/lena.png'
img = imread(image_path)
plt.imshow(img)

plt.show()
