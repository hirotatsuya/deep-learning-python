import numpy as np # ライブラリの読み込み

array = [1.0, 2.0, 3.0]
x = np.array(array) # NumPy用の配列を作成する
print(x)
print(type(x))

array_2 = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
y = np.array(array_2)
print(y[y < 2])