import numpy as np

print('One Dimensional')
A = np.arange(1, 4, 1)
print(A) # numpy配列
print(np.ndim(A)) # 次元数
print(A.shape) # 形状

print('Two Dimensional')
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))
print(B.shape)

print('Inner Product')
C = np.array([[1, 2], [3, 4]])
print(C.shape)
D = np.array([[5, 6], [7, 8]])
print(D.shape)
print(np.dot(C, D)) # 内積
