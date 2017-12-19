import numpy as np
import matplotlib.pylab as plt

# numpy配列を引数に受け取り、配列の各要素に対してステップ関数を実行し、結果を配列として返す
def step_function(x):
  return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
print(y)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()