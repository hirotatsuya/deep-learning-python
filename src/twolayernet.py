import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
import matplotlib.pylab as plt

# シグモイド関数
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# ソフトマックス関数
def softmax(x):
  if x.ndim == 2:
    x = x.T
    x = x - np.max(x, axis=0)
    y = np.exp(x) / np.sum(np.exp(x), axis=0)
    return y.T 
  x = x - np.max(x) # オーバーフロー対策
  return np.exp(x) / np.sum(np.exp(x))

# 交差エントロピー誤差
def cross_entropy_error(y, t):
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
  if t.size == y.size:
    t = t.argmax(axis=1)
  batch_size = y.shape[0]
  return -np.sum(t * np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def numerical_gradient(f, x):
  h = 1e-4 # 0.0001
  grad = np.zeros_like(x)
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    idx = it.multi_index
    tmp_val = x[idx]
    x[idx] = float(tmp_val) + h
    fxh1 = f(x) # f(x+h)
    x[idx] = tmp_val - h 
    fxh2 = f(x) # f(x-h)
    grad[idx] = (fxh1 - fxh2) / (2*h)
    x[idx] = tmp_val # 値を元に戻す
    it.iternext()   
  return grad

class TwoLayerNet:
  # input_size: 入力ニューロン数, hidden_size: 隠れ層のニューロン数, output_size: 出力層のニューロン数
  def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
    # ニューラルネットワークのパラメータを保持するディクショナリ変数
    self.params = {}
    # W: 重み, b: バイアス
    # 1: 一層目, 2: 二層目
    self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  # 推論を行う
  # x: 画像データ
  def predict(self, x):
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']
    a1 = np.dot(x, W1) + b1 # 畳み込み
    z1 = sigmoid(a1) # シグモイド関数
    a2 = np.dot(z1, W2) + b2
    y = softmax(a2) # ソフトマックス関数
    return y

  # 損失関数の値を求める
  # x: 入力データ(画像データ), t: 教師データ(正解ラベル)
  def loss(self, x, t):
    y = self.predict(x)
    return cross_entropy_error(y, t) # 交差エントロピー誤差

  # 認識制度を求める
  # x: 入力データ(画像データ), t: 教師データ(正解ラベル)
  def accuracy(self, x, t):
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)
    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy

  # 重みパラメータに対する勾配を求める
  # x: 入力データ(画像データ), t: 教師データ(正解ラベル)
  def numerical_gradient(self, x, t):
    loss_W = lambda W: self.loss(x, t)
    grads = {} # 勾配を保持するディクショナリ変数
    # W1: 一層目の重みの勾配, b1: 一層目のバイアスの勾配
    grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
    grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
    grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
    return grads

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_loss_list = []
train_acc_list = []
test_acc_list = []

# ハイパーパラメータ
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
  # ミニバッチの取得
  batch_mask = np.random.choice(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]

  # 勾配の計算
  grad = network.numerical_gradient(x_batch, t_batch)

  # パラメータの更新
  for key in ('W1', 'b1', 'W2', 'b2'):
    network.params[key] -= learning_rate * grad[key]

  # 学習経過の記録
  loss = network.loss(x_batch, t_batch)
  train_loss_list.append(loss)

  if i % iter_per_epoch == 0:
    train_acc = network.accuracy(x_train, t_train)
    test_acc = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
