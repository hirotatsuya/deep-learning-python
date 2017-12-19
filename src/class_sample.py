# クラスの定義
class Man:
  # コンストラクタ(初期化メソッド)
  # 第一引数はself
  def __init__(self, name):
    self.name = name # インスタンス変数を初期化
    print('Initialized')
  def hello(self):
    print('hello ' + self.name)
  def goodbye(self):
    print('good-bye ' + self.name)

m = Man('David') # インスタンスの生成
m.hello()
m.goodbye()