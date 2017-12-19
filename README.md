# deep-learning-python

## Requirements
- python 3.6
- pipenv 9.0.0

## First
- pipenvのインストール

```
pip install pipenv
```

## Setup
- 仮想環境にpythonのパッケージをインストール

```
pipenv install
```

#### macosの場合`matplotlib`を使うために`matplotlibrc`を修正する必要がある
- matplotlibrcの場所を特定

```
pipenv run python -c "import matplotlib;print(matplotlib.matplotlib_fname())"
```

- matplotlibrcの`backend : macosx`を`backend : Tkagg`に修正

```
vi (上記のパス)
```

## Usage
- コンソール実行

```
pipenv run python (target).py
```

- jupyter

```
pipenv run jupyter notebook
```

## Reference
- 書籍: ゼロから作るDeep Learning
- リポジトリ: `https://github.com/oreilly-japan/deep-learning-from-scratch`
