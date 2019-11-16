# Pytorch Deep Learning Template
### A clean and simple template to kick start your project
*Francesco Saverio Zuppichini*

This template aims to make it easier for you to start a new deep learning computer vision project with PyTorch. The main features are:

- modularity: we splitted each logic piece into a different python submodule
- data-augmentation: we included [imgaug](https://imgaug.readthedocs.io/en/latest/)
- ready to go: by using [poutyne](https://pypi.org/project/Poutyne/) a Keras like framework you don't have to write any train loop

### Motivation
Let's face it, most of data scientist are not software engineers and they usually end up with spaghetti code, most of the times on a big unusable jupyter-notebook. With this repo you have a clean example of how your code should be splitted and modularized in order to make scalability and sharability possible.

## Structure
```
.
├── callbacks
│   ├── CometCallback.py
│   └── __init__.py
├── checkpoint
│   ├── 1573678553.5113187-model.pt
│   └── 1573905051.4520051-model.pt
├── config.json
├── data
│   ├── __init__.py
│   ├── MyDataset.py
│   ├── __pycache__
│   └── transformation
├── dataset
│   ├── train
│   └── val
├── logger.py
├── losses
│   └── __init__.py
├── main.py
├── models
│   ├── __init__.py
│   ├── MyCNN.py
│   ├── __pycache__
│   ├── resnet.py
│   └── utils.py
├── playground.ipynb
├── Project.py
├── __pycache__
│   ├── logger.cpython-37.pyc
│   ├── Project.cpython-37.pyc
│   └── utils.cpython-37.pyc
├── README.md
├── requirements.txt
├── test
│   └── test_myDataset.py
└── utils.py
```