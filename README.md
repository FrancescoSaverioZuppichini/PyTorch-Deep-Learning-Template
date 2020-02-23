# Pytorch Deep Learning Template
### A clean and simple template to kick start your next dl project 🚀🚀
*Francesco Saverio Zuppichini*

In this article, we present you a deep learning template based on Pytorch. This template aims to make it easier for you to start a new deep learning computer vision project with PyTorch. The main features are:

- modularity: we split each logic piece into a different python submodule
- data-augmentation: we included [imgaug](https://imgaug.readthedocs.io/en/latest/)
- ready to go: by using [poutyne](https://pypi.org/project/Poutyne/) a Keras-like framework you don't have to write any train loop.
- [torchsummary](https://github.com/sksq96/pytorch-summary) to show a summary of your models
- reduce the learning rate on a plateau
- auto-saving the best model
- experiment tracking with [comet](https://www.comet.ml/)
- logging using python [logging](https://docs.python.org/3/library/logging.html) module
- a playground notebook to quick test/play around
## Installation
Clone the repo and go inside it. Then, run:

```
pip install -r requirements.txt
```

### Motivation
Let's face it, usually data scientists are not software engineers and they usually end up with spaghetti code, most of the time on a big unusable Jupiter-notebook. With this repo, I have proposed a clean example of how your code should be split and modularized to make scalability and sharability possible. In this example, we will try to classify Darth Vader and Luke Skywalker. We have 100 images per class gathered using google images. The dataset is [here](https://drive.google.com/open?id=1LyHJxUVjOgDIgGJL4MnDhA10xjejWuw7). You just have to extract it in this folder and run main.py. We are fine-tuning resnet18 and it should be able to reach > 90% accuracy in 5/10 epochs.
## Structure
The template is inside `./template`.
```
.
├── callbacks // here you can create your custom callbacks
├── checkpoint // were we store the trained models
├── data // here we define our dataset
│ └── transformation // custom transformation, e.g. resize and data augmentation
├── dataset // the data
│ ├── train
│ └── val
├── logger.py // were we define our logger
├── losses // custom losses
├── main.py
├── models // here we create our models
│ ├── MyCNN.py
│ ├── resnet.py
│ └── utils.py
├── playground.ipynb // a notebook that can be used to fast experiment with things
├── Project.py // a class that represents the project structure
├── README.md
├── requirements.txt
├── test // you should always perform some basic testing
│ └── test_myDataset.py
└── utils.py // utilities functions
```
**We strongly encourage to play around with the template**
### Keep your structure clean and concise
Every deep learning project has at least three mains steps:
- data gathering/processing
- modeling
- training/evaluating
## Project
One good idea is to store all the paths at an interesting location, e.g. the dataset folder, in a shared class that can be accessed by anyone in the folder. You should never hardcode any paths and always define them once and import them. So, if you later change your structure you will only have to modify one file.
If we have a look at `Project.py` we can see how we defined the `data_dir` and the `checkpoint_dir` once for all. We are using the 'new' [Path](https://docs.python.org/3/library/pathlib.html) APIs that support different OS out of the box, and also makes it easier to join and concatenate paths.
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Skeletron/master/images/Project.png)
For example, if we want to know the data location we can :
```python3
from Project import Project
project = Project() 
print(project.data_dir) # /foo/baa/…/dataset
```
## Data
In the `data` package you can define your own Dataset, as always by subclassing `torch.data.utils.Dataset`, exposing transformations and utilities to work with your data.
In our example, we directly used `ImageDataset` from `torchvision` but we included a skeleton for a custom `Dataset` in `/data/MyDataset`
### Transformation
You usually have to do some preprocessing on the data, e.g. resize the images and apply data augmentation. All your transformation should go inside `.data.trasformation`. In our template, we included a wrapper for
[imgaug](https://imgaug.readthedocs.io/en/latest/)
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Skeletron/master/images/transformation.png)
### Dataloaders
As you know, you have to create a `Dataloader` to feed your data into the model. In the `data.__init__.py` file we expose a very simple function `get_dataloaders` to automatically configure the *train, val and test* data loaders using few parameters
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Skeletron/master/images/data.png)
## Losses
Sometimes you may need to define your custom losses, you can include them in the `./losses` package. For example
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Skeletron/master/images/losses.png)
## Metrics
Sometimes you may need to define your custom metrics. For example
![alt](https://raw.githubusercontent.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Skeletron/master/images/metrics.png)
## Logging 
We included python [logging](https://docs.python.org/3/library/logging.html) module. You can import and use it by:

```python
from logger import logging
logging.info('print() is for noobs')
```

## Models
All your models go inside `models`, in our case, we have a very basic cnn and we override the `resnet18` function to provide a frozen model to finetune.

![alt](https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Skeletron/blob/master/images/resnet.png?raw=true)
## Train/Evaluation
In our case we kept things simple, all the training and evaluation logic is inside `.main.py` where we used [poutyne](https://pypi.org/project/Poutyne/) as the main library. We already defined a useful list of callbacks:
- learning rate scheduler
- auto-save of the best model
- early stopping
Usually, this is all you need!
![alt](https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Skeletron/blob/master/images/main.png?raw=true)
### Callbacks 
You may need to create custom callbacks, with [poutyne](https://pypi.org/project/Poutyne/) is very easy since it support Keras-like API. You custom callbacks should go inside `./callbacks`. For example, we have created one to update Comet every epoch.
![alt](https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Skeletron/blob/master/images/CometCallback.png?raw=true)

### Track your experiment
We are using [comet](https://www.comet.ml/) to automatically track our models' results. This is what comet's board looks like after a few models run.
![alt](https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Skeletron/blob/master/images/comet.jpg?raw=true)
Running `main.py` produces the following output:
![alt](https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Skeletron/blob/master/images/output.jpg?raw=true)
## Utils
We also created different utilities function to plot both dataset and dataloader. They are in `utils.py`. For example, calling `show_dl` on our train and val dataset produces the following outputs.
![alt](https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Skeletron/blob/master/images/Figure_1.png?raw=true)
![alt](https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Skeletron/blob/master/images/Figure_2.png?raw=true)
As you can see data-augmentation is correctly applied on the train set
## Conclusions
I hope you found some useful information and hopefully it this template will help you on your next amazing project :)

Let me know if you have some ideas/suggestions to improve it.

Thank you for reading

## TODO 
- [ ] one example for [lightning](https://github.com/williamFalcon/pytorch-lightning)
- [ ] one example with [fastai](https://www.fast.ai/)
- [ ] show how to setup anaconda 
