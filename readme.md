# PRIS: Practical robust invertible network for image steganography
This repo is the official code for

* **PRIS: Practical robust invertible network for image steganography**



 
## Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux)).
- [PyTorch = 1.11.0](https://pytorch.org/) .
- See requirements.txt for other dependencies.


## Get Started
- Run `python main.py` for training.


## Dataset
- In this paper, we use the commonly used dataset DIV2K.

- For train or test on your own dataset, change the code in `config.py`:

    `line27:  TRAIN_PATH = '' ` 

    `line28:  VAL_PATH = '' `


## Demo
- Here we provide a [Demo](http://47.94.105.69/hide/).

- You can hide a secret image into a host image in our Demo by clicking your mouse.


