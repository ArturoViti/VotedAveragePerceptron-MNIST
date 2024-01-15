# Voted Peceptron
<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue"  alt="Python"/>

<!-- TOC -->
* [Voted Peceptron](#voted-peceptron)
    * [Dataset and Papers](#dataset-and-papers)
  * [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation Guide](#installation-guide)
  * [Usage](#usage)
  * [Project Structure](#project-structure)
  * [Built With](#built-with)
<!-- TOC -->

Implementation of the algorithm _Voted Perceptron_ and its variant _Average_ on MNIST Dataset with 784 variables.

This algorithm classifies MNIST record doing a binary classification: number less of 5 and bigger or equale of 5.
### Dataset and Papers
| URL                                                                                              | Title                                                          | 
|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| [**Freund & Schapire(1999)**](https://link.springer.com/content/pdf/10.1023/A:1007662407062.pdf) | **Large Margin Classification Using the Perceptron Algorithm** |
| [**MNIST_784**](https://www.openml.org/search?type=data&sort=runs&id=554&status=active)          | **The MNIST database of handwritten digits with 784 features** |


## Getting Started
### Prerequisites
To run this project, is necessary have Python 3.x environment
### Installation Guide
After cloning this repository, you need to install the basic dependencies to run the project on your system:
`pip install -r requirements.txt`

You can download MNIST Dataset clicking [**here**](https://www.openml.org/search?type=data&sort=runs&id=554&status=active). 

## Usage
You can run this program in the following ways, using shell parameters:
- Draw **Test Error Plot in function of epochs** (_Figure 2_ of **Freund & Schapire(1999)** paper with $d=1$):


    `python3 main.py --drawTestErrorPlot --withAverageModel`


- Predict single record: predict the label of random record of Test Set. In this mode, script print on terminal
  the correct prediction, a plot of selected record and predicted label. Including `--withAverageModel`, it also predicts
  with average perceptron:


    `python3 main.py --numberEpoch=100 --withAverageModel`

## Project Structure

- `dataSetFunctions.py`: functions to load, manipolate, split and read dataset
- `parameters.py`: global variable to configure dataset splitting
- `perceptronModel.py`: perceptron's class with train and predictions
- `dataset/`: folder with dataset
- `docs/`: folder with documentation and request

## Built With
* [Numpy](https://numpy.org/) - Fundamental package for math operations with arrays
* [Matplotlib](https://matplotlib.org/) - Package to draw result and MNIST Record
* [Argparse](https://docs.python.org/3/library/argparse.html) - Library to manage shell arguments
* [Tqdm](https://pypi.org/project/tqdm/) - Package to draw and manage progress bar