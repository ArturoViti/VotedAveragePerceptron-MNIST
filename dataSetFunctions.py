from scipy.io import arff
import matplotlib.pyplot as plt
import numpy as np


def loadDataset( dsPath ):
    """
        Carica un Dataset arff come Numpy Array
        :param dsPath: Percorso relativo del dataset.arff
        :return: Numpy Dataset
    """
    data, meta = arff.loadarff(dsPath)
    df = np.array(data.tolist(), dtype=int)
    return df


def splitDataset( numpyDataset, splitRatio ):
    """
        Divide il dataset in training set e test set utilizzando lo split ratio
        :param numpyDataset: Numpy Dataset
        :param splitRatio: Percentuale normalizzata a 1 della dimensione del training set
        :return: Training Set, Test Set, Test Label
    """
    np.random.shuffle(numpyDataset)

    splitIndex = int( len(numpyDataset) * splitRatio )
    trainData = numpyDataset[:splitIndex]
    testData = numpyDataset[splitIndex:]

    return trainData, testData[:, :-1], testData[:, -1:]


def classifyDataset( numpyDataset, temporaryClassIndex ):
    """
        Crea le etichette partendo da quelle del dataset. Per le cifre < 5 => -1, 1 altrimenti
        :param numpyDataset: Numpy Dataset
        :param temporaryClassIndex: Posizione della label corrente nel dataset
        :return: Array Labels, Training Data
    """
    trainTemporaryClasses = numpyDataset[:, temporaryClassIndex]
    trainClasses = [ (-1 if int(tempLabel) < 5 else 1) for tempLabel in trainTemporaryClasses ]
    trainData = numpyDataset[:, :-1]
    return trainClasses, trainData


def drawMNISTRecord( mnistRecord ):
    """
        Disegna un record MNIST sulla scala di grigi
        :param mnistRecord: il record MNIST
        :return: None
    """
    mnistImage = np.array(mnistRecord).reshape( 28, 28 )
    plt.imshow( mnistImage, cmap='gray' )
    plt.show()
