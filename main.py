from tqdm import notebook

from dataSetFunctions import *
from perceptronModel import VotedPerceptron, AveragePerceptron
from parameters import *
import matplotlib.pyplot as plt

import argparse
import random
from tqdm import tqdm

# Gestione dei parametri
parser = argparse.ArgumentParser( description='Usage: main.py --drawTestErrorPlot or main.py --numberEpoch <number> '
                                              '--withAverageModel' )
parser.add_argument('--drawTestErrorDiagram', action='store_true',
                    help='Draw Test Error Diagram on 10 Epochs' )
parser.add_argument('--numberEpoch', type=int, default=5, help='Epoch Number (Default 10)')
parser.add_argument('--withAverageModel', action='store_true',
                    help='Include Average Perceptron Model Prediction' )
args = parser.parse_args()

# Carica il Dataset con Numpy
print( "Loading Dataset ⏳" )
mnistData = loadDataset('dataset/mnist_784.arff')
trainData, testData, testLabel = splitDataset(mnistData, SPLIT_RATIO)
trainLabels, trainData = classifyDataset(trainData, CLASS_INDEX)
numberEpoch = range(args.numberEpoch)
print( "Dataset loaded & splitted successfully ✅" )


if args.drawTestErrorDiagram:
    # Traing e Predizioni Voted Perceptron
    errorVoted = []
    for i in tqdm( range(1, args.numberEpoch + 1 ), desc='Voted Perceptron Training & Prediction'):
        errorVoted.append(0)
        vPerceptron = VotedPerceptron( n_iter=i )
        vPerceptron.train( trainData, trainLabels )
        prediction = vPerceptron.multiplePredict( testData )
        for j in range(len(prediction)):
            if prediction[j] != (-1 if testLabel[j] < 5 else 1):
                errorVoted[i-1] += 1

    errorVotedPercent = [(value / len(testData)) * 100 for value in errorVoted]
    plt.plot( numberEpoch, errorVotedPercent, label='Voted', color='r', linewidth=2, markersize=12 )

    if args.withAverageModel:
        # Traing e Predizioni Average Perceptron
        errorAveraged = []
        for i in tqdm( range(1, args.numberEpoch + 1 ), desc='Average Perceptron Training & Prediction'):
            errorAveraged.append(0)
            avgPerceptron = AveragePerceptron(n_iter=i)
            avgPerceptron.train(trainData, trainLabels)
            prediction = avgPerceptron.multiplePredict( testData )
            for j in range(len(prediction)):
                if prediction[j] != (-1 if testLabel[j] < 5 else 1):
                    errorAveraged[i - 1] += 1

        errorAveragedPercent = [ (value / len(testData)) * 100 for value in errorAveraged ]
        plt.plot( numberEpoch, errorAveragedPercent, label='Average', linestyle='dotted', linewidth=2,
                  markersize=12 )

    plt.xlabel( 'Epoch Number' )
    plt.xticks( range(1, max(numberEpoch) + 1, 1) )
    plt.xticks( np.arange(len(numberEpoch)), np.arange(1, len(numberEpoch) + 1) )
    plt.ylabel( 'Test Error (%)')

    plt.title( 'Test Error Perceptron for d = 1' )
    plt.legend()
    plt.show()
else:
    vPerceptron = VotedPerceptron( n_iter=args.numberEpoch )
    vPerceptron.train( trainData, trainLabels )

    randomTestIndex = random.randint(0, len(testData) - 1)
    prediction = vPerceptron.predict( testData[randomTestIndex] )
    print( "Test Set Index: " + str(randomTestIndex) )
    print( "Test Set Value: " + str(testLabel[randomTestIndex]) )
    print( "Voted Perceptron Prediction: " + str(prediction) )

    if args.withAverageModel:
        avgPerceptron = AveragePerceptron( n_iter=args.numberEpoch )
        avgPerceptron.train( trainData, trainLabels )
        avgPrediction = avgPerceptron.predict( testData[randomTestIndex] )
        print( "Average Perceptron Prediction: " + str(avgPrediction) )

    drawMNISTRecord( testData[randomTestIndex] )





