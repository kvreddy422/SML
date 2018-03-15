
from sklearn.datasets import fetch_mldata;
import math as math
import pandas as pd
import operator
import numpy
import scipy
from collections import Counter
import matplotlib.pyplot as plt
class KNN(object):

    def input_data(self):
        custom_data_home='C:/Users/Vaibhav Kalakota/Desktop/Spring 2018/SML/Assignments/Assignment 2';
        mnist = fetch_mldata('MNIST original', data_home=custom_data_home);
        # print(mnist.data.shape)

    def convert(self,imgf,labelf,outf,n):
        f = open(imgf, "rb")
        o = open(outf, "w")
        l = open(labelf, "rb")

        f.read(16)
        l.read(8)
        images = []

        for i in range(n):
            image = [ord(l.read(1))]
            for j in range(28 * 28):
                image.append(ord(f.read(1)))
            images.append(image)

        for image in images:
            o.write(",".join(str(pix) for pix in image) + "\n")
        f.close()
        o.close()
        l.close()
    def getEuclidianDistance(self,dataPoint1,dataPoint2):
        length=dataPoint1.shape[1];
        distance=0;
        for loop in range(length):
            if(loop>0):
                distance+=pow(dataPoint1[loop].iloc[0]-dataPoint2[loop].iloc[0],2)
        return math.sqrt(distance)

    def KNNAlgo(self,testSet,trainingSet):
        # K = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
        K = [1];
        kTest = {1: [], 9: [], 19: [], 29: [], 39: [], 49: [], 59: [], 69: [], 79: [], 89: [], 99: []}
        kTraining = {1: [], 9: [], 19: [], 29: [], 39: [], 49: [], 59: [], 69: [], 79: [], 89: [], 99: []}
        accuracy_K=[];
        for Kinput in kTest.keys():
            accuracy = [];
            getClass = []
            finalClass=[]
            finalClass_training = []
            ary = scipy.spatial.distance.cdist(testSet.iloc[:, 1:], trainingSet.iloc[:, 1:], metric='euclidean')
            result = pd.DataFrame(ary).T
            ary_training = scipy.spatial.distance.cdist(trainingSet.iloc[:, 1:], trainingSet.iloc[:, 1:], metric='euclidean')
            result_training = pd.DataFrame(ary_training).T

            for testLoop in range(testSet.shape[0]):
                distances_from_i_dict = dict(enumerate(result[testLoop]))
                distances_from_i_dict=sorted(distances_from_i_dict.items(), key=operator.itemgetter(1))
                getClass=self.findClass(distances_from_i_dict[:Kinput],Kinput)
                kArrayTrainingClasses=[];
                kArrayTrainingClasses_training = [];
                for classLoop in range(len(getClass)):
                    kArrayTrainingClasses.append(trainingSet.iloc[:, 0:1].iloc[getClass[classLoop]][0])
                finalClass.append(numpy.amax(kArrayTrainingClasses) );

            for trainingLoop in range(trainingSet.shape[0]):
                distances_from_i_dict_training = dict(enumerate(result_training[trainingLoop]))
                distances_from_i_dict_training = sorted(distances_from_i_dict_training.items(), key=operator.itemgetter(1))
                getClass_training = self.findClass(distances_from_i_dict_training[:Kinput], Kinput)
                kArrayTrainingClasses_training = [];
                for classLoop_training in range(len(getClass_training)):
                    kArrayTrainingClasses_training.append(trainingSet.iloc[:, 0:1].iloc[getClass_training[classLoop_training]][0])
                finalClass_training.append(numpy.amax(kArrayTrainingClasses_training));

            finalClass=pd.DataFrame(numpy.array(finalClass).reshape(1, 1000))
            data = Counter(numpy.sum(finalClass == testSet.iloc[:, :1].T, axis=0))
            accuracy=data.most_common(1)[0][1]/1000;
            testError=1-accuracy;
            kTest[Kinput]=testError;

            finalClass_training = pd.DataFrame(numpy.array(finalClass_training).reshape(1, 6000))
            data_training = Counter(numpy.sum(finalClass_training == trainingSet.iloc[:, :1].T, axis=0))
            accuracy_training = data_training.most_common(1)[0][1] / 6000;
            trainingError = 1 - accuracy_training;
            kTraining[Kinput] = trainingError;
        self.plot_learning_curve(kTest,kTraining)
            # self.findAccuracy(finalClass)



    def findClass(self,trainingSet,k):
        getInitialClass=[];
        i=0;
        while(i<k):
            getInitialClass.append(trainingSet[i][0]);
            i=i+1;
        return getInitialClass;

    def plot_learning_curve(self, plot_curve1_dict, plot_curve2_dict):
        x, y_error1 = zip(*sorted(plot_curve1_dict.items()))
        x, y_error2 = zip(*sorted(plot_curve2_dict.items()))
        fig = plt.figure()
        plt.title('Error vs Value of K')
        plt.plot(x, y_error1, 'b', label='Training Error')
        plt.plot(x, y_error2, 'r', label='Test Error')
        plt.ylabel('Error Value')
        plt.xlabel('K')
        plt.legend()
        plt.show()
        fig.savefig('KNN_mine_final.png')

        return None











if __name__ == "__main__":
    obj = KNN();
    # obj.input_data()
    # obj.convert('train-images.idx3-ubyte','train-labels.idx1-ubyte','Training.csv',6000);
    # obj.convert('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte', 'Testing.csv', 1000);
    obj.KNNAlgo(pd.read_csv('Testing.csv', sep=",", header=None),pd.read_csv('Training.csv', sep=",", header=None))



