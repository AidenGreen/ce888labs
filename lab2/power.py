from random import randint
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import numpy as np

def power(sample1, sample2, reps, size, alpha):
    times =0
    for i in range(reps):
        newSample1 = sampleGenerater(sample1,size)
        newSample2 = sampleGenerater(sample2,size)
        mean1 = np.mean(newSample1)
        mean2 = np.mean(newSample2)
        mean = mean2-mean1
        if mean < 1-alpha:
            times += 1
    result = times/reps
    print(result)


def sampleGenerater(sample,size):
    oldSample = sample
    newSample=[]
    for i in range(size):
        id = randint(0, 9)
        newSample.append(oldSample[id])

    return newSample

sample1=[0,0,0,0,0,0,1,0,0,1]
sample2=[0,1,1,1,0,1,1,0,0,1]
power(sample1,sample2,1000,10,0.2)