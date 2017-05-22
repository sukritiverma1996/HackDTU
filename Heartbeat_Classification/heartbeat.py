from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioFeatureExtraction as aF
import matplotlib.pyplot as plt
import numpy as np
import cPickle
import sys
import time
import os
import glob
import aifc
import math
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import utilities

def stFeatureExtraction(signal, Fs, Win, Step):
    Win = int(Win)
    Step = int(Step)

    # Signal normalization
    signal = np.double(signal)

    signal = signal / (2.0 ** 15)
    DC = signal.mean()
    MAX = (np.abs(signal)).max()
    signal = (signal - DC) / MAX

    N = len(signal)                                # total number of samples
    curPos = 0
    countFrames = 0
    nFFT = Win / 2

    [fbank, freqs] = aF.mfccInitFilterBanks(Fs, nFFT)                # compute the triangular filter banks used in the mfcc calculation
    nChroma, nFreqsPerChroma = aF.stChromaFeaturesInit(nFFT, Fs)

    numOfTimeSpectralFeatures = 8
    numOfHarmonicFeatures = 0
    nceps = 13
    numOfChromaFeatures = 13
    totalNumOfFeatures = numOfTimeSpectralFeatures + nceps + numOfHarmonicFeatures + numOfChromaFeatures
    #totalNumOfFeatures = numOfTimeSpectralFeatures + nceps + numOfHarmonicFeatures

    stFeatures = []
    while (curPos + Win - 1 < N):                        # for each short-term window until the end of signal
        countFrames += 1
        x = signal[curPos:curPos+Win]                    # get current window
        curPos = curPos + Step                           # update window position
        X = abs(aF.fft(x))                                  # get fft magnitude
        X = X[0:nFFT]                                    # normalize fft
        X = X / len(X)
        if countFrames == 1:
            Xprev = X.copy()                             # keep previous fft mag (used in spectral flux)
        curFV = np.zeros((totalNumOfFeatures, 1))
        curFV[0] = aF.stZCR(x)                              # zero crossing rate
        curFV[1] = aF.stEnergy(x)                           # short-term energy
        curFV[2] = aF.stEnergyEntropy(x)                    # short-term entropy of energy
        [curFV[3], curFV[4]] = aF.stSpectralCentroidAndSpread(X, Fs)    # spectral centroid and spread
        curFV[5] = aF.stSpectralEntropy(X)                  # spectral entropy
        curFV[6] = aF.stSpectralFlux(X, Xprev)              # spectral flux
        curFV[7] = aF.stSpectralRollOff(X, 0.90, Fs)        # spectral rolloff
        curFV[numOfTimeSpectralFeatures:numOfTimeSpectralFeatures+nceps, 0] = aF.stMFCC(X, fbank, nceps).copy()    # MFCCs

        chromaNames, chromaF = aF.stChromaFeatures(X, Fs, nChroma, nFreqsPerChroma)
        curFV[numOfTimeSpectralFeatures + nceps: numOfTimeSpectralFeatures + nceps + numOfChromaFeatures - 1] = chromaF
        curFV[numOfTimeSpectralFeatures + nceps + numOfChromaFeatures - 1] = chromaF.std()
        stFeatures.append(curFV)
        # delta features
        '''
        if countFrames>1:
            delta = curFV - prevFV
            curFVFinal = numpy.concatenate((curFV, delta))            
        else:
            curFVFinal = numpy.concatenate((curFV, curFV))
        prevFV = curFV
        stFeatures.append(curFVFinal)        
        '''
        # end of delta
        Xprev = X.copy()

    stFeatures = np.concatenate(stFeatures, 1)
    return stFeatures


def mtFeatureExtraction(signal, Fs, mtWin, mtStep, stWin, stStep):
    """
    Mid-term feature extraction
    """

    mtWinRatio = int(round(mtWin / stStep))
    mtStepRatio = int(round(mtStep / stStep))

    mtFeatures = []

    stFeatures = stFeatureExtraction(signal, Fs, stWin, stStep)
    numOfFeatures = len(stFeatures)
    numOfStatistics = 2

    mtFeatures = []
    #for i in range(numOfStatistics * numOfFeatures + 1):
    for i in range(numOfStatistics * numOfFeatures):
        mtFeatures.append([])

    for i in range(numOfFeatures):        # for each of the short-term features:
        curPos = 0
        N = len(stFeatures[i])
        while (curPos < N):
            N1 = curPos
            N2 = curPos + mtWinRatio
            if N2 > N:
                N2 = N
            curStFeatures = stFeatures[i][N1:N2]

            mtFeatures[i].append(np.mean(curStFeatures))
            mtFeatures[i+numOfFeatures].append(np.std(curStFeatures))
            #mtFeatures[i+2*numOfFeatures].append(numpy.std(curStFeatures) / (numpy.mean(curStFeatures)+0.00000010))
            curPos += mtStepRatio

    return np.array(mtFeatures), stFeatures
    
def dirWavFeatureExtraction(dirName, mtWin, mtStep, stWin, stStep, computeBEAT=False):
    allMtFeatures = np.array([])
    processingTimes = []

    types = ('*.wav', '*.aif',  '*.aiff', '*.mp3','*.au')
    wavFilesList = []
    for files in types:
        wavFilesList.extend(glob.glob(os.path.join(dirName, files)))

    wavFilesList = sorted(wavFilesList)    

    for i, wavFile in enumerate(wavFilesList):        
        print "Analyzing file {0:d} of {1:d}: {2:s}".format(i+1, len(wavFilesList), wavFile.encode('utf-8'))
        if os.stat(wavFile).st_size == 0:
            print "   (EMPTY FILE -- SKIPPING)"
            continue        
        [Fs, x] = audioBasicIO.readAudioFile(wavFile)            # read file                
        t1 = time.clock()        
        x = audioBasicIO.stereo2mono(x)                          # convert stereo to mono                
        if x.shape[0]<float(Fs)/10:
            print "  (AUDIO FILE TOO SMALL - SKIPPING)"
            continue
        if computeBEAT:                                          # mid-term feature extraction for current file
            [MidTermFeatures, stFeatures] = mtFeatureExtraction(x, Fs, round(mtWin * Fs), round(mtStep * Fs), round(Fs * stWin), round(Fs * stStep))
            [beat, beatConf] = aF.beatExtraction(stFeatures, stStep)
        else:
            [MidTermFeatures, _] = mtFeatureExtraction(x, Fs, round(mtWin * Fs), round(mtStep * Fs), round(Fs * stWin), round(Fs * stStep))

        MidTermFeatures = np.transpose(MidTermFeatures)
        MidTermFeatures = MidTermFeatures.mean(axis=0)         # long term averaging of mid-term statistics
        if computeBEAT:
            MidTermFeatures = np.append(MidTermFeatures, beat)
            MidTermFeatures = np.append(MidTermFeatures, beatConf)
        if len(allMtFeatures) == 0:                              # append feature vector
            allMtFeatures = MidTermFeatures
        else:
            allMtFeatures = np.vstack((allMtFeatures, MidTermFeatures))
        t2 = time.clock()
        duration = float(len(x)) / Fs
        processingTimes.append((t2 - t1) / duration)
    if len(processingTimes) > 0:
        print "Feature extraction complexity ratio: {0:.1f} x realtime".format((1.0 / np.mean(np.array(processingTimes))))
    return (allMtFeatures, wavFilesList)

def dirsWavFeatureExtraction(dirNames, mtWin, mtStep, stWin, stStep, computeBEAT=False):
    # feature extraction for each class:
    features = []
    classNames = []
    fileNames = []
    for i, d in enumerate(dirNames):
        [f, fn] = dirWavFeatureExtraction(d, mtWin, mtStep, stWin, stStep, computeBEAT=computeBEAT)
        if f.shape[0] > 0:       # if at least one audio file has been found in the provided folder:
            features.append(f)
            fileNames.append(fn)
            if d[-1] == "/":
                classNames.append(d.split(os.sep)[-2])
            else:
                classNames.append(d.split(os.sep)[-1])
    return features, classNames, fileNames

def featureAndTrain(listOfDirs, mtWin, mtStep, stWin, stStep, classifierType, modelName, computeBEAT=False, perTrain=0.90):
    # STEP A: Feature Extraction:
    [features, classNames, _] = dirsWavFeatureExtraction(listOfDirs, mtWin, mtStep, stWin, stStep, computeBEAT=computeBEAT)

    if len(features) == 0:
        print "trainSVM_feature ERROR: No data found in any input folder!"
        return

    numOfFeatures = features[0].shape[1]
    featureNames = ["features" + str(d + 1) for d in range(numOfFeatures)]

    aT.writeTrainDataToARFF(modelName, features, classNames, featureNames)

    for i, f in enumerate(features):
        if len(f) == 0:
            print "trainSVM_feature ERROR: " + listOfDirs[i] + " folder is empty or non-existing!"
            return

    # STEP B: Classifier Evaluation and Parameter Selection:
    if classifierType == "svm" or classifierType == "svm_rbf":
        classifierParams = np.array([0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0])
    elif classifierType == "randomforest":
        classifierParams = np.array([10, 25, 50, 100,200,500])
    elif classifierType == "knn":
        classifierParams = np.array([1, 3, 5, 7, 9, 11, 13, 15])        
    elif classifierType == "gradientboosting":
        classifierParams = np.array([10, 25, 50, 100,200,500])        
    elif classifierType == "extratrees":
        classifierParams = np.array([10, 25, 50, 100,200,500])        

    # get optimal classifeir parameter:
    bestParam, acc = aT.evaluateClassifier(features, classNames, 100, classifierType, classifierParams, 0, perTrain)

    print "Selected params: {0:.5f}".format(bestParam)

    C = len(classNames)
    [featuresNorm, MEAN, STD] = aT.normalizeFeatures(features)        # normalize features
    MEAN = MEAN.tolist()
    STD = STD.tolist()
    featuresNew = featuresNorm

    # STEP C: Save the classifier to file
    if classifierType == "svm":
        Classifier = aT.trainSVM(featuresNew, bestParam)        
    elif classifierType == "svm_rbf":
        Classifier = aT.trainSVM_RBF(featuresNew, bestParam)
    elif classifierType == "randomforest":
        Classifier = aT.trainRandomForest(featuresNew, bestParam)
    elif classifierType == "gradientboosting":
        Classifier = aT.trainGradientBoosting(featuresNew, bestParam)

    if classifierType == "knn":
        [X, Y] = aT.listOfFeatures2Matrix(featuresNew)
        X = X.tolist()
        Y = Y.tolist()
        fo = open(modelName, "wb")
        cPickle.dump(X, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(Y,  fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(MEAN, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(STD,  fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(classNames,  fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(bestParam,  fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(mtWin, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(mtStep, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(stWin, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(stStep, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(computeBEAT, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        fo.close()
    elif classifierType == "svm" or classifierType == "svm_rbf" or classifierType == "randomforest" or classifierType == "gradientboosting":
        with open(modelName, 'wb') as fid:                                            # save to file
            cPickle.dump(Classifier, fid)            
        fo = open(modelName + "MEANS", "wb")
        cPickle.dump(MEAN, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(STD, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(classNames, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(mtWin, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(mtStep, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(stWin, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(stStep, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(computeBEAT, fo, protocol=cPickle.HIGHEST_PROTOCOL)
        fo.close()
    return acc

acc1 = featureAndTrain(["artifact","extrahls","murmur","normal"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmMusicGenre3", True)
acc2 = featureAndTrain(["artifact","extrahls","murmur","normal"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knnMusicGenre3", True)
acc3 = featureAndTrain(["artifact","extrahls","murmur","normal"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "gradientboosting", "gbMusicGenre3", True)
acc4 = featureAndTrain(["artifact","extrahls","murmur","normal"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "randomforest", "rfMusicGenre3", True)
fig, axes = plt.subplots( nrows=1, ncols=1, figsize=(10,20) )
plt.subplots_adjust( wspace=0.20, hspace=0.20, top=0.97 )
plt.suptitle("Classifier Comparisons", fontsize=20)
y=[acc1,acc2,acc3,acc4]
bins     = np.arange(4)
width    = 0.5
axes.bar(bins+0.025,y,width,align="center",edgecolor=["crimson"],color=["crimson"],label="Accuracy")
axes.set_xlabel("Classifiers",fontsize=15)
axes.set_ylabel("Accuracy",fontsize=15)
axes.set_xticks(bins)
axes.set_xticklabels(["svm- "+str(y[0]), "knn- "+str(y[1]), "gradientboosting- "+str(y[2]),"randomforest- "+str(y[3])],
                          ha="right")

plt.show()
fig.savefig('result.png')