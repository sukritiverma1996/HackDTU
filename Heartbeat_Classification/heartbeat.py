from pyAudioAnalysis import audioTrainTest as aT
import matplotlib.pyplot as plt
import numpy as np

acc1 = aT.featureAndTrain(["artifact","extrahls","murmur","normal"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmMusicGenre3", True)
acc2 = aT.featureAndTrain(["artifact","extrahls","murmur","normal"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "knn", "knnMusicGenre3", True)
acc3 = aT.featureAndTrain(["artifact","extrahls","murmur","normal"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "gradientboosting", "gbMusicGenre3", True)
acc4 = aT.featureAndTrain(["artifact","extrahls","murmur","normal"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "randomforest", "rfMusicGenre3", True)
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