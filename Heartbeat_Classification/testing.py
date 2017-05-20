from pyAudioAnalysis import audioAnalysis as aA 
his = aA.classifyFolderWrapper("testdata", "svm", "svmMusicGenre3")
r1=[]
for i, r in enumerate(his):
    r1.append(r)
his = aA.classifyFolderWrapper("testdata", "knn", "knnMusicGenre3")
r2=[]
for i, r in enumerate(his):
    r2.append(r)
his = aA.classifyFolderWrapper("testdata", "gradientboosting", "gbMusicGenre3")
r3=[]
for i, r in enumerate(his):
    r3.append(r)
his = aA.classifyFolderWrapper("testdata", "randomforest", "rfMusicGenre3")
r4=[]
for i, r in enumerate(his):
    r4.append(r)
import matplotlib.pyplot as plt
import numpy as np
fig, axes = plt.subplots( nrows=1, ncols=1, figsize=(10,20) )
plt.subplots_adjust( wspace=0.20, hspace=0.20, top=0.97 )
plt.suptitle("Classifier Comparisons", fontsize=20)
bins     = np.arange(4)
width    = 0.1
rects1 = axes.bar(bins+0.025,r1,width,align="center",edgecolor=["crimson"],color=["crimson"],label="svm")
rects2 = axes.bar(bins+0.125,r2,width,align="center",edgecolor=["green"],color=["green"],label="knn")
rects3 = axes.bar(bins+0.225,r3,width,align="center",edgecolor=["blue"],color=["blue"],label="gradientboosting")
rects4 = axes.bar(bins+0.325,r4,width,align="center",edgecolor=["yellow"],color=["yellow"],label="randomforest")
axes.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('svm', 'knn', 'gradientboosting','randomforest') )
axes.set_xlabel("Classes",fontsize=15)
axes.set_ylabel("No. of instances",fontsize=15)
axes.set_xticks(bins)
axes.set_xticklabels(["artifact- "+str(r1[0])+", "+str(r2[0])+", "+str(r3[0])+", "+str(r4[0]), "extrahls- "+str(r1[1])+", "+str(r2[1])+", "+str(r3[1])+", "+str(r4[1]), "murmur- "+str(r1[2])+", "+str(r2[2])+", "+str(r3[2])+", "+str(r4[2]),"normal- "+str(r1[3])+", "+str(r2[3])+", "+str(r3[3])+", "+str(r4[3])], ha="center")
plt.show()
fig.savefig('testres.png')