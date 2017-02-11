from pyAudioAnalysis import audioTrainTest as aT

#aT.featureAndTrain(["artifact","extrahls","murmur","normal"], 1.0, 1.0, aT.shortTermWindow, aT.shortTermStep, "svm", "svmMusicGenre3", True)

from pyAudioAnalysis import audioAnalysis as aA 
aA.classifyFolderWrapper("set_a", "svm", "svmMusicGenre3") 
