from __future__ import division  # Python 2 users only
from collections import defaultdict, Counter
import os, sys, string
import copy
import numpy as np
import nltk, re, pprint
from collections import OrderedDict
from sklearn.naive_bayes import MultinomialNB
from nltk import wordnet, FreqDist
from os import listdir
from os.path import isfile, join
from nltk.stem.porter import *
from nltk.corpus import stopwords

##### GLOBAL VARIABLES ######
vocabularyGlobal = OrderedDict()
vocabularyGlobal['UNKNOWN'] = 0
vocabularyGlobal['love'] = 0
vocabularyGlobal['wonderful'] = 0
vocabularyGlobal['best'] = 0
vocabularyGlobal['great'] = 0
vocabularyGlobal['superb'] = 0
vocabularyGlobal['still'] = 0
vocabularyGlobal['beautiful'] = 0
vocabularyGlobal['bad'] = 0
vocabularyGlobal['worst'] = 0
vocabularyGlobal['stupid'] = 0
vocabularyGlobal['waste'] = 0
vocabularyGlobal['boring'] = 0
vocabularyGlobal['?'] = 0
vocabularyGlobal['!'] = 0


set = {'loves', 'loving', 'loved'}
##### FUNCTIONS ######
def transferDirectory(fileDj, vocabulary):
    BOWDj = []
    set = {'loves', 'loving', 'loved'}
    for subdir, dirs, files in os.walk(fileDj):
        for file in files:
            if file.endswith(".txt"):
                dict = copy.deepcopy(vocabulary)
                f = open(os.path.join(subdir, file),'r')
                a = f.read()
                #print(a)
                for word in a.split():
                    if word in dict:
                        dict[word] = dict[word] + 1
                    elif word in set:
                        dict['love'] += 1
                    else:
                        dict['UNKNOWN'] += 1
                f.close()
            BOWDj.append(dict.values())

    return BOWDj

def transfer(fileDj, vocabulary):
    # parameter:
    # @fileDj: absolute directory
    # @vocabulary: dictionary data structure
    #print(fileDj)
    set = {'loves', 'loving', 'loved'}
    with open(os.path.join(fileDj), 'r') as f:
        a = f.read()
        dict = copy.deepcopy(vocabulary)
        for word in a.split():
            if word in dict :
                #print("transferkey", list(vocabularyGlobal.keys()).index(word))
                dict[word] = dict[word] + 1
            elif word in set:
                dict['love'] += 1
            else:
                dict['UNKNOWN'] += 1
    return np.fromiter(iter(dict.values()), dtype=int)


def loadData(Path):
    posPath = Path + "training_set/pos/"
    onlyfiles = [f for f in listdir(posPath) if isfile(join(posPath, f))]
    pos = []
    totalPosFile = 0
    totalNegFile = 0
    for filename in onlyfiles:
        totalPosFile += 1
        posarray = transfer(posPath + filename, vocabularyGlobal)
        pos.append(posarray)
    pos = np.array(pos) #shape:(700, 15)
    posy = np.full(totalPosFile, 1)
    posy = np.asmatrix(posy).T
    negPath = Path + "training_set/neg/"
    onlyfiles = [f for f in listdir(negPath) if isfile(join(negPath, f))]
    neg = []
    for filename in onlyfiles:
        totalNegFile += 1
        negarray = transfer(negPath + filename, vocabularyGlobal)
        neg.append(negarray)
    neg = np.array(neg)  # shape:(700, 15)
    negy = np.full(totalNegFile, 0)
    negy = np.asmatrix(negy).T

    Xtrain = np.concatenate((pos,neg), axis = 0)
    ytrain = np.concatenate((posy,negy), axis = 0)


    #load test data, might get error if subdir goes a diff name! #
    posTestPath = Path + "test_set/pos/"
    onlyfiles = [f for f in listdir(posTestPath) if isfile(join(posTestPath, f))]
    posTest = []
    totalPosFile = 0
    totalNegFile = 0
    for filename in onlyfiles:
        totalPosFile += 1
        posTestarray = transfer(posTestPath + filename, vocabularyGlobal)
        posTest.append(posTestarray)
    posTest = np.array(posTest)  # shape:(700, 15)
    posyTest = np.full(totalPosFile, 1)
    posyTest = np.asmatrix(posyTest).T
    negTestPath = Path + "test_set/neg/"
    onlyfiles = [f for f in listdir(negTestPath) if isfile(join(negTestPath, f))]
    negTest = []
    for filename in onlyfiles:
        totalNegFile += 1
        negTestarray = transfer(negTestPath + filename, vocabularyGlobal)
        negTest.append(negTestarray)
    negTest = np.array(negTest)  # shape:(700, 15)
    negyTest = np.full(totalNegFile, 0)
    negyTest = np.asmatrix(negyTest).T

    Xtest = np.concatenate((posTest, negTest), axis=0)
    ytest = np.concatenate((posyTest, negyTest), axis=0)


    return Xtrain, Xtest, ytrain, ytest



def naiveBayesMulFeature_train(Xtrain, ytrain):

    XtrainUpperHalf = Xtrain[:700,:]
    VocabAbs = len(vocabularyGlobal)
    totalWords = XtrainUpperHalf.sum() #total words in positive class
    sumVertical = np.sum(XtrainUpperHalf, axis = 0) # each words occurance
    #print(sumVertical)
    thetaPos = []
    for i in range(len(vocabularyGlobal)):
        thetaPos.append((sumVertical.item(i) + 1) / (totalWords + len(vocabularyGlobal)))

    XtrainLowerHalf = Xtrain[700:1400, :]
    probabilityOverClassifier = 0.5  # We are not gonna use it
    totalWords2 = XtrainLowerHalf.sum()  # total words in positive class
    sumVertical2 = np.sum(XtrainLowerHalf, axis=0)  # each words occurance
    # print(sumVertical)
    thetaNeg = []
    for i in range(len(vocabularyGlobal)):
        thetaNeg.append((sumVertical2.item(i) + 1) / (totalWords2 + len(vocabularyGlobal)))
    return np.array(thetaPos), np.array(thetaNeg)

def naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg):
    XtestUpperHalf = Xtest[:600, :]
    # for each row, calculate sum(n(wj)log{(theta(wj)|c)}
    predict = []
    for i in range(600):
        toTest = XtestUpperHalf[i:i+1,:]
        PrPos = 0
        PrNeg = 0
        for j in range(len(vocabularyGlobal)):
            PrPos = PrPos + toTest.item(j) * np.log10(thetaPos.item(j))
            PrNeg = PrNeg + toTest.item(j) * np.log10(thetaNeg.item(j))

        if PrPos > PrNeg:
            predict.append(1)
        else:
            predict.append(0)
    predictarray = np.matrix(predict)

    shouldbepos = predictarray[:,0:300]
    #print(shouldbepos.shape)
    accuracyUpperHalf = 0

    for i in range(300):
        if shouldbepos.item(i) == 1:
            accuracyUpperHalf += 1/600

    shouldbeneg = predictarray[:,300:600]
    #print(shouldbeneg.shape)
    accuracyLowerHalf = 0
    totalZero = 0
    for i in range(300):
        if shouldbeneg.item(i) == 0:

            accuracyLowerHalf += 1 / 600
    Accuracy = accuracyLowerHalf + accuracyUpperHalf
    yPredict = predict
    return yPredict, Accuracy

def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
    clf = MultinomialNB(alpha=1.0)
    clf.fit(Xtrain, ytrain)
    predict = clf.predict(Xtest)
    a = clf.score(Xtest,ytest)
    predictarray = np.matrix(predict)
    #print(predictarray.shape)
    shouldbepos = predictarray[:,0:300]
    accuracyUpperHalf = 0
    for i in range(300):
        if shouldbepos.item(i) == 1:
            accuracyUpperHalf += 1 / 600

    shouldbeneg = predictarray[:,300:600]
    accuracyLowerHalf = 0
    for i in range(300):
        if shouldbeneg.item(i) == 0:
            accuracyLowerHalf += 1 / 600
    Accuracy = accuracyLowerHalf + accuracyUpperHalf
    return Accuracy

def naiveBayesMulFeature_testDirectOne(path, thetaPos, thetaNeg):
    PrPos = 0
    PrNeg = 0
    predict = 0
    with open(os.path.join(path), 'r') as f:
        a = f.read()
        for word in a.split():
            if word in vocabularyGlobal:
                PrPos = PrPos + np.log10(thetaPos.item(list(vocabularyGlobal.keys()).index(word)))
                PrNeg = PrNeg + np.log10(thetaNeg.item(list(vocabularyGlobal.keys()).index(word)))
            elif word in set:
                PrPos = PrPos + np.log10(thetaPos.item(list(vocabularyGlobal.keys()).index('love')))
                PrNeg = PrNeg + np.log10(thetaNeg.item(list(vocabularyGlobal.keys()).index('love')))
            else:
                PrPos = PrPos + np.log10(thetaPos.item(list(vocabularyGlobal.keys()).index('UNKNOWN')))
                PrNeg = PrNeg + np.log10(thetaNeg.item(list(vocabularyGlobal.keys()).index('UNKNOWN')))

    if PrPos > PrNeg:
        predict = 1
    return predict

def naiveBayesMulFeature_testDirect(XtestTextFileNameInFullPathOne, thetaPos, thetaNeg):
    predict = []
    accuracy = 0
    posDirectory = XtestTextFileNameInFullPathOne + 'pos/'
    onlyfiles = [f for f in listdir(posDirectory) if isfile(join(posDirectory, f))]

    for filename in onlyfiles:
        PrPos = 0
        PrNeg = 0
        with open(os.path.join(posDirectory + filename), 'r') as f:
            a = f.read()
            for word in a.split():
                if word in vocabularyGlobal:
                    PrPos = PrPos + np.log10(thetaPos.item(list(vocabularyGlobal.keys()).index(word)))
                    PrNeg = PrNeg + np.log10(thetaNeg.item(list(vocabularyGlobal.keys()).index(word)))
                elif word in set:
                    PrPos = PrPos + np.log10(thetaPos.item(list(vocabularyGlobal.keys()).index('love')))
                    PrNeg = PrNeg + np.log10(thetaNeg.item(list(vocabularyGlobal.keys()).index('love')))
                else:
                    PrPos = PrPos + np.log10(thetaPos.item(list(vocabularyGlobal.keys()).index('UNKNOWN')))
                    PrNeg = PrNeg + np.log10(thetaNeg.item(list(vocabularyGlobal.keys()).index('UNKNOWN')))

        if PrPos > PrNeg:
            predict.append(1)
            accuracy = accuracy + 1/600
        else:
            predict.append(0)

    negDirectory = XtestTextFileNameInFullPathOne + 'neg/'
    onlyfiles = [f for f in listdir(negDirectory) if isfile(join(negDirectory, f))]
    for filename in onlyfiles:
        PrPos = 0
        PrNeg = 0
        with open(os.path.join(negDirectory + filename), 'r') as f:
            a = f.read()
            for word in a.split():
                if word in vocabularyGlobal:
                    PrPos = PrPos + np.log10(thetaPos.item(list(vocabularyGlobal.keys()).index(word)))
                    PrNeg = PrNeg + np.log10(thetaNeg.item(list(vocabularyGlobal.keys()).index(word)))
                elif word in set:
                    PrPos = PrPos + np.log10(thetaPos.item(list(vocabularyGlobal.keys()).index('love')))
                    PrNeg = PrNeg + np.log10(thetaNeg.item(list(vocabularyGlobal.keys()).index('love')))
                else:
                    PrPos = PrPos + np.log10(thetaPos.item(list(vocabularyGlobal.keys()).index('UNKNOWN')))
                    PrNeg = PrNeg + np.log10(thetaNeg.item(list(vocabularyGlobal.keys()).index('UNKNOWN')))
        if PrPos > PrNeg:
            predict.append(1)

        else:
            predict.append(0)
            accuracy = accuracy + 1 / 600
    return predict, accuracy

def naiveBayesBernFeature_train(Xtrain, ytrain):
    thetaPos = []
    thetaNeg = []
    for x in np.nditer(Xtrain, op_flags=['readwrite']):
        if x > 0:
            x[...] = 1
    upper = Xtrain[:700,:]

    upperTotal = np.sum(upper, axis=0)
    for x in np.nditer(upperTotal, op_flags=['readwrite']):
        anspos = (x + 1) / 702
        thetaPos.append(anspos)

    lower = Xtrain[700:1400, :]
    lowerTotal = np.sum(lower, axis=0)

    for x in np.nditer(lowerTotal, op_flags=['readwrite']):
        ansneg = (x + 1) / 702
        thetaNeg.append(ansneg)
    return np.array(thetaPos), np.array(thetaNeg)

def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
    XtestUpperHalf = Xtest[:600, :]
    # for each row, calculate sum(n(wj)log{(theta(wj)|c)}
    #print("thetaPosTrue", thetaPosTrue)
    predict = []
    for i in range(600):
        toTest = XtestUpperHalf[i:i + 1, :]
        PrPos = 0
        PrNeg = 0
        for j in range(len(vocabularyGlobal)):
            if toTest.item(j) > 0:
                PrPos = PrPos + np.log10(thetaPosTrue.item(j))
                PrNeg = PrNeg + np.log10(thetaNegTrue.item(j))
            else:
                PrPos = PrPos + np.log10(1 - thetaPosTrue.item(j))
                PrNeg = PrNeg + np.log10(1 - thetaNegTrue.item(j))

        #print(PrPos)
        if PrPos > PrNeg:
            predict.append(1)
        else:
            predict.append(0)
    predictarray = np.matrix(predict)
    # print(predictarray.shape)
    shouldbepos = predictarray[:, 0:300]
    # print(shouldbepos.shape)
    accuracyUpperHalf = 0
    for i in range(300):
        if shouldbepos.item(i) == 1:
            accuracyUpperHalf += 1 / 600

    shouldbeneg = predictarray[:, 300:600]
    # print(shouldbeneg.shape)
    accuracyLowerHalf = 0
    for i in range(300):
        if shouldbeneg.item(i) == 0:
            accuracyLowerHalf += 1 / 600
    Accuracy = accuracyLowerHalf + accuracyUpperHalf
    yPredict = predict
    return yPredict, Accuracy

def NLTKstem(path):
    posPath = path + "training_set/pos/"
    onlyfiles = [f for f in listdir(posPath) if isfile(join(posPath, f))]
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    mergedlist = []
    for filename in onlyfiles:
        with open(posPath + filename, 'r') as shakes:
            text = shakes.read()
            lowers = text.lower()
            no_punctuation = lowers.translate(remove_punctuation_map)
            tokens = nltk.word_tokenize(no_punctuation)
            filtered = [w for w in tokens if not w in stopwords.words('english')]
            mergedlist += filtered
            count = Counter(filtered)
            #print(count.most_common(10))
    stemmed = []
    stemmer = PorterStemmer()
    for item in mergedlist:
        stemmed.append(stemmer.stem(item))

    # also put neg files in the stemmed:
    negPath = path + "training_set/neg/"
    onlyfiles = [f for f in listdir(negPath) if isfile(join(negPath, f))]
    mergedlistneg = []
    for filename in onlyfiles:
        with open(negPath + filename, 'r') as shakes:
            text = shakes.read()
            lowers = text.lower()
            no_punctuation = lowers.translate(remove_punctuation_map)
            tokens = nltk.word_tokenize(no_punctuation)
            filtered = [w for w in tokens if not w in stopwords.words('english')]
            mergedlistneg += filtered
    for item in mergedlistneg:
        stemmed.append(stemmer.stem(item))
    count = Counter(stemmed)
    l = range(250)
    mostcommon = np.matrix(count.most_common(250))
    mostcommon = mostcommon[:,0:1]
    #print(mostcommon.item(3))

    orderedDict = OrderedDict()
    for i in range(250):
        orderedDict[mostcommon.item(i)] = 0
    return orderedDict


def transferNLTK(fileDj, nltkdict):
    BOWDj = []
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    with open(os.path.join(fileDj), 'r') as f:
        dictorder = copy.deepcopy(nltkdict)
        a = f.read()
        lowers = a.lower()
        no_punctuation = lowers.translate(remove_punctuation_map)
        tokens = nltk.word_tokenize(no_punctuation)
        filtered = [w for w in tokens if not w in stopwords.words('english')]
        for word in filtered:

            if word in dictorder:
                dictorder[word] = dictorder[word] + 1

        f.close()
    BOWDj.append(dictorder.values())
    #print(np.fromiter(iter(dictorder.values()), dtype=int))

    return np.fromiter(iter(dictorder.values()), dtype=int)

def loadDataNLTK(Path, nltkdict):
    posPath = Path + "training_set/pos/"
    onlyfiles = [f for f in listdir(posPath) if isfile(join(posPath, f))]
    pos = []
    totalPosFile = 0
    totalNegFile = 0
    for filename in onlyfiles:
        totalPosFile += 1
        posarray = transferNLTK(posPath + filename, nltkdict)
        pos.append(posarray)
    pos = np.array(pos)  # shape:(700, 15)
    posy = np.full(totalPosFile, 1)
    posy = np.asmatrix(posy).T
    negPath = Path + "training_set/neg/"
    onlyfiles = [f for f in listdir(negPath) if isfile(join(negPath, f))]
    neg = []
    for filename in onlyfiles:
        totalNegFile += 1
        negarray = transferNLTK(negPath + filename, nltkdict)
        neg.append(negarray)
    neg = np.array(neg)  # shape:(700, 15)
    negy = np.full(totalNegFile, 0)
    negy = np.asmatrix(negy).T

    Xtrain = np.concatenate((pos, neg), axis=0)
    ytrain = np.concatenate((posy, negy), axis=0)

    # load test data, might get error if subdir goes a diff name! #
    posTestPath = Path + "test_set/pos/"
    onlyfiles = [f for f in listdir(posTestPath) if isfile(join(posTestPath, f))]
    posTest = []
    totalPosFile = 0
    totalNegFile = 0
    for filename in onlyfiles:
        totalPosFile += 1
        posTestarray = transferNLTK(posTestPath + filename, nltkdict)
        posTest.append(posTestarray)
    posTest = np.array(posTest)  # shape:(700, 15)
    posyTest = np.full(totalPosFile, 1)
    posyTest = np.asmatrix(posyTest).T
    negTestPath = Path + "test_set/neg/"
    onlyfiles = [f for f in listdir(negTestPath) if isfile(join(negTestPath, f))]
    negTest = []
    for filename in onlyfiles:
        totalNegFile += 1
        negTestarray = transferNLTK(negTestPath + filename, nltkdict)
        negTest.append(negTestarray)
    negTest = np.array(negTest)  # shape:(700, 15)
    negyTest = np.full(totalNegFile, 0)
    negyTest = np.asmatrix(negyTest).T

    Xtest = np.concatenate((posTest, negTest), axis=0)
    ytest = np.concatenate((posyTest, negyTest), axis=0)

    return Xtrain, Xtest, ytrain, ytest

def naiveBayesMulFeature_trainNLTK(Xtrain, ytrain, nltkDict):

    XtrainUpperHalf = Xtrain[:700,:]
    VocabAbs = len(nltkDict)
    totalWords = XtrainUpperHalf.sum() #total words in positive class
    sumVertical = np.sum(XtrainUpperHalf, axis = 0) # each words occurance
    #print(sumVertical)
    thetaPos = []
    for i in range(len(nltkDict)):
        thetaPos.append((sumVertical.item(i) + 1) / (totalWords + len(nltkDict)))

    XtrainLowerHalf = Xtrain[700:1400, :]
    probabilityOverClassifier = 0.5  # We are not gonna use it
    totalWords2 = XtrainLowerHalf.sum()  # total words in positive class
    sumVertical2 = np.sum(XtrainLowerHalf, axis=0)  # each words occurance
    # print(sumVertical)
    thetaNeg = []
    for i in range(len(nltkDict)):
        thetaNeg.append((sumVertical2.item(i) + 1) / (totalWords2 + len(nltkDict)))
    return np.array(thetaPos), np.array(thetaNeg)

def naiveBayesMulFeature_testNLTK(Xtest, ytest, thetaPos, thetaNeg, nltkDict):
    XtestUpperHalf = Xtest[:600, :]
    # for each row, calculate sum(n(wj)log{(theta(wj)|c)}
    predict = []
    for i in range(600):
        toTest = XtestUpperHalf[i:i+1,:]
        PrPos = 0
        PrNeg = 0
        for j in range(len(nltkDict)):
            PrPos = PrPos + toTest.item(j) * np.log10(thetaPos.item(j))
            PrNeg = PrNeg + toTest.item(j) * np.log10(thetaNeg.item(j))

        if PrPos > PrNeg:
            predict.append(1)
        else:
            predict.append(0)
    predictarray = np.matrix(predict)

    shouldbepos = predictarray[:,0:300]
    #print(shouldbepos.shape)
    accuracyUpperHalf = 0

    for i in range(300):
        if shouldbepos.item(i) == 1:
            accuracyUpperHalf += 1/600

    shouldbeneg = predictarray[:,300:600]
    #print(shouldbeneg.shape)
    accuracyLowerHalf = 0
    totalZero = 0
    for i in range(300):
        if shouldbeneg.item(i) == 0:

            accuracyLowerHalf += 1 / 600
    Accuracy = accuracyLowerHalf + accuracyUpperHalf
    yPredict = predict
    return yPredict, Accuracy


def naiveBayesMulFeature_testDirectNLTK(XtestTextFileNameInFullPathOne, thetaPos, thetaNeg, nltkDict):
    predict = []
    accuracy = 0
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    posDirectory = XtestTextFileNameInFullPathOne + 'pos/'
    onlyfiles = [f for f in listdir(posDirectory) if isfile(join(posDirectory, f))]

    for filename in onlyfiles:
        PrPos = 0
        PrNeg = 0
        with open(os.path.join(posDirectory + filename), 'r') as f:
            dictorder = copy.deepcopy(nltkDict)
            a = f.read()
            lowers = a.lower()
            no_punctuation = lowers.translate(remove_punctuation_map)
            tokens = nltk.word_tokenize(no_punctuation)
            filtered = [w for w in tokens if not w in stopwords.words('english')]
            for word in filtered:
                if word in nltkDict:
                    PrPos = PrPos + np.log10(thetaPos.item(list(nltkDict.keys()).index(word)))
                    PrNeg = PrNeg + np.log10(thetaNeg.item(list(nltkDict.keys()).index(word)))


        if PrPos > PrNeg:
            predict.append(1)
            accuracy = accuracy + 1/600
        else:
            predict.append(0)

    negDirectory = XtestTextFileNameInFullPathOne + 'neg/'
    onlyfiles = [f for f in listdir(negDirectory) if isfile(join(negDirectory, f))]
    for filename in onlyfiles:
        PrPos = 0
        PrNeg = 0
        with open(os.path.join(negDirectory + filename), 'r') as f:
            dictorder = copy.deepcopy(nltkDict)
            a = f.read()
            lowers = a.lower()
            no_punctuation = lowers.translate(remove_punctuation_map)
            tokens = nltk.word_tokenize(no_punctuation)
            filtered = [w for w in tokens if not w in stopwords.words('english')]
            for word in filtered:
                if word in nltkDict:
                    PrPos = PrPos + np.log10(thetaPos.item(list(nltkDict.keys()).index(word)))
                    PrNeg = PrNeg + np.log10(thetaNeg.item(list(nltkDict.keys()).index(word)))

        if PrPos > PrNeg:
            predict.append(1)

        else:
            predict.append(0)
            accuracy = accuracy + 1 / 600
    return predict, accuracy



def naiveBayesBernFeature_testNLTK(Xtest, ytest, thetaPosTrue, thetaNegTrue, nltkDict):
    XtestUpperHalf = Xtest[:600, :]
    # for each row, calculate sum(n(wj)log{(theta(wj)|c)}
    #print("thetaPosTrue", thetaPosTrue)
    predict = []
    for i in range(600):
        toTest = XtestUpperHalf[i:i + 1, :]
        PrPos = 0
        PrNeg = 0
        for j in range(len(nltkDict)):
            if toTest.item(j) > 0:
                PrPos = PrPos + np.log10(thetaPosTrue.item(j))
                PrNeg = PrNeg + np.log10(thetaNegTrue.item(j))
            else:
                PrPos = PrPos + np.log10(1 - thetaPosTrue.item(j))
                PrNeg = PrNeg + np.log10(1 - thetaNegTrue.item(j))

        #print(PrPos)
        if PrPos > PrNeg:
            predict.append(1)
        else:
            predict.append(0)
    predictarray = np.matrix(predict)
    # print(predictarray.shape)
    shouldbepos = predictarray[:, 0:300]
    # print(shouldbepos.shape)
    accuracyUpperHalf = 0
    for i in range(300):
        if shouldbepos.item(i) == 1:
            accuracyUpperHalf += 1 / 600

    shouldbeneg = predictarray[:, 300:600]
    # print(shouldbeneg.shape)
    accuracyLowerHalf = 0
    for i in range(300):
        if shouldbeneg.item(i) == 0:
            accuracyLowerHalf += 1 / 600
    Accuracy = accuracyLowerHalf + accuracyUpperHalf
    yPredict = predict
    return yPredict, Accuracy


##################################
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python naiveBayes.py dataSetPath testSetPath")
        sys.exit()

    print("--------------------")

    textDataSetsDirectoryFullPath = sys.argv[1]
    testFileDirectoryFullPath = sys.argv[2]

    Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)

    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    print("thetaPos =", thetaPos)
    print("thetaNeg =", thetaNeg)
    print("--------------------")

    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print("MNBC classification accuracy =", Accuracy)

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print("Sklearn MultinomialNB accuracy =", Accuracy_sk)

    yPredict, Accuracy = naiveBayesMulFeature_testDirect(testFileDirectoryFullPath, thetaPos, thetaNeg)
    print("Directly MNBC tesing accuracy =", Accuracy)
    print("--------------------")
    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print("thetaPosTrue =", thetaPosTrue)
    print("thetaNegTrue =", thetaNegTrue)
    print("--------------------")

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print("BNBC classification accuracy =", Accuracy)
    print("--------------------")

    ordereddict = NLTKstem(textDataSetsDirectoryFullPath)
    Xtrain, Xtest, ytrain, ytest = loadDataNLTK(textDataSetsDirectoryFullPath, ordereddict)
    thetaPos, thetaNeg = naiveBayesMulFeature_trainNLTK(Xtrain, ytrain, ordereddict)
    yPredict, Accuracy = naiveBayesMulFeature_testNLTK(Xtest, ytest, thetaPos, thetaNeg, ordereddict)
    print("MNBC accuracy STEMMING WHEN CHOOSING TOP 250 WORDs=", Accuracy)

    yPredict, Accuracy = naiveBayesMulFeature_testDirectNLTK(testFileDirectoryFullPath, thetaPos, thetaNeg, ordereddict)
    print("Directly MNBC tesing accuracy extra credit=", Accuracy)
    print("--------------------")

    yPredict, Accuracy = naiveBayesBernFeature_testNLTK(Xtest, ytest, thetaPosTrue, thetaNegTrue, ordereddict)
    print("BNBC classification accuracy extra credit=", Accuracy)
    print("--------------------")