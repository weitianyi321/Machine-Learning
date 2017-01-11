import numpy as np
import sys, glob, os, gzip, multiprocessing
from sklearn import tree
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA

def decision_tree(train, test):
    f = gzip.open(train, 'rb')
    file_content = f.read()
    filearray = np.fromstring(file_content, dtype=np.float, sep=' ')
    reshaped = filearray.reshape(7291, 257)
    y = reshaped[:, 0:1]
    X = reshaped[:, 1:]


    ######## ADJUST PARAMETERS HERE ##########
    clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=100, min_samples_split=10,
                                      min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                      random_state=None, max_leaf_nodes=100, min_impurity_split=1e-07,
                                      class_weight=None, presort=False)
    clf.fit(X, y)
    f = gzip.open(test, 'rb')
    file_content = f.read()
    filearray2 = np.fromstring(file_content, dtype=np.float, sep=' ')
    reshapedtest = filearray2.reshape(2007, 257)
    ytest = reshapedtest[:, 0:1]
    Xtest = reshapedtest[:, 1:]
    predicted = clf.predict(Xtest)
    totalCorrect = 0
    total = 0
    for i in range(ytest.size):

        if (predicted.item(i)) == ytest.item(i):
            totalCorrect += 1
        total += 1
    print('DecisionTre Test Error: ',1- totalCorrect / total)
    return ytest

def knn(train, test):
    f = gzip.open(train, 'rb')
    file_content = f.read()
    filearray = np.fromstring(file_content, dtype=np.float, sep=' ')
    reshaped = filearray.reshape(7291, 257)
    y = reshaped[:, 0:1]
    X = reshaped[:, 1:]
    ######## ADJUST PARAMETERS HERE ##########
    for i in range(1,6):
        clf = neighbors.KNeighborsClassifier(n_neighbors=i+1)
        clf.fit(X, y)
        f = gzip.open(test, 'rb')
        file_content = f.read()
        filearray2 = np.fromstring(file_content, dtype=np.float, sep=' ')
        reshapedtest = filearray2.reshape(2007, 257)
        ytest = reshapedtest[:, 0:1]
        Xtest = reshapedtest[:, 1:]
        predicted = clf.predict(Xtest)
        totalCorrect = 0
        total = 0
        for j in range(ytest.size):

            if (predicted.item(j)) == ytest.item(j):
                totalCorrect += 1
            total += 1
        print('KNN Test Error of nearest neighbor: ',i+1,' ', 1-totalCorrect / total)
    cv(X, y)
    return ytest

def cv(X, y):
    for i in range(0,1):
        kFoldTest = []
        kFoldTrain = []
        yFoldTest = []
        yFoldTrain = []
        k = 200 * i
        kFoldTest.append(X[k:k+1100,0:])
        yFoldTest.append(y[k:k+1100,:])
        if k == 0:
            kFoldTrain.append(X[1100:2000,0:])
            yFoldTrain.append(y[1100:2000,:])


        # print(kFoldTrain)
        yFoldTrain1 = []
        kFoldTrain1 = []
        kFoldTest1 = []
        yFoldTest1 = []
        for i in yFoldTrain:
            yFoldTrain1 = i
        for j in kFoldTrain:
            kFoldTrain1 = j
        for j in kFoldTest:
            kFoldTest1 = j
        for j in yFoldTest:
            yFoldTest1 = j
        for j in range(1,6):
            clf = neighbors.KNeighborsClassifier(n_neighbors=(j+1))
            clf.fit(kFoldTrain1, yFoldTrain1)
            predicted = clf.predict(kFoldTest1)
            error = errorHelper(predicted, yFoldTest)
            print('2-Fold Validation KNN Training Error : ',(j+1),1-error)
def errorHelper(predicted, actual):

    error = 0
    total = len(predicted)
    for i in range(len(predicted)):

        if predicted[i] == actual[0].item(i):
            error += 1/total
    return error

def svm(train, test):
    f = gzip.open(train, 'rb')
    file_content = f.read()
    filearray = np.fromstring(file_content, dtype=np.float, sep=' ')
    reshaped = filearray.reshape(7291, 257)
    y = reshaped[:, 0:1]
    X = reshaped[:, 1:]
    f = gzip.open(test, 'rb')
    file_content = f.read()
    filearray2 = np.fromstring(file_content, dtype=np.float, sep=' ')
    reshapedtest = filearray2.reshape(2007, 257)
    ytest = reshapedtest[:, 0:1]
    Xtest = reshapedtest[:, 1:]
    ######## ADJUST PARAMETERS HERE ##########

    learnedmodel = SVC(kernel='linear')

    learnedmodel.fit(X, y)
    pool = multiprocessing.Pool()
    predictresult = pool.map(learnedmodel.predict, Xtest)
    # predictresult = learnedmodel.predict(kFoldTest[j])
    predictresult = np.asmatrix(predictresult)



    totalCorrect = 0
    total = 0
    for i in range(ytest.size):

        if (predictresult.item(i)) == ytest.item(i):
            totalCorrect += 1
        total += 1
    print('SVM Testing Error', 1 - totalCorrect / total)
    return ytest

def pca_knn(train, test):
    f = gzip.open(train, 'rb')
    file_content = f.read()
    filearray = np.fromstring(file_content, dtype=np.float, sep=' ')
    reshaped = filearray.reshape(7291, 257)
    y = reshaped[:, 0:1]
    X = reshaped[:, 1:]
    pca = RandomizedPCA(n_components=160)
    X = pca.fit_transform(X)
    #print(X)

    ######## ADJUST PARAMETERS HERE ##########
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X, y)
    f = gzip.open(test, 'rb')
    file_content = f.read()
    filearray2 = np.fromstring(file_content, dtype=np.float, sep=' ')
    reshapedtest = filearray2.reshape(2007, 257)
    ytest = reshapedtest[:, 0:1]
    Xtest = reshapedtest[:, 1:]
    Xtest = pca.fit_transform(Xtest)
    predicted = clf.predict(Xtest)
    totalCorrect = 0
    total = 0
    for i in range(ytest.size):

        if (predicted.item(i)) == ytest.item(i):
            totalCorrect += 1
        total += 1
    print('Pca knnaccuracy', 1-totalCorrect / total)
    return ytest

def pca_svm(train, test):
    f = gzip.open(train, 'rb')
    file_content = f.read()
    filearray = np.fromstring(file_content, dtype=np.float, sep=' ')
    reshaped = filearray.reshape(7291, 257)
    y = reshaped[:, 0:1]
    X = reshaped[:, 1:]
    pca = RandomizedPCA(n_components=160)
    X = pca.fit_transform(X)


    ######## ADJUST PARAMETERS HERE ##########

    filearray2 = np.fromstring(file_content, dtype=np.float, sep=' ')
    reshapedtest = filearray2.reshape(-1, 257)
    ytest = reshapedtest[:, 0:1]
    Xtest = reshapedtest[:, 1:]
    Xtest = pca.fit_transform(Xtest)
    ######## ADJUST PARAMETERS HERE ##########

    learnedmodel = SVC(kernel='linear')

    learnedmodel.fit(X, y)
    pool = multiprocessing.Pool()
    predictresult = pool.map(learnedmodel.predict, Xtest)
    # predictresult = learnedmodel.predict(kFoldTest[j])
    predictresult = np.asmatrix(predictresult)

    totalCorrect = 0
    total = 0
    for i in range(ytest.size):

        if (predictresult.item(i)) == ytest.item(i):
            totalCorrect += 1
        total += 1
    print('PCA SVM Testing Error', 1 - totalCorrect / total)
    return ytest


if __name__ == '__main__':


    # train = "/Users/kevinwei/Desktop/HW6/zip.train.gz"
    # test = "/Users/kevinwei/Desktop/HW6/zip.test.gz"
    model = sys.argv[1]
    train = sys.argv[2]
    test = sys.argv[3]
    if model == "dtree":
        decision_tree(train, test)
    elif model == "knn":
        knn(train, test)
    elif model == "svm":
        svm(train, test)
    elif model == "pcaknn":
        pca_knn(train, test)
    elif model == "pcasvm":
        pca_svm(train, test)
    else:
        print("Invalid method selected!")


