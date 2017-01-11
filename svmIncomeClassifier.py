import numpy as np
import pandas as pd
import multiprocessing
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
import random


pool = multiprocessing.Pool()
class SvmIncomeClassifier:
    def __init__(self):
        random.seed(0)

    def train_and_select_model(self, filename):

        pandasreadraw = pd.read_csv(filename, header=None)  # Be aware that pandas automatically omits the first row
        pandasread = pd.DataFrame.as_matrix(pandasreadraw)
        # CONTINUOUS DATA #
        # AGE #
        age = pandasread[:, :1]  # Age Column
        minmax_scale = preprocessing.MinMaxScaler(copy=True)
        agescaled = minmax_scale.fit_transform(age)
        # print(agescaled)
        # FNLWGT #
        fnlwgt = pandasread[:, 2:3]
        # print(fnlwgt)
        fnlwgtscaled = minmax_scale.fit_transform(fnlwgt)
        # print(fnlwgtscaled)
        # EDUCATION_NUM #
        education_num = pandasread[:, 4:5]
        # print(education_num)
        education_numscaled = minmax_scale.fit_transform(education_num)
        # print(education_numscaled)
        # CAPITAL_GAIN #
        capital_gain = pandasread[:, 10:11]
        # print(capital_gain)
        capital_gainscaled = minmax_scale.fit_transform(capital_gain)
        # print(capital_gainscaled)
        # CAPITAL_LOSS #
        capital_loss = pandasread[:, 11:12]
        # print(capital_loss)
        capital_lossscaled = minmax_scale.fit_transform(capital_loss)
        # print(capital_lossscaled)
        # HOURS #
        hours = pandasread[:, 12:13]
        # print(hours)
        hoursscaled = minmax_scale.fit_transform(hours)
        # print(hoursscaled)

        # SPARSE DATA #
        workclass = pandasread[:, 1:2]
        # print(workclass)
        bwork, cwork, toPrint = np.unique(workclass, return_inverse=True, return_counts=True)
        # toPrint here combined with bwork and cwork shows items converted as 4'Private' shows up 26939 times. I chose to replace '?' to 'private',
        # i.e. all 0 -> 4
        cwork2 = np.asmatrix(cwork)
        print(cwork2.shape)
        for i in np.nditer(cwork2, op_flags=['readwrite']):
            if i == 0:
                i[...] = 4
        # the second dimension is decreased by 1 now

        education = pandasread[:, 3:4]
        # print(education)
        bedu, cedu = np.unique(education, return_inverse=True)
        cedu2 = np.asmatrix(cedu)
        # print(c)
        maritalstatus = pandasread[:, 5:6]
        bmar, cmar = np.unique(maritalstatus, return_inverse=True)
        cmar2 = np.asmatrix(cmar)
        occupation = pandasread[:, 6:7]
        boccu, coccu = np.unique(occupation, return_inverse=True)
        for i in np.nditer(coccu, op_flags=['readwrite']):
            if i == 0:
                i[...] = 1

        coccu2 = np.asmatrix(coccu)
        relationship = pandasread[:, 7:8]
        brel, crel = np.unique(relationship, return_inverse=True)
        crel2 = np.asmatrix(crel)
        race = pandasread[:, 8:9]
        brace, crace = np.unique(race, return_inverse=True)
        crace2 = np.asmatrix(crace)
        sex = pandasread[:, 9:10]
        bsex, csex = np.unique(sex, return_inverse=True)
        csex2 = np.asmatrix(csex)
        nativecountry = pandasread[:, 13:14]
        bnat, cnat, toPrint = np.unique(nativecountry, return_inverse=True, return_counts=True)
        #print(bnat)
        #print(cnat)
        cnat2 = np.asmatrix(cnat)
        for i in np.nditer(cnat2, op_flags=['readwrite']):
            if i == 0:
                i[...] = 38

        continuous = np.concatenate((agescaled, fnlwgtscaled, education_numscaled,
                                     capital_gainscaled, capital_lossscaled,
                                     hoursscaled), axis=1)  # all sparse data
        catnumeric = np.concatenate((cwork2.T, cedu2.T, cmar2.T, coccu2.T, crel2.T,
                                     crace2.T, csex2.T, cnat2.T), axis=1)  # all sparse data

        enc = OneHotEncoder(categorical_features='all',
                            handle_unknown='error', n_values='auto', sparse=True)
        enc.fit(catnumeric)
        binarydenote = enc.transform(catnumeric).toarray()
        print(binarydenote.shape)
        preprocessed = np.concatenate((continuous, binarydenote), axis=1)  # (38842,101+6)
        print(preprocessed.shape)
        # encode y input #
        y = pandasread[:, 14:15]
        by, cy = np.unique(y, return_inverse=True)

        for x in np.nditer(cy, op_flags=['readwrite']):
            if x == 0:
                x[...] = -1
        cy = np.asmatrix(cy)
        #print(cy)
        preprocessedwithy = np.concatenate((preprocessed, cy.T), axis=1)
        print(preprocessedwithy.shape)
        # np.savetxt(X=preprocessedwithy,fname='1.txt')
        # shape:(38842,105)
        # cy is the +1/-1 output right now
        # clf = svm.SVC(kernel='linear')
        # clf.fit(preprocessed, cy)
        # for i in range(1,10):
        #    for j in range(0,3):
        #        print('C = {0}, Degree = {1}'.format(2**i,(j)))
        CVscores = SvmIncomeClassifier.CrossValidationRBF(self, data = preprocessedwithy,Cin = 256,gammain = 0.0078125)
        train = preprocessedwithy[:, 0:104]
        test = preprocessedwithy[:, 104:105]
        trainedModel = SVC(kernel='rbf', C=256, gamma=0.0078125)
        trainedModel.fit(train, test)

        # predict part#
        predictreadraw = pd.read_csv('salary.2Predict.csv', header=None)
        predictread = pd.DataFrame.as_matrix(predictreadraw)
        return trainedModel, CVscores

    def CrossValidationLinear(self, data):
        kFoldTest = []
        kFoldTrain = []
        yFoldTest = []
        yFoldTrain = []
        accuracy = 1
        for i in range(0, 3):
            totalWrong = 0
            total = 0
            k = 12947 * i
            kFoldTest.append(data[k:k + 12947, 0:104])
            yFoldTest.append(data[k:k + 12947, 104:105])
            if k == 0:
                kFoldTrain.append(data[12947:38842, 0:104])
                yFoldTrain.append(data[12947:38842, 104:105])
            else:
                kFoldTrain.append(np.concatenate((data[0:k + 1, 0:104], data[k + 12947:38842, 0:104]), axis=0))
                yFoldTrain.append(np.concatenate((data[0:k + 1, 104:105], data[k + 12947:38842, 104:105]), axis=0))

        for j in range(3):
            learnedmodel = SVC(kernel='linear')
            print(j)
            print(kFoldTrain[j].shape)
            print(yFoldTrain[j].shape)
            learnedmodel.fit(kFoldTrain[j], yFoldTrain[j])
            predictresult = learnedmodel.predict([kFoldTest[j]])
            # predictresult = learnedmodel.predict(kFoldTest[j])
            predictresult = np.asmatrix(predictresult)
            print(predictresult.shape)
            print(yFoldTest[j].shape)
            print(predictresult.T)
            print(yFoldTest[j])
            # predict result is a (12947,1) matrix(or vector?) dot multiply it with real test outcome,
            # accuracy = n(+1) / n(-1)
            # to print certain item in the matrix:use  np.item(x)
            plus = predictresult.T + yFoldTest[j]
            for i in range(plus.size):
                if plus.item(i) == 0:
                    totalWrong += 1
                total += 1
            print(1 - totalWrong / total)
        return accuracy

    # use this function to do rbf
    def CrossValidationRBF(self, data, Cin, gammain):
        kFoldTest = []
        kFoldTrain = []
        yFoldTest = []
        yFoldTrain = []
        scoreaverage = 0
        for i in range(0, 3):
            totalWrong = 0
            total = 0
            k = 12947 * i
            kFoldTest.append(data[k:k + 12947, 0:104])
            yFoldTest.append(data[k:k + 12947, 104:105])
            if k == 0:
                kFoldTrain.append(data[12947:38842, 0:104])
                yFoldTrain.append(data[12947:38842, 104:105])
            else:
                kFoldTrain.append(np.concatenate((data[0:k + 1, 0:104], data[k + 12947:38842, 0:104]), axis=0))
                yFoldTrain.append(np.concatenate((data[0:k + 1, 104:105], data[k + 12947:38842, 104:105]), axis=0))

        for j in range(3):
            learnedmodel = SVC(kernel='rbf', C=Cin, gamma=gammain)
            learnedmodel.fit(kFoldTrain[j], yFoldTrain[j])
            print(kFoldTest[j])
            predictresult = pool.map(learnedmodel.predict, [kFoldTest[j]])
            # predictresult = learnedmodel.predict(kFoldTest[j])
            predictresult = np.asmatrix(predictresult)
            # predict result is a (12947,1) matrix(or vector?) dot multiply it with real test outcome,
            # accuracy = n(+1) / n(-1)
            # to print certain item in the matrix:use  np.item(x)
            plus = predictresult.T + yFoldTest[j]
            for i in range(plus.size):
                if plus.item(i) == 0:
                    totalWrong += 1
                total += 1
            print('Cross Validation Serial {0} has an accuracy of: {1}'.format(j, 1 - totalWrong / total))
            scoreaverage = scoreaverage + (1 - totalWrong / total) / 3
        return scoreaverage

    def CrossValidationPoly(self, data, Cin, degreein):
        kFoldTest = []
        kFoldTrain = []
        yFoldTest = []
        yFoldTrain = []
        accuracy = 1
        for i in range(0, 3):
            totalWrong = 0
            total = 0
            k = 12947 * i
            kFoldTest.append(data[k:k + 12947, 0:104])
            yFoldTest.append(data[k:k + 12947, 104:105])
            if k == 0:
                kFoldTrain.append(data[12947:38842, 0:104])
                yFoldTrain.append(data[12947:38842, 104:105])
            else:
                kFoldTrain.append(np.concatenate((data[0:k + 1, 0:104], data[k + 12947:38842, 0:104]), axis=0))
                yFoldTrain.append(np.concatenate((data[0:k + 1, 104:105], data[k + 12947:38842, 104:105]), axis=0))

        for j in range(3):
            learnedmodel = SVC(kernel='poly', C=Cin, degree=degreein)
            learnedmodel.fit(kFoldTrain[j], yFoldTrain[j])
            predictresult = pool.map(learnedmodel.predict, [kFoldTest[j]])
            # predictresult = learnedmodel.predict(kFoldTest[j])
            predictresult = np.asmatrix(predictresult)
            # predict result is a (12947,1) matrix(or vector?) dot multiply it with real test outcome,
            # accuracy = n(+1) / n(-1)
            # to print certain item in the matrix:use  np.item(x)
            plus = predictresult.T + yFoldTest[j]
            for i in range(plus.size):
                if plus.item(i) == 0:
                    totalWrong += 1
                total += 1
            print('Cross Validation Serial {0} has an accuracy of: {1}'.format(j, 1 - totalWrong / total))
        return learnedmodel

    def predict(self, filename, trainedModel):

        pandasreadraw = pd.read_csv(filename, header=None)  # Be aware that pandas automatically omits the first row
        pandasread = pd.DataFrame.as_matrix(pandasreadraw)
        # CONTINUOUS DATA #
        # AGE #
        age = pandasread[:, :1]  # Age Column
        minmax_scale = preprocessing.MinMaxScaler(copy=True)
        agescaled = minmax_scale.fit_transform(age)
        # print(agescaled)
        # FNLWGT #
        fnlwgt = pandasread[:, 2:3]
        # print(fnlwgt)
        fnlwgtscaled = minmax_scale.fit_transform(fnlwgt)
        # print(fnlwgtscaled)
        # EDUCATION_NUM #
        education_num = pandasread[:, 4:5]
        # print(education_num)
        education_numscaled = minmax_scale.fit_transform(education_num)
        # print(education_numscaled)
        # CAPITAL_GAIN #
        capital_gain = pandasread[:, 10:11]
        # print(capital_gain)
        capital_gainscaled = minmax_scale.fit_transform(capital_gain)
        # print(capital_gainscaled)
        # CAPITAL_LOSS #
        capital_loss = pandasread[:, 11:12]
        # print(capital_loss)
        capital_lossscaled = minmax_scale.fit_transform(capital_loss)
        # print(capital_lossscaled)
        # HOURS #
        hours = pandasread[:, 12:13]
        # print(hours)
        hoursscaled = minmax_scale.fit_transform(hours)
        # print(hoursscaled)

        # SPARSE DATA #
        workclass = pandasread[:, 1:2]
        # print(workclass)
        bwork, cwork, toPrint = np.unique(workclass, return_inverse=True, return_counts=True)
        # toPrint here combined with bwork and cwork shows items converted as 4'Private' shows up 26939 times. I chose to replace '?' to 'private',
        # i.e. all 0 -> 4
        cwork2 = np.asmatrix(cwork)
        #print(cwork2.shape)
        for i in np.nditer(cwork2, op_flags=['readwrite']):
            if i == 0:
                i[...] = 4
        # the second dimension is decreased by 1 now

        education = pandasread[:, 3:4]
        # print(education)
        bedu, cedu = np.unique(education, return_inverse=True)
        cedu2 = np.asmatrix(cedu)
        # print(c)
        maritalstatus = pandasread[:, 5:6]
        bmar, cmar = np.unique(maritalstatus, return_inverse=True)
        cmar2 = np.asmatrix(cmar)
        occupation = pandasread[:, 6:7]
        boccu, coccu = np.unique(occupation, return_inverse=True)
        for i in np.nditer(coccu, op_flags=['readwrite']):
            if i == 0:
                i[...] = 1

        coccu2 = np.asmatrix(coccu)
        relationship = pandasread[:, 7:8]
        brel, crel = np.unique(relationship, return_inverse=True)
        crel2 = np.asmatrix(crel)
        race = pandasread[:, 8:9]
        brace, crace = np.unique(race, return_inverse=True)
        crace2 = np.asmatrix(crace)
        sex = pandasread[:, 9:10]
        bsex, csex = np.unique(sex, return_inverse=True)
        csex2 = np.asmatrix(csex)
        nativecountry = pandasread[:, 13:14]
        bnat, cnat, toPrint = np.unique(nativecountry, return_inverse=True, return_counts=True)
        print(bnat)
        print(cnat)
        cnat2 = np.asmatrix(cnat)
        for i in np.nditer(cnat2, op_flags=['readwrite']):
            if i == 0:
                i[...] = 39
        for i in np.nditer(cnat2, op_flags=['readwrite']):
            if i == 15:
                i[...] = 39
        for i in np.nditer(cnat2, op_flags=['readwrite']):
            if i > 15:
                i[...] = i - 1
        continuous = np.concatenate((agescaled, fnlwgtscaled, education_numscaled,
                                     capital_gainscaled, capital_lossscaled,
                                     hoursscaled), axis=1)  # all sparse data
        catnumeric = np.concatenate((cwork2.T, cedu2.T, cmar2.T, coccu2.T, crel2.T,
                                     crace2.T, csex2.T, cnat2.T), axis=1)  # all sparse data

        enc = OneHotEncoder(categorical_features='all',
                            handle_unknown='error', n_values='auto', sparse=True)
        enc.fit(catnumeric)
        binarydenote = enc.transform(catnumeric).toarray()
        # print(binarydenote.shape)
        preprocessed = np.concatenate((continuous, binarydenote), axis=1)  # (38842,101+6)
        # print(preprocessed.shape)
        predictresult = pool.map(trainedModel.predict, preprocessed)
        # predictresult = learnedmodel.predict(kFoldTest[j])
        predictresult = np.asmatrix(predictresult)
        predictStringArray = []
        for i in np.nditer(predictresult, op_flags=['readwrite']):
            if i == -1:
                predictStringArray.append('<=50K')
            else:
                predictStringArray.append('>50K')
        #print(predictStringArray)
        return predictresult

    def predictDataSet(self, filename):
        trainedModel, CVscore = SvmIncomeClassifier.train_and_select_model(filename)
        prediction = SvmIncomeClassifier.predict(filename, trainedModel)
        return prediction

    def output_results(self, something):
        with open('predictions.txt', 'w') as f:
            for pred in predictions:
                if pred == -1:
                    f.write('<=50K\n')
                else:
                    f.write('>50K\n')


if __name__ == "__main__":
    training_csv = "salary.labeled.csv"
    testing_csv = "salary.2Predict.csv"
    clf = SvmIncomeClassifier()
    trained_model, cv_score = clf.train_and_select_model(training_csv)

    predictions = clf.predict(testing_csv, trained_model)
    clf.output_results(predictions)
    print("The best model was scored %.2f" % cv_score)

    #predict('salary.2Predict.csv',trainedModel)