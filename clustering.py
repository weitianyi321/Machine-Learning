import numpy as np
import os, copy, sys
import matplotlib.pyplot as plt

colors = ['r','y','g','c','m','b']
def loadData(fileDj):
    with open(fileDj) as file:
        array2d = [[float(digit) for digit in line.split()] for line in file]
    array2d = np.asarray(array2d)
    firstTwoColumn = array2d[:,0:2]
    lastColumn = array2d[:,array2d.shape[1] - 1:array2d.shape[1]]
    return np.concatenate((firstTwoColumn,lastColumn), axis = 1)

def kmeans(data, k, maxIter):
    centroids = []
    centroids = randomize_centroids(data, centroids, k)

    old_centroids = [[] for i in range(k)]

    iterations = 0
    while not (has_converged(centroids, old_centroids, iterations, maxIter)):
        clusters = [[] for i in range(k)]
        clusters = euclidean_dist(data, centroids, clusters)
        index = 0
        for cluster in clusters:
            old_centroids[index] = centroids[index]
            centroids[index] = np.mean(cluster, axis=0).tolist()
            index += 1
    return clusters

def purity(X, clusters):

    purities = []
    #Your code here

    return purities

def kmeans2(data, k, maxIter):
    centroids = []


    centroids = randomize_centroids(data, centroids, k)

    old_centroids = [[] for i in range(k)]

    iterations = 0
    while not (has_converged(centroids, old_centroids, iterations, maxIter)):
        #iterations += 1
        clusters = [[] for i in range(k)]

        # assign data points to clusters
        clusters = euclidean_dist(data, centroids, clusters)

        index = 0
        for cluster in clusters:
            old_centroids[index] = centroids[index]
            centroids[index] = np.mean(cluster, axis=0).tolist()
            index += 1

    return centroids

# Calculates euclidean distance between
# a data point and all the available cluster
# centroids.
def euclidean_dist(data, centroids, clusters):

    for instance in data:
        # Find which centroid is the closest
        # to the given data point.
        mu_index = min(
            [(i[0], np.linalg.norm(instance-centroids[i[0]])) for i in enumerate(centroids)], key=lambda t:t[1])[0]
        try:
            clusters[mu_index].append(instance)
        except KeyError:
            clusters[mu_index] = [instance]

    # If any cluster is empty then assign one point
    # from data set randomly so as to not have empty
    # clusters and 0 means.
    for cluster in clusters:
        if not cluster:
            cluster.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())

    return clusters


# randomize initial centroids
def randomize_centroids(data, centroids, k):
    for cluster in range(0, k):
        centroids.append(data[np.random.randint(0, len(data), size=1)].flatten().tolist())
    return centroids


# check if clusters have converged
def has_converged(centroids, old_centroids, iterations, maxIter):
    MAX_ITERATIONS = maxIter
    if iterations > MAX_ITERATIONS:
        return True
    return old_centroids == centroids


def kneeFinding(data):
    graphdict = {1:0,2:0,3:0,4:0,5:0,6:0}
    for i in range(1, 7):
        objectivefunction = 0
        centroids = np.asmatrix(kmeans2(data,i,1000))
        #print(centroids)
        for instance in data:
            #print(instance)
            smallest = 10000
            for j in range(len(centroids)):
                #print(centroids[j:j+1,:2])
                distance = np.linalg.norm(np.asmatrix(instance) - centroids[j:j+1,:])
                if distance < smallest:
                    smallest = distance
            objectivefunction += smallest

            #print(objectivefunction)
        graphdict[i] = objectivefunction
    numbers = []
    for i in range(1, 7):
        numbers.append(graphdict.get(i))
        plt.scatter(i,numbers[i-1])
    plt.show()
    #print(graphdict)
        #for instance in data:

def purity(X, clusters):
    totalCorrectOne = 0
    totalCorrectTwo = 0
    ClusterOne = 0
    ClusterTwo = 0
    for each in clusters[0]: # all of 2s
        if each[2] == 2:
            totalCorrectOne += 1
        ClusterOne += 1
    for each in clusters[1]:
        if each[2] == 1:
            totalCorrectTwo += 1
        ClusterTwo += 1
    print('#########KMEANS PURITY#########')
    print("Purity of Cluster One: ", max(totalCorrectOne / ClusterOne, 1-totalCorrectOne / ClusterOne))
    print("Purity of Cluster Two: ", max(totalCorrectTwo / ClusterTwo, 1-totalCorrectTwo / ClusterTwo))

def visualizeClusters(clusters):
    colorindex = -1
    for cluster in clusters:
        colorindex += 1
        cluster = np.asmatrix(cluster)
        plt.scatter(cluster[:,:1],cluster[:,1:2], c = colors[colorindex])
    plt.show()


def getInitialsGMM(X,k,covType):
    if covType == 'full':
        dataArray = np.transpose(np.array([pt[0:-1] for pt in X]))
        covMat = np.cov(dataArray)
    else:
        covMatList = []
        for i in range(len(X[0])-1):
            data = [pt[i] for pt in X]
            cov = np.asscalar(np.cov(data))
            covMatList.append(cov)
        covMat = np.diag(covMatList)
    initialClusters = {}

    firstColSum = np.sum(X,axis=0)
    if X.shape[0] == 2234:
        list1 = [[74,218], covMat, 1/2]
        list2 = [[68,136], covMat, 1/2]
    else:
        list1 = [[92, -2], covMat, 1/2]
        list2 = [[82, -4], covMat, 1/2]
    initialClusters[1] = list1
    initialClusters[2] = list2

    return initialClusters


def calcLogLikelihood(X,clusters,k):
    loglikelihood = 0
    #Your code here
    return loglikelihood

#E-step
def updateEStep(X,clusters,k):

    Ematrix = []
    mu1 = clusters.get(1)[0]  # mu1, datatype:python list, e.g. [50,90]
    mu2 = clusters.get(2)[0]  # mu2, datatype:python list e.g. [60,100]
    sigma1 = clusters.get(1)[1]  # sigma1, dtype: numpy array, e.g. [[   12.46785639   122.59252597]
    # [  122.59252597  1645.54722052]]

    for instance in X[:, :2]:
        Ei = []

        diff1 = (instance - mu1).dot(np.asmatrix(sigma1).I).dot((instance - mu1).T)
        diff1 = np.exp(-0.5 * diff1) * clusters.get(1)[2]

        diff2 = (instance - mu2).dot(np.asmatrix(sigma1).I).dot((instance - mu2).T)
        diff2 = np.exp(-0.5 * diff2) * clusters.get(2)[2]
        diff1 = diff1.item(0)
        diff2 = diff2.item(0)

        Ei.append(diff1 / (diff1 + diff2))
        Ei.append(diff2 / (diff1 + diff2))
        Ematrix.append(Ei)
        #print('Ei',Ei)
    Emat = []
    for i in Ematrix:
        Emat.append(i[0])
        Emat.append(i[1])
    Emat = np.asarray(Emat)
    Emat = Emat.reshape(-1, 2)
    return Emat


#M-step
def updateMStep(X,clusters,EMatrix):
    mu = []
    EMatrixSumAxis = np.sum(EMatrix, axis=0)
    for j in range(0, 2):
        mubeforedivision = 0
        i = 0
        for instance in EMatrix[:,j:j+1]:
            mubeforedivision += X[i:i + 1,:2] * instance.item(0)
            i += 1
        mu.append(np.asarray(mubeforedivision).item(0) / EMatrixSumAxis.item(j))
        mu.append(np.asarray(mubeforedivision).item(1) / EMatrixSumAxis.item(j))
    mu = np.asarray(mu)
    mu = np.asmatrix((mu).reshape(-1, X.shape[1]-1))
    EeachCluster = np.sum(EMatrix,axis = 0)
    for i in range(len(EeachCluster)):
        clusters[i + 1][2] = EeachCluster.item(i)/X.shape[0]
        clusters[i + 1][0] = np.asarray(mu[i:i+1])
    return clusters

def visualizeClustersGMM(X,labels,clusters,covType):

    arr1 = []
    arr2 = []
    i = 0
    firstTime1 = True
    firstTime2 = True
    for instance in X[:,:2]:

        instance = np.asmatrix(instance)
        if labels[i] == 1:
            if firstTime1:
                arr1 = instance
                firstTime1 = False
            else:
                arr1 = np.concatenate((arr1,instance), axis = 0)

        else:
            if firstTime2:
                arr2 = instance
                firstTime2 = False
            else:
                arr2 = np.concatenate((arr2, instance), axis = 0)
        i += 1
    plt.scatter(arr1[:,0:1],arr1[:,1:2], c = 'b')
    plt.scatter(arr2[:,0:1],arr2[:,1:2], c = 'c')
    plt.show()
    return


def gmmCluster(X, k, covType, maxIter=1000):
    #initial clusters
    clustersGMM = getInitialsGMM(X,k,covType)
    labels = []
    for i in range(maxIter):
        temp = copy.deepcopy(clustersGMM)
        EMat = updateEStep(X, clustersGMM,2)
        clustersGMM = updateMStep(X, clustersGMM, EMat)
        if (GMMCluster(clustersGMM, temp)): break
    for instance in EMat:
        if instance.item(0) > instance.item(1):
            labels.append(1)
        else:
            labels.append(2)
    visualizeClustersGMM(X,labels,clustersGMM,covType)
    return labels,clustersGMM

def GMMCluster(clusters1, clusters2):
    ACCURACY = 0.001
    flag1 = np.linalg.norm(clusters1.get(1)[0][0] - clusters2.get(1)[0][0])
    flag3 = np.linalg.norm(clusters1.get(2)[0][0] - clusters2.get(2)[0][0])

    if flag1 + flag3 < ACCURACY:
        return True
    return False


def purityGMM(X, clusters, labels):
    purities = []
    totalOneCorrect = 0
    totalOne = 0
    totalTwoCorrect = 0
    totalTwo = 0
    for i in range(X.shape[0]):
        if labels[i] == 1:
            if labels[i] == X[i:i+1,X.shape[1] - 1:X.shape[1]].item(0):
                totalOneCorrect += 1
            totalOne += 1
        else:
            if labels[i] == X[i:i+1,X.shape[1] - 1:X.shape[1]].item(0):
                totalTwoCorrect += 1
            totalTwo += 1
    purities.append(max(totalOneCorrect/totalOne, 1-totalOneCorrect/totalOne))
    purities.append(max(totalTwoCorrect/totalTwo, 1-totalTwoCorrect/totalTwo))
    return purities



if __name__ == '__main__':
    #######dataset path
    datadir = sys.argv[1]
    pathDataset1 = datadir + '/humanData.txt'
    pathDataset2 = datadir + '/audioData.txt'
    dataset1 = loadData(pathDataset1)
    dataset2 = loadData(pathDataset2)

    #Q4
    kneeFinding(dataset1)

    #Q5
    clusters = kmeans(dataset1,2,1000)
    purity(dataset1, clusters)
    visualizeClusters(clusters)

    #Q7
    np.set_printoptions(suppress=True)
    labels11, clustersGMM11 = gmmCluster(dataset1, 2, 'diag')
    labels12, clustersGMM12 = gmmCluster(dataset1, 2, 'full')

    # Q8
    labels21, clustersGMM21 = gmmCluster(dataset2, 2, 'diag')
    labels22, clustersGMM22 = gmmCluster(dataset2, 2, 'full')

    # Q9
    purities11 = purityGMM(dataset1, clustersGMM11, labels11)
    purities12 = purityGMM(dataset1, clustersGMM12, labels12)
    purities21 = purityGMM(dataset2, clustersGMM21, labels21)
    purities22 = purityGMM(dataset2, clustersGMM22, labels22)
