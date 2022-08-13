import numpy as np
import math
from sklearn import datasets
import kmeans_pp
import argparse
import random
import matplotlib.pyplot as plt

# main file is where we make all of the things needed for Spectrul clustering algorithm, and also for kmeanspp
# that means finding the eigengap heuristic k, the matrix T and so on...
# in main we also make the different files like the pdf file, data.txt and clusters.txt
# it includes many small functions like Gramshmidt, qr iteration, and more, all used by "main" function



def GramSchmidt(A):
    # this function takes a matrix and returns two matrices that are the QR decomposition of the given matrix using the
    # modified gram schmidt algorithm, we could have done it the naive way but since we are using numpy we managed to
    # do it using only 1 loop which is a lot faster
    n = A.shape[0]
    U = A.astype('float64').copy()
    R = np.zeros([n, n] , dtype=np.float64)
    Q = np.zeros([n, n] , dtype=np.float64)
    for i in range(n):
        R[i][i] = np.linalg.norm(U[:, i])
        Q[:, i] = U[:, i] / R[i][i]
        R[i, i + 1:] = Q[:, i].dot(U[:, i + 1:])
        U[:, i + 1:] = U[:, i + 1:] - Q[:, i, np.newaxis].dot(R[np.newaxis, i, i + 1:])
    return Q, R


def QR_Iteration(A):
    # this function recieves a matrix A and returns two other matrices such that one of them is orthogonal whose columns
    # approach the eigenvectors of A and the other matrix's diagonal values approach the eigenvalues
    # if we get very close to the real values we return the matrices and don't continue the loop
    n = A.shape[0]
    Anew = A.astype('float64').copy()
    Qnew = np.eye(n, dtype=np.float64)
    for i in range(n):
        Q, R = GramSchmidt(Anew)
        Anew = R.dot(Q)
        T = Qnew.dot(Q)
        if abs(abs(Qnew) - abs(T)).max() <= 0.0001:
            return Anew, Qnew
        Qnew = T
    return Anew, Qnew


def InitAdjacencyMatrix(X):
    # given the N observations of our program this function will create the weighted adjacency matrix (NxN) which is the matrix W
    # where W(ij) = exp(-norm(xi-xj)/2) for i!=j, and w(ii)=0
    n = X.shape[0]
    W = np.zeros([n, n], dtype=np.float64)
    for i in range(n):
        W[i] = pow(math.e, (-0.5 * np.linalg.norm(X[i] - X, axis=1)))
    for i in range(n):
        W[i][i] = 0
    return W


def DiagonalDegreeMatrix(W):
    # given the weighted adjacency matrix we return the diagonal matrix D^(-0.5) that is described in the assignment
    n = W.shape[0]
    D = np.zeros([n, n],dtype=np.float64)
    for i in range(n):
        D[i][i] = 1 / np.sqrt(np.sum(W[i, :]), dtype=np.float64)
    return D


def NormalizedGraphLaplacian(W, D):
    # given the matrices W,D that we computed earlier (the weighted adjacency matrix and the diagonal matrix) we return
    # the matrix lnorm which is I - D^(0.5)*W*D^(0.5), we need that matrix for the eigengap heuristic
    n = W.shape[0]
    I = np.identity(n, dtype=np.float64)
    Lnorm = I - (D.dot(W)).dot(D)
    Lnorm = Lnorm + np.eye(n)
    return Lnorm


def DetermineKAndObtainU(Lnorm, K, RANDOM):
    # this function gets Lnorm, K made by user(it will use it only if the input to RANDOM parameter of the program was false),
    # and RANDOM inputed by the user
    # it will find the eigengap huristic k and the matrix U from algorithm 3 in the assignment
    U = []
    n = Lnorm.shape[0]
    Anew, Qnew = QR_Iteration(Lnorm)
    arr = Anew.diagonal()
    sortedindexes = arr.argsort()
    max = -1
    k = 0
    for i in range(n // 2):
        t = abs(arr[sortedindexes[i]]-arr[sortedindexes[i+1]])
        if (t>max):
            max = t
            k = i + 1
    if RANDOM == "False":
        U = Qnew[:, sortedindexes[0:K]]
    if RANDOM == "True":
        U = Qnew[:,sortedindexes[0:k]]

    return k,U

def FormT(U):
    # this function gets the matrix U and returns T that is needed for algorithm 3 in the assignment, same as U but normalized
    n = U.shape[0]
    k = U.shape[1]
    T = np.zeros([n,k] , dtype=np.float64)
    for i in range(n):
        T[i] = U[i]/np.linalg.norm(U[i])
    return T

def JaccardMeasure(XImapping, Y):
    # this function gets the mapping of the observations into clusters by our algorithem(one of them), and also the
    # real mapping made by make blobs, and it measures the quality of our clustering, returns value between 0 and 1
    # when 0 is very bad clustering and 1 is very good.
    numerator = 0
    denominator = 0
    for i in range(len(XImapping)):
        for j in range(i+1,len(XImapping)):
           if((XImapping[i]==XImapping[j]) and (Y[i]==Y[j])):
               numerator = numerator + 1
           if((XImapping[i]==XImapping[j]) or (Y[i]==Y[j])):
               denominator = denominator + 1
    jaccard = numerator / denominator
    return jaccard


def GenerateString(N, number_of_clusters_data, number_of_clusters_for_algo, XImapping, XImapping2,  Y):
    # this function generates the string that will be printed in the pdf file made by MakePdf function.
    # it recives number of observations, number of clusters used by make blobs, number of clusters that will be used for
    # the algorithm, the mappings of the observations by each algorithm and also the real mapping(Y)
    str1 = "Data was generated from the values:\n"
    str1 += "n = " + str(N) + " , " + "k = " + str(number_of_clusters_data) + "\n"
    str1 += "The k that was used for both algorithms was " + str(number_of_clusters_for_algo) + "\n"
    str1 += "The Jaccard measure for Spectral Clustering: " + str(np.round(JaccardMeasure(XImapping, Y),2)) + "\n"
    str1 += "The Jaccard measure for K-means " + str(np.round(JaccardMeasure(XImapping2, Y),2))
    return str1

def MakePdf(D, X , XImapping, XImapping2, str1):
    # gets dimension of each observation, the observations, and the mapping of each of the observation to a cluster
    # by both of the algorithems (XImapping1 for spectrul, and 2 for kmeans), it also gets str1 which is the message printed for the user in the pdf file
    # it outputs the pdf as wanted in the assignment
    f = plt.figure()
    if (D == 2):
        plt.subplot(1, 2, 1)
        plt.scatter(X[:, 0], X[:, 1], c=XImapping)
        plt.title("Normalized Spectral Clustering")
        plt.subplot(1, 2, 2)
        plt.scatter(X[:, 0], X[:, 1], c=XImapping2)
        plt.title("K-means")

        ax1 = f.add_subplot(223)
        ax1.set_axis_off()
        ax1.text(1.23, -0.9, str1, ha="center")

    if (D == 3):
        ax = f.add_subplot(1, 2, 1, projection='3d')
        ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=XImapping, marker='o')
        plt.title("Normalized Spectral Clustering")
        ax = f.add_subplot(1, 2, 2, projection='3d')
        ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=XImapping2, marker='o')
        plt.title("K-means")

        ax1 = f.add_subplot(223)
        ax1.set_axis_off()
        ax1.text(1.23, -0.9, str1, ha="center")

    f.savefig("clusters.pdf", bbox_inches='tight')

def MakeDataTxtFile(N,D,X,Y):
    # recives number of observations(N), each observation dimension(D), the observations, and the true clustering of them
    # all with respect to make blobs, and it outputs the data txt file as wanted in the assignment
    datafile = open("data.txt", "w")
    for i in range(N):
        for j in range(D):
            datafile.write(str(X[i][j]) + ",")
        datafile.write(str(Y[i]) + "\n")
    datafile.close()

def MakeClustersTxtFile(number_of_clusters_for_algo, clusters, clusters2):
    # this function needs to know how many clusters are going to be used for both algorithms and the clusters
    # that each algorithem returned, it will make the clusters txt file as wanted in the assignment
    file = open("clusters.txt", "w")
    file.write(str(number_of_clusters_for_algo))
    file.write("\n")
    for i in range(number_of_clusters_for_algo):
        for j in range(len(clusters[i])):
            file.write(str(clusters[i][j]))
            if (j < len(clusters[i]) - 1):
                file.write(",")
        file.write("\n")

    for i in range(number_of_clusters_for_algo):
        for j in range(len(clusters2[i])):
            file.write(str(clusters2[i][j]))
            if (j < len(clusters2[i]) - 1):
                file.write(",")
        file.write("\n")
    file.close()


def main():
    # the main function that holds our program together, here we print the max capacity and then extract the input from
    # the user and then use make blobs in order to get the random observations around k centers, and then we use both
    # algorithms that we implemented (kmeans and normalised clustering) in order to cluster the random observations we got
    # we create also the needed files
    MAX_N_CAP_2D = 420
    MAX_N_CAP_3D = 420
    MAX_K_CAP_2D = 20
    MAX_K_CAP_3D = 20
    print("Max capacity for 2D is : n = "+str(MAX_N_CAP_2D) + " K = "+str(MAX_K_CAP_2D))
    print("Max capacity for 3D is : n = "+str(MAX_N_CAP_3D) + " K = "+str(MAX_K_CAP_3D))
    XImapping = []
    XImapping2 = []
    number_of_clusters_for_algo = 0
    number_of_clusters_data = 0
    parser = argparse.ArgumentParser()
    parser.add_argument("firstparam",type=int)
    parser.add_argument("secondparam",type=int)
    parser.add_argument("thirdparam")
    args = parser.parse_args()
    D = random.randint(2,3)
    N = args.firstparam
    K = args.secondparam
    RANDOM = args.thirdparam
    MAX_ITER = 300

    if RANDOM == "True":
        if D == 2:
            N = random.randint(MAX_N_CAP_2D // 2, MAX_N_CAP_2D)
            K = random.randint(MAX_K_CAP_2D // 2, MAX_K_CAP_2D)
        if D == 3:
            N = random.randint(MAX_N_CAP_3D // 2, MAX_N_CAP_3D)
            K = random.randint(MAX_K_CAP_3D // 2, MAX_K_CAP_3D)
        number_of_clusters_data = K

    if RANDOM == "False":
        number_of_clusters_data = K
        number_of_clusters_for_algo = K

    if (N <= 0 or K <= 0 or K >= N):
        print("invalid input for k or n")
        exit()

    X, Y = datasets.make_blobs(n_samples=N, n_features=D, centers=number_of_clusters_data)
    MakeDataTxtFile(N,D,X,Y)
    W = InitAdjacencyMatrix(X)
    Diag = DiagonalDegreeMatrix(W)
    Lnorm = NormalizedGraphLaplacian(W, Diag)
    k, U = DetermineKAndObtainU(Lnorm, K , RANDOM)

    if (((k == 1) and (RANDOM == "True")) or ((K == 1) and (RANDOM == "False"))):
        XImapping = [0 for i in range(N)]
        XImapping2 = [0 for i in range(N)]

    T = FormT(U)
    T_column_size = K # if random = false
    if RANDOM == "True":
        number_of_clusters_for_algo = k
        T_column_size = k

    indexes = np.zeros(number_of_clusters_for_algo, dtype=int)
    centroids = np.zeros([number_of_clusters_for_algo, T_column_size], dtype=float)
    indexes2 = np.zeros(number_of_clusters_for_algo, dtype=int)
    centroids2 = np.zeros([number_of_clusters_for_algo,D], dtype=float)

    if not((((k == 1) and (RANDOM == "True")) or ((K == 1) and (RANDOM == "False")))):
        XImapping = kmeans_pp.k_means_pp(T, indexes, centroids, N, number_of_clusters_for_algo, MAX_ITER)
        XImapping2 = kmeans_pp.k_means_pp(X, indexes2, centroids2, N, number_of_clusters_for_algo, MAX_ITER)

    clusters = [[] for i in range(number_of_clusters_for_algo)]
    clusters2 = [[] for i in range(number_of_clusters_for_algo)]
    for i in range(N):
        clusters[XImapping[i]].append(i)
        clusters2[XImapping2[i]].append(i)

    MakeClustersTxtFile(number_of_clusters_for_algo, clusters, clusters2)
    str1 = GenerateString(N, number_of_clusters_data, number_of_clusters_for_algo, XImapping, XImapping2, Y)
    MakePdf(D, X, XImapping, XImapping2, str1)

if __name__ == '__main__':
    main()