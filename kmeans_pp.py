import numpy as np
import mykmeanssp

# kmeanspp file is where we decide the K centroids that K clusters will be built around
# from here we also call kmeans algorithm that is implemented in c using capi to cluster our data

def new_min_distances(N, array, nextCent, DI, newaxis):
    # finds the minimum distance between each observation and all centroids
    dists = np.zeros([N, 2])
    dists[:, 0] = pow((np.linalg.norm(array - nextCent, axis=newaxis)),2)
    dists[:, 1] = DI
    return dists.min(axis=newaxis)

def k_means_pp(array, indexes, centroids, N, K, MAX_ITER):
    # gets the observations, indexes and centroids are just np.zero arrays, N,K are the number of obs and clusters wanted, and max iter.
    # kmeanspp algorithem, chooses the K centroids that kmeans algorithm will use.
    np.random.seed(0)  # Seed randomness

    CentroidIndex = np.random.choice(N, 1)[0]
    indexes[0] = CentroidIndex
    centroids[0,:] = array[CentroidIndex,:]
    DI = pow((np.linalg.norm(array - centroids[0,:], axis=1)),2)

    for j in range(1, K):
        NextIndex = np.random.choice(N, 1, p=DI / sum(DI))
        indexes[j] = NextIndex
        centroids[j, :] = array[NextIndex, :]
        nextCent = array[NextIndex, :]
        DI = new_min_distances(N, array,nextCent, DI, 1)

    arraylist = array.tolist()
    centroidslist = centroids.tolist()
    return mykmeanssp.kmeans(MAX_ITER, arraylist, centroidslist)