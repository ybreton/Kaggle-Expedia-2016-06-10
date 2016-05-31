import matplotlib.pyplot as plt
import numpy as np


def pcaplot(X):
    """

    :param X: Result of pca.fit_transform
    :return: tuple of object handles
    """

    rows = X.shape[0]
    cols = X.shape[1]

    fh = plt.figure()
    ah = list(np.zeros((cols,cols)))
    for ix in range(cols):
        for iy in range(cols):
            p = (cols * iy) + (ix + 1)
            ah=fh.add_subplot(cols, cols, p)

            if ix != iy:
                ah.plot(X[:,ix], X[:,iy], 'k.')
            else:
                ah.text(0.5, 0.5, 'Col{0}'.format(ix+1), horizontalalignment='center')
                ah.set_xlim(0, 1)
                ah.set_ylim(0, 1)


    return fh


def cumeigenplot(pca):
    P = pca.explained_variance_ratio_
    C = np.cumsum(P)
    C = np.concatenate((np.zeros(1), C),0)
    fh=plt.figure()
    plt.plot(np.linspace(0,len(C)-1,len(C)),C,'ks-')
    plt.xlabel('Principal component')
    plt.ylabel('Cum. Prop. Var. Explained')
    plt.xlim(0,len(C)-1)
    plt.ylim(-0.05,1.05)
    return fh


def eigenplot(pca):
    P = pca.explained_variance_ratio_
    P = np.concatenate((np.zeros(1), P),0)
    fh=plt.figure()
    plt.plot(np.linspace(0,len(P)-1,len(P)),P,'ks-')
    plt.xlabel('Principal component')
    plt.ylabel('Prop. Added Var. Explained')
    plt.xlim(0,len(P)-1)
    plt.ylim(-0.05,1.05)
    return fh

def gmmplot(X,g):
    """

    :param X: n x d input data for Gaussian mixture model in d-dimensions
    :param g: Gaussian mixture model g with m components
    :return: fh handle to plot object
    """
    m = g.means_

    rows = X.shape[0]
    cols = X.shape[1]

    fh = plt.figure()
    ah = list(np.zeros((cols, cols)))
    for ix in range(cols):
        for iy in range(cols):
            p = (cols * iy) + (ix + 1)
            ah = fh.add_subplot(cols, cols, p)

            if ix != iy:
                ah.plot(X[:, ix], X[:, iy], 'k.')
                ah.plot(m[:, ix], m[:, iy], 'rs')
            else:
                ah.text(0.5, 0.5, 'Col{0}'.format(ix + 1), horizontalalignment='center')
                ah.set_xlim(0, 1)
                ah.set_ylim(0, 1)

    return fh

def histcn(X, bins):
    pass