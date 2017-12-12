import sys
# append tinyfacerec to module search path
sys.path.append("..")
# import numpy and matplotlib colormaps
import numpy as np
# import tinyfacerec modules
from tinyfacerec.subspace import pca
from tinyfacerec.util import normalize, asRowMatrix, read_images
from tinyfacerec.visual import subplot

import matplotlib.cm as cm

# read images
[X,y] = read_images("/home/slemaignan/src/tutorial-face-recognition/dataset/orl_faces")

E = []
for i in xrange(min(len(X), 10) * 10):
    E.append(normalize(X[i],0,255))

# plot them and store the plot to "python_eigenfaces.pdf"
#subplot(title="Eigenfaces AT&T Facedatabase", images=E, rows=4, cols=4, sptitle="Eigenface", colormap=cm.jet, filename="python_pca_eigenfaces.pdf")
#subplot(title="AT&T Face dataset", images=E, rows=10, cols=10, sptitle=None, ticks_visible=False, colormap=cm.gray,filename="dataset.pdf")

# perform a full pca
[D, W, mu] = pca(asRowMatrix(X), y) # D: eigenvalues, W: eigenvectors, mu: mean


# turn the first (at most) 16 eigenvectors into grayscale
# images (note: eigenvectors are stored by column!)
E = []
for i in xrange(min(len(X), 16)):
    e = W[:,i].reshape(X[0].shape)
    E.append(normalize(e,0,255))
# plot them and store the plot to "python_eigenfaces.pdf"
#subplot(title="Eigenfaces AT&T Facedatabase", images=E, rows=4, cols=4, sptitle="Eigenface", colormap=cm.jet, filename="python_pca_eigenfaces.pdf")
subplot(title="Eigenfaces", images=E, rows=4, cols=4, sptitle="Eigenface", colormap=cm.gray)

sys.exit(0)
# random example of combining Eigenvectors to rebuild a face
#subplot(title="test", images=[E[0] + E[1] * 0.5 + E[2] * 0.3], rows=1, cols=1)

from tinyfacerec.subspace import project, reconstruct

# reconstruction steps
steps=[i for i in xrange(10, min(len(X), 320), 20)]
E = []
for i in xrange(min(len(steps), 16)):
    numEvs = steps[i]
    P = project(W[:,0:numEvs], X[0].reshape(1,-1), mu)
    R = reconstruct(W[:,0:numEvs], P, mu)
    # reshape and append to plots
    R = R.reshape(X[0].shape)
    E.append(normalize(R,0,255))
# plot them and store the plot to "python_reconstruction.pdf"
#subplot(title="Reconstruction AT&T Facedatabase", images=E, rows=4, cols=4, sptitle="Eigenvectors", sptitles=steps, colormap=cm.gray, filename="python_pca_reconstruction.pdf")
subplot(title="Reconstruction of one face", images=E, rows=4, cols=4, sptitle="", sptitles=steps, colormap=cm.gray, filename="one-face.pdf")



# compare the faces reconstructed from the first 10 Eigenvectors

for numEvs in [1, 10, 50]:
    E = []
    for i in xrange(0,len(X)/2,10):
        P = project(W[:,0:numEvs], X[i].reshape(1,-1), mu)
        R = reconstruct(W[:,0:numEvs], P, mu)
        # reshape and append to plots
        R = R.reshape(X[0].shape)
        E.append(normalize(R,0,255))
        E.append(normalize(X[i],0,255))

    subplot(title="Reconstruction with %d Eigenvectors" % numEvs, images=E, rows=5, cols=8, sptitle=None,ticks_visible=False, filename="reconstruction-%d-eigenfaces.pdf"%numEvs)
