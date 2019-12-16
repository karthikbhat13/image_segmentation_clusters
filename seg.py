from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import KMeans, spectral_clustering
from sklearn.feature_extraction import image

pic = plt.imread('1.jpeg')/255
mask = pic.astype(bool)
img = pic.astype(float)

graph = image.img_to_graph(img, mask=mask)
graph.data = np.exp(-graph.data / graph.data.std())

labels = spectral_clustering(graph, n_clusters=2, eigen_solver='arpack')

label_im = np.full(mask.shape, -1.)
label_im[mask] = labels

plt.imshow(img)
plt.show()

plt.imshow(label_im)
plt.show()
