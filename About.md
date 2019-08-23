# Unsupervised-Learning

Myles Lefkovitz - gth836x

### This directory contains two subfolders:
- /Wine/
- /Income/

Each subfolder has all of the code used in this assignment. 

## Wine
For the Wine Quality data set, data is loaded in the wine_data.py file, which pulls it directly from the UCI site.

The dimensionality reduction and clustering algorithms are split into their own files, and call the other files in the folder as necessary.
####Clustering files are:

- kmeans: wine_kmeans.py (this file dives into the clustering algorithm at key clusters)
- kmeans: wine\_kmeans_manyclusters.py (this file compares performance for various cluster sizes) 
- EM: wine_expecatationmaximization2D.py (this file dives into the clustering algorithm at key clusters)
- EM: wine_expecatationmaximization.py (this file compares performance for various cluster sizes) 

####Dimensionality reduction files are:

- wine_PCA.py
- wine_ICA.py
- wine_RandomizedProjections.py
- wine_TruncatedSVD.py
- wine_PCA CV.py
- wine_ICA CV.py
- wine_RandomizedProjections CV manual.py
- wine_TruncatedSVD CV.py

The CV files identify the optimal number of components for each dimensionality reduction algorithm. The initial files dive into the algorithms for more analysis. 

####Clustering on Dimensionality Reduction files:

- wine\_kmeans\_manycluster_reducedPCA.py
- wine\_kmeans\_manycluster_reducedICA.py
- wine\_kmeans\_manycluster_reducedRPA.py
- wine\_kmeans\_manycluster_reducedTruncatedSVD.py
- wine\_kmeans_reducedPCA.py
- wine\_kmeans_reducedICA.py
- wine\_kmeans_reducedRPA.py
- wine\_kmeans_reducedTruncatedSVD.py

The 'manycluster' files are used to compare cluster performance. The other files are for in-depth analysis

## Income
For the Adult Income data set, data is loaded in the income_data.py file, which pulls it from the train.csv file in the directory. I hard-coded my directory in this file, so you may have to adjust this.

The learners are split into their own file, and call the other files in the folder as necessary. The structure mirrors the structure for the Wine directory.
Clustering files are:

- kmeans: income_kmeans.py (this file dives into the clustering algorithm at key clusters)
- kmeans: income\_kmeans_manyclusters.py (this file compares performance for various cluster sizes) 
- EM: income_expecatationmaximization2D.py (this file dives into the clustering algorithm at key clusters)
- EM: income_expecatationmaximization.py (this file compares performance for various cluster sizes) 

####Dimensionality reduction files are:

- income_PCA.py
- income_ICA.py
- income_RandomizedProjections.py
- income_TruncatedSVD.py
- income_PCA CV.py
- income_ICA CV.py
- income_RandomizedProjections CV manual.py
- income_TruncatedSVD CV.py

The CV files identify the optimal number of components for each dimensionality reduction algorithm. The initial files dive into the algorithms for more analysis.

####Clustering on Dimensionality Reduction files:

- income\_kmeans\_manycluster_reducedPCA.py
- income\_kmeans\_manycluster_reducedICA.py
- income\_kmeans\_manycluster_reducedRPA.py
- income\_kmeans\_manycluster_reducedTruncatedSVD.py
- income\_kmeans_reducedPCA.py
- income\_kmeans_reducedICA.py
- income\_kmeans_reducedRPA.py
- income\_kmeans_reducedTruncatedSVD.py

The 'manycluster' files are used to compare cluster performance. The other files are for in-depth analysis

####Artificial Neural Networks run on Dimensionality Reduction files:

Neural network algorithms were run only for the income dataset.

- income_neural PCA.py
- income_neural ICA.py
- income_neural RandomizedProjections.py
- income_neural TruncatedSVD.py 

####Artificial Neural Networks run on Dimensionality Reduction files with clustering:

Neural network algorithms were run only for the income dataset.

- income_neural PCA em.py
- income_neural PCA km.py
- income_neural ICA em.py
- income_neural ICA km.py
- income_neural RDA km.py
- income_neural RP em.py
- income_neural TruncatedSVD em.py
- income_neural TruncatedSVD km.py 