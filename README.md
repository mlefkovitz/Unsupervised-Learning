# Unsupervised Learning

### Introduction

In this project I compared 6 popular unsupervised learning algorithms (2 clustering, 4 dimensionality reduction) against 2 different data sets (wine quality and adult income) with the goal being accurate prediction. Only the adult income data set results are described below, see the full report for wine quality results.

Clustering Algorithms:
- k-means clustering (KM)
- Expectation Maximization (EM)

Dimensionality Reduction Algorithms:
- Principal Component Analysis (PCA)
- Independent Component Analysis (ICA)
- Randomized Projections (RP)
- Truncated SVD

After intially trying each unsupervised learning approach independently, I couple each dimensionality reduction algorithm with a clustering algorithm. Then a neural network supervised learning algorithm is run on dimensionality reduced data and further, dimensionality reduced and clustered data.

The full report is available here: [Report](/Analysis.pdf)

#### The Data

The **adult income** problem classifies adults into one of two income categories: ‘>50K’, or ‘<=50K’. The ‘>50K’ category identifies individuals that earned more than $50,000 in the given year, 1994. The ‘<=50K’ category identifies individuals that earned less than or equal to $50,000. $50,000 in 1994 is approximately $81,000 in today’s terms. The data has 13 attributes, 5 of which are real valued (age, hours worked per week, etc), and 8 of which are categorical (education, marital status, race, etc).

#### Clustering Algorithms

The data set was scaled and standardized and run through 1-100 clusters for each clustering algorithm. The results were reviewed and graphs were produced for deeper analysis (full results available in the report linked above).

On analyzing **k-means clustering**, 2 clusters was found as the optimal number of clusters.

When performing **expectation maximization**, 2 clusters was found as the optimal number of clusters.

Expectation maximization appears to be more willing to create clusters from multiple features.

#### Dimensionality Reduction Algorithms

The data set was scaled and standardized and run through 1-45 components.  The output of each dimensionality reduction algorithm was run through an SVM learner and optimized based on the results. Reconstruction error was checked for PCA, RP, and SVD. Kurtosis was checked for ICA.

![Income Dimensionality Reduction Chart](./Income/Final%20Graphs/Income%20Dimensionality%20Reduction%20Table.png)

By categorizing 84% of the CV results and 84% of the test set correctly with SVM, **Truncated SVD** algorithm performed best on the data with 34 components.

#### Clustering on Dimensionality Reduced Data

I used the same methodology discussed in the clustering section to determine the appropriate number of clusters for each clustering algorithm on each dataset. I used the optimal number of components discussed in dimensionality reduction above as inputs and reduced the dimension of the data before running the clustering algorithms. 

The accuracy of these algorithms in combination for correctly identifying labels is fairly low (lower than random chance).

#### Neural Network Learning on Dimensionality Reduced Data

![Neural Network Learning on Dimensionality Reduced Data](./Income/Final%20Graphs/NN%20Learning%20on%20Dimensionality%20Reduced%20Data.png)

In this section I ran a neural network supervised learning algorithm on the previously dimensionality reduced data.

**Principle Component Analysis (PCA)**

Using the optimal number of components, PCA returned stronger results than the ANN learner did alone, in under 30 seconds (~1% of the time the learner alone took).

**Independent Component Analysis (ICA)**

Using the optimal number of components, ICA returned stronger results than the ANN learner did alone, in under 30 seconds (~1% of the time the learner alone took).

**Randomized Projection (RP)**

Using the optimal number of components, randomized projections returned strong results in under 30 seconds (~1% of the time the learner alone took). These results were not as strong as the learner alone.

**Truncated Singular Value Decomposition (Truncated SVD)**

Using the optimal number of components, Truncated SVD returned stronger results than the ANN learner did alone, in under 30 seconds (~1% of the time the learner alone took).

In summary, the curse of dimensionality is very real. It's shocking to see that dimensionality reduction algorithms used before supervised learning could improve the learner’s effectiveness while increasing their speed dramatically.

#### Neural Network Learning on Dimensionality Reduced and Clustered Data

![Neural Network Learning on Dimensionality Reduced and Clustered Data](./Income/Final%20Graphs/NN%20Learning%20on%20Dimensionality%20Reduced%20and%20Clustered%20Data.png)

Running the neural network learner after **k-means clustering** on **dimensionality reduced** data did not appear to yeild strong results. The accuracy is lower than running the algorithm on the raw data and the dimensionality reduced data. Runtimes were especially low for some of these algorithms, but the decreased cost isn’t enough to make up for the lost performance.

Running the neural network learner on data that had been **dimensionally reduced** by one of the algorithms and again by **expectation maximization** yielded far better results. Some results (PCA with 34 components + EM with 48 clusters) outperformed the neural network learner on raw data, with test accuracy of 84.9% (the highest ANNs had achieved was 83.5% test set accuracy; SVMs were able to go as high as 83.9%). On top of this increased accuracy was a dramatic decrease in runtime. The ANN on raw data ran for 49 minutes. The ANNs ran against our reduced datasets in under 30 seconds each time. 
