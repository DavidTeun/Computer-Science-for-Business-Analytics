# Computer-Science-for-Business-Analytics
This repository contains the algorithm accompanying the paper for Computer Science for Business Analytics '22-'23

The code given in this repository is a duplicate detection algorithm written in Python. Also included are functions used for bootstrapping results presented in the paper. A description of each function is given below.

**cleandata**
Within this function, the data set in data.frame format is used as input. The model words are extracted from the title and is returned as a vector. Model ID's are extracted and used for performance evaluation. The data is standardized and cleaned and pre-selection vectors are returned: shop, brand, resolution.

**binSigM**
This function takes as input the cleaned data set from cleandata, the vector of model words, and the number of rows and bands. Using this a binary matrix is created representing the product descriptions. Using min-hashing with randomized hash functions, a signature matrix is created which are split into a dictionary consisting of the bands.

**candMat**
This function takes as input the band dictionary and the number of bands and returns a binary matrix indicating candidates.

**disMatrix**
This function takes as input the candidate matrix, the binary matrix and the data in order to determine a dissimilarity matrix using a cosimne measure.

**clustering**
This function takes as input the clustering treshhold, the dissimilarity matrix, and the original data set to cluster the products in order to detect duplicates.

**perf**
This function uses the output of the clustering algorithm, the data set and the candidate matrix in order to evaluate the performance of the method.

**dupepredictor**
This function takes all functions into a single function in order to conveniently predict duplicates and evaluate the method

**booteval**


**tune**
Two tune functions are defined in order to optimalize the choice of bands, rows and the clustering threshold. 
