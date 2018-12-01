tilitools
=========

![Travis-CI](https://travis-ci.org/nicococo/tilitools.svg?branch=master)

**ti**ny **li**ttle machine learning **tool**box


Tilitools is a collection of (non-mainstream) machine learning model and tools 
with a special focus on anomaly detection, one-class learning, and structured data. 
Author: Nico Goernitz

Examples, description, and lectures examples can be found in the notebooks/ directory.

Test data was collected from ODDS ([outlier detection database](http://odds.cs.stonybrook.edu)). 


Currently available models:
- Bayesian data description
- support vector data description: 
    1. dual qp 
    2. latent 
    3. cluster
    4. multiple kernel learning   
- one-class support vector machine: 
    1. huber loss primal 
    2. latent 
    3. primal lp-norm w/ SGD solver 
    4. dual qp
    5. convex semi-supervised
    6. multiple kernel learning   
- lp-norm mkl wrapper
- structured output support vector machine (primal)
- latent variable principle components analysis
- LODA

structured objects:
- multi-class  
- hidden markov model 

supported kernels and features:
- rbf, linear kernel
- histogram intersection kernel
- histogram features
- hog features

lectures:
Lectures contains exercise and solution notebooks for various topics. 
Right now the following are available:
- introduction to anomaly detection
- optimization I + II
- learning with kernels I + II
- geoscience project grainstones: detecting fossils in microscopic images   

notebooks:
Contains notebooks to various topics related to machine learning, anomaly detection,
and structured output learning. Currently, the following are available:
- high-dimensional outlier detection
- the anomaly detection setting
- Bayesian data description
- introduction to SVDD and OCSVM  

data:
The data sub-directory contains well-known benchmark datasets which are
modified to fit in the anomaly detection setting. These modified datasets
can be downloaded from ODDS website ([outlier detection database](http://odds.cs.stonybrook.edu)).

