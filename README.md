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
    (a) dual qp 
    (b) latent 
    (c) cluster
    (f) multiple kernel learning   
- one-class support vector machine: 
    (a) huber loss primal 
    (b) latent 
    (c) primal lp-norm w/ SGD solver 
    (d) dual qp
    (e) convex semi-supervised
    (f) multiple kernel learning   
- lp-norm mkl wrapper
- structured output support vector machine (primal)
- latent variable principle components analysis
- LODA

structured objects:
- multi-class  
- hidden markov model 

kernels and features:
- rbf, linear kernel
- histogram intersection kernel
- histogram features
- hog features


