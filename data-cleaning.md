Under construction. Coming soon!

## Surveys

* [IEEE Data Engineering Bulletin March 2021 Special Issue on Data Validation for ML](http://sites.computer.org/debull/A21mar/issue1.htm}
* Xu Chu, Ihab F. Ilyas, Sanjay Krishnan, Jiannan Wang: Data Cleaning: Overview and Emerging Challenges. SIGMOD Conference 2016: 2201-2206


## Traditional and ML-based Data Cleaning

These tools focus on identify errors in datasets, without taking the downstream model or application into account.
These include traditional constraint-based data cleaning methods, as well as those that _use_ machine learning to 
detect and resolve data errors.

* [HoloClean](https://arxiv.org/pdf/1702.00820.pdf) functional dependencies, quantitative statistics, external information as a single factor-graph model.
* [Raha](https://dl.acm.org/doi/abs/10.1145/3299869.3324956) uses a library of error detectors, and treats the output of each as a feature in a holistic detection model.  It then uses clustering and active learning to train the holistic model with few labels.
* [Picket: Self-supervised Data Diagnostics for ML Pipelines](https://arxiv.org/abs/2006.04730): self-supervision to learn an error detection model.


## ML-Aware Data Cleaning

These data cleaning tools are meant to clean training datasets, and are co-designed with the trained model
in mind.  

* [ActiveClean](https://dl.acm.org/doi/pdf/10.14778/2994509.2994514) VLDB 2016: leverages model convexity to treat cleaning as an active learning problem.
* [CPClean](https://arxiv.org/pdf/2005.05117.pdf) VLDB 2021: leverages robustness of NN classifiers to local perturbations.
* [Boost](https://arxiv.org/abs/1711.01299) and [Alpha](https://arxiv.org/abs/1904.11827)Clean: models data cleaning pipeline generation as an optimization problems, given a "data quality" function.
* [Conformance Constraints](https://dl.acm.org/doi/10.1145/3448016.3452795) SIGMOD 21: learning constraints that should fail if inference over a test record may be untrustworthy.

## Application-Aware Data Cleaning

These data cleaning tools are used to clean training datasets by using errors detected in the downstream application results.
For instance, the application may use the model as part of an analytic query and visualize the result.  If the user sees an anomaly in the visualization, she can submit the issue as a _complaint_.   

* [From Cleaning before ML to Cleaning for ML](http://sites.computer.org/debull/A21mar/p24.pdf) DE Bulletin 2021: recent survey of cleaning for and using machine learning.
* [Complaint-driven Training Data Debugging for Query 2.0](https://arxiv.org/pdf/2004.05722.pdf) SIGMOD 2020: leveraging downstream query outputs to identify erroneous training data errors as an influence analysis problem.
* [Explaining Inference Queries with Bayesian Optimization](https://arxiv.org/abs/2102.0530://arxiv.org/abs/2102.05308) VLDB 2021: leveraging downstream query outputs to identify erroneous training data errors as a hyperparameter search problem.

This line of work is closely related to the area of query explanations (e.g., [Wu2013](http://sirrice.github.io/files/papers/scorpion-vldb13.pdf), [Roy2014](https://dl.acm.org/doi/abs/10.1145/2588555.2588578), [Abuzaid2019](https://cs.stanford.edu/~matei/papers/2019/vldb_macrobase_diff.pdf)) in that it uses errors in downstream results for data debugging..


## Tools

* Data Standardization: 
  * [DataPrep.Clean](https://docs.dataprep.ai/user_guide/clean/introduction.html)
  * [Great Expectations](https://greatexpectations.io/)
* [Label Clean](https://pypi.org/project/cleanlab/)
