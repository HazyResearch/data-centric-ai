# MLOps

The central role of data makes the development and deployment of ML/AI applications an human-in-the-loop process. 
This is a complex process in which human engineers could make mistakes, require guidance, or need to be warned when something unexpected happens. The goal of MLOps is to provide principled ways for lifecycle management, monitoring, and validation.

Researchers have started tackling theses challenges by developing new techniques and building systems such as [TFX](https://arxiv.org/pdf/2010.02013.pdf), [Ease.ML](http://cidrdb.org/cidr2021/papers/cidr2021_paper26.pdf) or [Overton](https://www.cs.stanford.edu/~chrismre/papers/overton-tr.pdf) designed to handle the entire lifecycle of a machine learning model both during development and in production. These systems typically constis of distinct components in charge of handling specific stages (e.g., pre- or post-training) or aspects (e.g., monitoring or debugging) of MLOps.

We provide a more detailed overview of prominent MLOps stages or aspects with a strong research-focus in the remainder of this area, noting that this field of research is relatively young and consits of many connections to other areas of data-centric AI. Additionally, notice that most sections are inspired by well-estiblished DevOps techniques one encounters when developing traditional software artifacts. Adopting these techniques to MLOps in a rigorous and statistical sound way is often non-trivial as one has the take into account the inherent randomness of ML tasks and its finite-sample data dependency.

<h2 id="mlops-data-acquisition-feasibility-study">Feasibility Study and Data Acquisition</h2>

In DevOps practices, new projects typically are evaluated upon theire probability of success via a feasibility study. Whilst performing a similar task in the context of ML dev has some similiarties such as hardware and engineering availability, there are two different approaches to evaluated the feasibility of ML project with respect to its data.

The first one aims at inspecting the probability of success for a fixed class of models and the amount of data given. Well studied relations between the sample and model complexity in machine learning (i.e., [VC dimension](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_dimension)) have found little application in estimating the performance of ML models. [Kong and Valiant](https://arxiv.org/abs/1805.01626) estimate the best possible accuracy a model class can achieve for a fixed distribution based on a sublinear amount of data (with respect to its dimension). Beeing a promising approach, the method is currenlty only applicable to linear models. More recently, [Hashimoto](http://proceedings.mlr.press/v139/hashimoto21a/hashimoto21a.pdf) shows that assuming a simple log-linear evolution of the empirical performance of a ML model with respect to the number of training samples, one can provably prodict the model performance along with potentional compositions of multiple data source for simpel linear models or general M-estimators. Whilst the theory does not hold for more complex models such as neural networks, the author provides some empiricial evidence showing the successfull application in this setting.

The second approach is model class agnostic and tackles feasibility study of a ML project by estimating the Bayes error rate (BER) of the data distribution and hence reporting the limitations in best possible accuracy which even acquiring more data could not bypass. Accurately estimating the BER with finite data is a hard problem with years of research. Recently, [Ease.ML/snoopy](http://www.vldb.org/pvldb/vol13/p2837-renggli.pdf) suggests to used the power of a very simple 1-nearest-neighbor estimator on top of pre-trained embeddings to achieved faster and more accurate BER bounds estimation values without beeing sensitive to any distribution dependent hyper-parameters. Estimating the BER for a ML tasks is not only usefull to prevent unrealisitc expectations when starting to work on a new ML project, it has also its application in providing security guarantees for website fingerprinting defences (see [Cherubin](https://petsymposium.org/2017/papers/issue4/paper50-2017-4-source.pdf)).


<h2 id="mlops-cicdct">CI/CD/CT</h2>

Continuous integration (CI), continuous delivery (CD) and continuous testing (CT) are well established techniques in DevOps in order to ensure safe and faster lifecycles whilst continuously updating code in production.
Various systems such as DVC's [Continyous Machine Learning (CML)](https://cml.dev/) or [MLModelCI](https://arxiv.org/abs/2006.05096) have been designed to efficiently handle the CI/CD parts. The key challenges from a ML specific viewpoint lie in runtime (latency and throughput) requirements when moving new models into production, which rely on many hyper-parameters such as batch-size and underlying hardware properties. 

When continously testing ML models one has to be carefull to not be fooled by the inhertig randomness of ML, nor to overfitt to a testset if one plans on re-using the same dataset for testing the ML model multiple times. [Ease.ML/CI](https://mlsys.org/Conferences/2019/doc/2019/162.pdf) handles both aspectes for specific test conditions (e.g., the new model has to be better than the old by a fixed number of points) from theoretical perspective. It offeres strong statistical guarantees with reducing the samples complexity required as much as possible. The technical challenges for efficiently building these techniques into a CI system are described by [Karlaš et. al.](https://dl.acm.org/doi/abs/10.1145/3394486.3403290) This [blog post](https://ds3lab.ghost.io/ci/) further described the statistical and technical challenges and how they are approached.

<h2 id="mlops-deployment-model-managemen">Deployment and Model Management</h2>

There is typically not only a single model being developed or active in production. Various online repositories such as [Hugging Face](https://huggingface.co/models), [PyTorch Hub](https://pytorch.org/hub/) or [TensorFlow Hub](https://tfhub.dev/) facilitate sharing and reusing pre-trained models. Other systems such as [ModelDB](https://dm-gatech.github.io/CS8803-Fall2018-DML-Papers/hilda-modeldb.pdf), [DVC](https://dvc.org/) or [MLFlow](https://cs.stanford.edu/~matei/papers/2018/ieee_mlflow.pdf) extend the repository functionality by further enabling version of models and dataset, tracking of experiments and efficient deployment into production.

<h2 id="mlops-monitoring">Monitoring and Adaptation</h2>

It is well known that the accuracy of active models typically diminishes over time. The main reason for this lies in distribution shift between the new real-time test data and the data used to train the model originally. The most prominent remedy to this problem still represents a periodic (sometimes on a daily or even hourly basis) re-training using fresh training data. This is a very costly undertaken which can be prevented by having access to so called drift detectors (also called anomaly or outlier detectors). [MLDemon](https://arxiv.org/abs/2104.13621) models a human-in-the-loop approach to minimize the number of required verifications. [Klaise et. al.](https://arxiv.org/abs/2007.06299) suggest that outlier detectors should be coupled with explainable AI (XAI) techniques to help humans understand the predictions and potentional model drift.

In an ideal world, we would want an ML system in production to automaticaly adapt to changes in data distribution without the need of re-training from scratch. This area if known as continual learning (CL) or lifelong learning. Merging these algorithmic ideas into a working system is non-trivial as shown by [ModelCI-e](https://arxiv.org/pdf/2106.03122.pdf) and the related work cited therein.

<h2 id="mlops-debugging">Debugging</h2>

Debugging a ML model is likely to be required in any stages of MLOps. There are many approaches to debug, or likewise prevent ML failures to happen. We summarize the most prominent approaches next, noting that all ideas somehow relate to human-generated or -assisted tests.

- [TFX Validation](https://mlsys.org/Conferences/2019/doc/2019/167.pdf): The idea of this work is to generate a schema for the data. Failures in validating this schema either require the data to be fixed, or the schema to be changed.
- [Deequ](https://ieeexplore.ieee.org/document/8731462): Enables unit-test for data via a declarative API by combinning common quality constraints with userdefined validation code.
- [SliceLine](https://dl.acm.org/doi/10.1145/3448016.3457323): Finds problematic, potentially overlapping slices by exploiting various system specific aspects such as monotonicity properties and a linear-algebra-based enumeration algorithm on top of existing ML systems.
- [MLINSPECT](https://dl.acm.org/doi/abs/10.1145/3448016.3452759): Detects data-distribution bugs by using linage-based annotations in a ML pipline modeled as a DAG of operations.
- [Amazon SageMaker Debugger](https://proceedings.mlsys.org/paper/2021/file/d1f491a404d6854880943e5c3cd9ca25-Paper.pdf): Debugger consisting of efficient tensor processing library along with built-in rules executed in dedicated Docker containers.
- [Checklist](https://homes.cs.washington.edu/~marcotcr/acl20_checklist.pdf): Comprehensive behavioral testing of NLP models by modeling linguistic capabilities a NLP models should be able to capture.
- [Model Assertion](https://arxiv.org/pdf/2003.01668.pdf): Abstraction for providing model assertions at runtime and during training in the form of arbitrary functions that can indicate when an error is likely to have occurred.

<h2 id="mlops-additional">Additional Resources</h2>

The [Awesome MLOps GitHub Repository](https://github.com/visenger/awesome-mlops) offeres an additional, complementary view of MLOps. It is more focued on best practices and tools, opposed to our research and data-centric view of MLOps.
