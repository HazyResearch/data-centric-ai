# MLOps

The central role of data makes the development and deployment of ML/AI applications an human-in-the-loop process. 
This is a complex process in which human engineers could make mistakes, require guidance, or need to be warned when something unexpected happens. The goal of MLOps is to provide principled ways for lifecycle management, monitoring, and validation.

Researchers have started tackling these challenges by developing new techniques and building systems such as [TFX](https://arxiv.org/pdf/2010.02013.pdf), [Ease.ML](http://cidrdb.org/cidr2021/papers/cidr2021_paper26.pdf) or [Overton](https://www.cs.stanford.edu/~chrismre/papers/overton-tr.pdf) designed to handle the entire lifecycle of a machine learning model both during development and in production. These systems typically consist of distinct components in charge of handling specific stages (e.g., pre- or post-training) or aspects (e.g., monitoring or debugging) of MLOps.

We provide a more detailed overview of prominent MLOps stages or aspects with a strong research-focus in the remainder of this area, noting that this field of research is relatively young and consists of many connections to other areas of data-centric AI. Additionally, notice that most sections are inspired by well-established DevOps techniques one encounters when developing traditional software artifacts. Adopting these techniques to MLOps in a rigorous and statistical sound way is often non-trivial as one has to take into account the inherent randomness of ML tasks and its finite-sample data dependency.

Readers can refer to the [MLOps whitepaper](https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_whitepaper.pdf) from Google for a broad overview of MLOps. A canonical MLOps workflow taken from the whitepaper is shown below.


<p align="center">
  <img src="https://user-images.githubusercontent.com/5894780/133565963-f757db23-f656-4e5a-9599-fa1195ce5ead.png" alt="drawing" width="750" />
</p>



<h2 id="mlops-data-acquisition-feasibility-study">Feasibility Study and Data Acquisition</h2>

In DevOps practices, new projects typically are evaluated upon their probability of success via a feasibility study. Whilst performing such a task in the context of ML dev has some similarities such as the hardware and engineering availability, there are two different approaches to evaluate the feasibility of ML project with respect to its data.

The first one aims at inspecting the probability of success for a fixed class of models and the amount of data given. Well studied relations between the sample and model complexity in machine learning (i.e., [VC dimension](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_dimension)) have found little application in estimating the performance of ML models. [Kong and Valiant](https://arxiv.org/abs/1805.01626) estimate the best possible accuracy a model class can achieve for a fixed distribution based on a sublinear amount of data (with respect to its dimension). Being a promising approach, the method is currently only applicable to linear models. More recently, [Hashimoto](http://proceedings.mlr.press/v139/hashimoto21a/hashimoto21a.pdf) shows that assuming a simple log-linear evolution of the empirical performance of a ML model with respect to the number of training samples, one can provably predict the model performance along with potential compositions of multiple data source for simple linear models or general M-estimators. Whilst the theory does not hold for more complex models such as neural networks, the author provides some empirical evidence showing the successful application in this setting.

The second approach is model class agnostic and tackles feasibility study of a ML project by estimating the Bayes error rate (BER) of the data distribution and hence reporting the limitations in best possible accuracy which even acquiring more data could not bypass. Accurately estimating the BER with finite data is a hard problem with years of research. Recently, [Ease.ML/snoopy](http://www.vldb.org/pvldb/vol13/p2837-renggli.pdf) suggests to used the power of a simple 1-nearest-neighbor estimator on top of pre-trained embeddings to achieved faster and more accurate estimations of BER bounds without being sensitive to any distribution dependent hyper-parameters. Estimating the BER for a ML tasks is not only useful to prevent unrealistic expectations when starting to work on a new ML project, it has also its application in providing security guarantees for website fingerprinting defences (see [Cherubin's paper](https://petsymposium.org/2017/papers/issue4/paper50-2017-4-source.pdf)).


<h2 id="mlops-cicdct">CI/CD/CT</h2>

Continuous integration (CI), continuous delivery (CD) and continuous testing (CT) are well established techniques in DevOps in order to ensure safe and faster lifecycles whilst continuously updating code in production.
Various systems such as DVC's [Continuous Machine Learning (CML)](https://cml.dev/) or [MLModelCI](https://arxiv.org/abs/2006.05096) have been designed to efficiently handle the CI/CD part in ML projects. The key challenge from a ML specific viewpoint lies in runtime (latency and throughput) requirements when moving new models into production, which rely on many hyper-parameters such as the inference batch-size and underlying hardware properties. 

When continuously testing ML models one has to be careful to not be fooled by the inherent randomness of ML, nor to overfit to a testset if one plans on re-using the same dataset for testing the ML model multiple times. [Ease.ML/CI](https://mlsys.org/Conferences/2019/doc/2019/162.pdf) handles both aspects for specific test conditions (e.g., the new model has to be better than the old by a fixed number of points) from a theoretical perspective. It offers strong statistical guarantees whilst reducing the sample complexity required as much as possible. The technical challenges for efficiently adopting these techniques into a CI system are described by [Karlaš et. al.](https://dl.acm.org/doi/abs/10.1145/3394486.3403290) This [blog post](https://ds3lab.ghost.io/ci/) further described the statistical and technical challenges and how they are approached.

<h2 id="mlops-deployment-model-managemen">Deployment and Model Management</h2>

There is typically not only a single model being developed or active in production. Various online repositories such as [Hugging Face](https://huggingface.co/models), [PyTorch Hub](https://pytorch.org/hub/) or [TensorFlow Hub](https://tfhub.dev/) facilitate sharing and reusing pre-trained models. Other systems such as [ModelDB](https://dm-gatech.github.io/CS8803-Fall2018-DML-Papers/hilda-modeldb.pdf), [DVC](https://dvc.org/) or [MLFlow](https://cs.stanford.edu/~matei/papers/2018/ieee_mlflow.pdf) extend the repository functionality by further enabling version of models and dataset, tracking of experiments and efficient deployment.

Models may be deployed in the cloud to form prediction-serving systems.
Intelligent applications or services may then poll the model for predictions.
In this context of ML-as-a-service, the hosting platform must be able to 
respond to a high volume of bursty requests with very low latency.
[Clipper](https://www.usenix.org/system/files/conference/nsdi17/nsdi17-crankshaw.pdf) is an early example in this space. 

<h2 id="mlops-monitoring">Monitoring and Adaptation</h2>

It is well known that the accuracy of active models in production typically diminishes over time. The main reason for this lies in the distribution shift between the new real-time test data and the data used to train the model originally. The most prominent remedy to this problem still lies periodically (sometimes on a daily or even hourly basis) re-training models using fresh training data. This is a very costly undertaking which can be prevented by having access to so called drift detectors (also refered to as anomaly or outlier detectors). [MLDemon](https://arxiv.org/abs/2104.13621) models a human-in-the-loop approach to minimize the number of required verifications. [Klaise et. al.](https://arxiv.org/abs/2007.06299) suggest that outlier detectors should be coupled with explainable AI (XAI) techniques to help humans understand the predictions and potential distribution drift.

In an ideal world, we would want an ML system in production to automatically adapt to changes in data distribution without the need of re-training from scratch. This area is known as continual learning (CL) or lifelong learning. Merging these algorithmic ideas into a working system is non-trivial as shown by [ModelCI-e](https://arxiv.org/pdf/2106.03122.pdf) and the related work cited therein.

<h2 id="mlops-debugging">Debugging</h2>

For an ML application to be sustainable, support for debugging must exist. Debugging an ML model is likely to be necessary in any part of the MLOps stages. 
There are many approaches to debug, or likewise prevent ML failures from happening.
Unlike traditional forms of software, for which we rely on techniques like breakpoint-based cyclic debugging,
bugs in model training rarely express themselves as localized failures that raise exceptions.
Instead, a bug in model training is expressed in the loss or other metrics.
Thus, model developers cannot pause a training run to query state. Instead, they must trace the value of 
a stochastic variable over time: they must log training metrics.
Increasingly more mature systems for logging in ML are available. 
[TensorBoard](https://www.tensorflow.org/tensorboard) and [WandB](https://wandb.ai/site) are two examples.
In the event that the model developer may want to view or query more training data than they logged up-front, e.g. tensor histograms or images \& overlays,
they may add [hindsight logging](http://www.vldb.org/pvldb/vol14/p682-garcia.pdf) statements to their code post-hoc and do a fast replay from model checkpoints.

A model is just one step of the ML pipeline. Many ML pipeline bugs lie outside of the modeling stage (e.g. in data cleaning or feature generation). ML pipelines cannot be sustainable or easily debugged without some end-to-end [observability](https://arxiv.org/abs/2108.13557), or visibility into all of the steps of the pipeline. Adopting the software mindset of observability, we posit that much of the work in ML observability lies around end-to-end logging and monitoring of inputs and outputs, as well as developing query interfaces for practitioners to be able to ask questions about their ML pipeline health.

Next, we summarize prominent research directions in ML application sustainability:

- [TFX Validation](https://mlsys.org/Conferences/2019/doc/2019/167.pdf) gnerates and maintains a schema for the data. Failures in validating this schema either require the data to be fixed, or the schema to be changed.
- [Deequ](https://ieeexplore.ieee.org/document/8731462) enables unit-test for data via a declarative API by combining common quality constraints with user defined validation code.
- [SliceLine](https://dl.acm.org/doi/10.1145/3448016.3457323) finds problematic, potentially overlapping slices by exploiting various system specific aspects such as monotonicity properties and a linear-algebra-based enumeration algorithm on top of existing ML systems.
- [MLINSPECT](https://dl.acm.org/doi/abs/10.1145/3448016.3452759) detects data-distribution bugs by using linage-based annotations in a ML pipeline, which is modeled as a DAG.
- [Amazon SageMaker Debugger](https://proceedings.mlsys.org/paper/2021/file/d1f491a404d6854880943e5c3cd9ca25-Paper.pdf) consists of an efficient tensor processing library along with built-in rules executed in dedicated containers.
- [Checklist](https://homes.cs.washington.edu/~marcotcr/acl20_checklist.pdf) enables comprehensive behavioral testing of NLP models by modeling linguistic capabilities a NLP model should be able to capture.
- [Model Assertion](https://arxiv.org/pdf/2003.01668.pdf) provides an abstraction for model assertions at runtime and during training in the form of arbitrary functions that can indicate when an error is likely to have occurred.
- [FLOR](https://github.com/ucbrise/flor) Is a record-replay library designed for hindsight logging of model training.
- [mltrace](https://github.com/loglabs/mltrace) is an end-to-end observability system for ML pipelines.

<h2 id="mlops-additional">Additional Resources</h2>

The [Awesome MLOps GitHub Repository](https://github.com/visenger/awesome-mlops) offers an additional, complementary view of MLOps. It is more focused on best practices and tools, opposed to our research and data-centric view of MLOps.
