# MLOps

The central role of data makes the development and deployment of ML/AI applications an human-in-the-loop process. 
This is a complex process in which human engineers could make mistakes, require guidance, or need to be warned when something unexpected happens. The goal of MLOps is to provide principled ways for lifecycle management, monitoring, and validation.

Researchers have started tackling theses challenges by developing new techniques and building systems such as [TFX](https://arxiv.org/pdf/2010.02013.pdf) or [Ease.ML](http://cidrdb.org/cidr2021/papers/cidr2021_paper26.pdf) tailored to handle the entire lifecycle of a machine learning model both during development and in production. These systems typically constis of distinct components in charge of handling specific stages (e.g., pre- or post-training) or aspects (e.g., monitoring or debugging) of MLOps.

We provide a more detailed (incomplete) overview of some MLOps stages or aspects in the remainder of this area, noting that this field of research is relatively young and consits of many connections to other areas of data-centric AI. Additionally, notice that most sections are inspired by well-estiblished DevOps techniques one encounters when developing traditional software artifacts. Adopting these techniques to MLOps in a rigorous and statistical sound way is often non-trivial as one has the take into account the inherent randomness of ML tasks and its finite-sample data dependency.

<h2 id="mlops-data-acquisition-feasibility-study">Feasibility Study and Data Acquisition</h2>

In DevOps practices, new projects typically are evaluated upon theire probability of success via a feasibility study. Whilst performing a similar task in the context of ML dev has some similiarties such as hardware and engineering availability, there are two different approaches to evaluated the feasibility of ML project with respect to its data.

The first one aims at inspecting the probability of success for a fixed class of models. Well studied relations between the sample and model complexity in machine learning (i.e., [VC dimension](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_dimension)) have found little application in estimating the performance of ML models. [Kong and Valiant](https://arxiv.org/abs/1805.01626) estimate the best possible accuracy a model class can achieve for a fixed distribution based on a sublinear amount of data (with respect to its dimension). Beeing a promising approach, the method is currenlty only applicable to linear models. More recently, [Hashimoto](http://proceedings.mlr.press/v139/hashimoto21a/hashimoto21a.pdf) showed that assuming a simple log-linear evolution of the empirical performance of a ML model with respect to the number of training samples, one can provably prodict the model performance along with potentional compositions of multiple data source for simpel linear models or general M-estimators. Whilst the theory does not hold for more complex models such as neural networks, the author provide some empiricial evidence showing the successfull application in this setting.

The second approach is model class agnostic and tackles feasibility study of a ML project by estimating the Bayes error rate (BER) of the data distribution.


<h2 id="mlops-cicdct">CI/CD/CT</h2>

Continuous integration (CI), continuous delivery (CD) and continuous testing (CT) are well established...

<h2 id="mlops-deployment-model-managemen">Deployment & Model Management</h2>

<h2 id="mlops-monitoring">Monitoring</h2>

<h2 id="mlops-adaptation">Adapatation</h2>

<h2 id="mlops-debuggin">Debugging</h2>

Data quality...
