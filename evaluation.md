<h1 id="sec:evaluation">Evaluation</h1>

Model evaluation is a crucial part of the model development process in machine learning. The goal of evaluation is to understand the quality of a model, and anticipate if it will perform well in the future.

While evaluation is a classical problem in machine learning, data-centric AI approaches have catalyzed a shift towards _fine-grained evaluation_: moving beyond standard measures of average performance such as accuracy and F1 scores, to measuring performance on particular populations of interest. This enables a more granular understanding of model performance, and gives users a clearer idea of model capabilities. This shift is complentary to a growing interest in understanding model robustness, since access to fine-grained evaluation permits an enhanced ability to build more robust models.

Approaches to fine-grained evaluation include measuring performance on critical data subsets called slices, invariance or sensitivity to data transformations, and resistance to adversarial perturbations. While most evaluation is user-specified, an important line of work found that models often underperform on _hidden strata_ that are missed by model builders in evaluation, which can have profound consequences on our ability to deploy and use models. This motivates future work in automatically discovering these hidden strata, or more generally, finding all possible failure modes of a model by analyzing datasets and models in systematically in conjunction.

Another important facet of fine-grained evaluation is data and model monitoring in order to anticipate, measure and mitigate degradations in performance due to distribution shift. This includes identifying and isolating data points that may be considered outliers, estimating performance on unlabeled data that is streaming to a deployed model, and generating rich summaries of how the data distribution may be shifting over time. 

## Slice-Based Evaluation

A common evaluation paradigm is to _slice_ data in order to find important, fine-grained data
populations where models underperform, and then use this information to debug and repair models.

There are a variety of tools that assist in performing slice-based evaluations, as well as
open questions in how to discover and select the most important and relevant slices for a
given context.

[comment]: <> (### Foundations)

#### Slice Specification and Discovery

How are important slices of data found? Since it's impossible to enumerate all slices that
we might care about, slice discovery uses a combination of domain-expertise
about what factors of variation in the data are important for a particular problem, along with
automated methods to sift through the space of possible slices.

Manual methods for slice specification include,

- Metadata-based filtering, which can be used to test whether a model is disciminatory or biased
  across important data characteristics e.g. race.
- Slicing functions, programmatic functions that can be written by domain experts and practitioners
  to identify important slices of data e.g. `len(sentence) > 10`.

Automated methods for slice discovery include,

- [SliceFinder](https://research.google/pubs/pub47966/) is an interactive framework
  for finding interpretable slices of data.
- [SliceLine](https://mboehm7.github.io/resources/sigmod2021b_sliceline.pdf) uses a fast slice-enumeration method to make the process of slice discovery efficient and parallelizable.
- [GEORGE](https://arxiv.org/pdf/2011.12945.pdf) uses standard approaches to cluster representations of a deep model in order to discover underperforming subgroups of data.
- [Multiaccuracy Audit](https://arxiv.org/abs/1805.12317) is a model-agnostic approach that searches for slices on which the model performs poorly by training a simple "auditor" model to predict the full model's residual from input features. This idea of fitting a simple model to predict the predictions of the full model is also used in the context of [explainable ML](https://arxiv.org/pdf/1910.07969.pdf).

Future directions for slice discovery will continue to improve our understanding of how to find slices
that are interpretable, task-relevant, error-prone and susceptible to distribution shift.

#### Selecting Important Slices

Which slices are most relevant among a large set of given slices?
Slice selection is particularly important if the slices are used
as input to a process or algorithm whose behavior or runtime scales with the slices.

Example use cases include

- robust training with a subset of important slices in order to improve performance (e.g. [GroupDRO](https://arxiv.org/pdf/1911.08731.pdf))
- slice-based learning where model representation is explicitly allocated for selected slices (e.g. [SBL](https://arxiv.org/abs/1909.06349))
- slice-based importance weighting, where only slices relevant to distribution shift should be kept (e.g. Mandoline)

### Tools

- [Robustness Gym](https://github.com/robustness-gym/robustness-gym) is an evaluation toolkit
  for machine learning models that unifies several evaluation paradigms
  (e.g. subpopulations, transformations)
  using a common set of abstractions, along with standardized reporting.
- [Errudite](https://www.aclweb.org/anthology/P19-1073.pdf) is an interactive tool for performing
  reproducible error analyses in a domain-specific language that supports explicitly testing hypotheses
  through counterfactual examples.

### Applications

Slice-based analyses are commonplace and have been performed across all modalities with varying
degrees of thoroughness.

- Recent work on [hidden stratification](https://arxiv.org/pdf/1909.12475.pdf) describes the problem of subpar performance on hidden strata -- unlabeled subclasses of the data that are semantically and practically meaningful.
- The [GLUE](https://openreview.net/pdf?id=rJ4km2R5t7) benchmark paper describes important
  slices for text classification, which were used to decide what may be challenging examples for GLUE.
- The [Robustness Gym](https://arxiv.org/abs/2101.04840.pdf) paper contains slice-based analyses
  across several NLP tasks (natural language inference, named entity linking, text summarization).
- Subgroup robustness work typically specify slices.

[comment]: <> (Another approach to understand )

## Benchmarking

Benchmarking is a common practice for quantifying progress in the machine learning community and
comparing models across tasks and applications.

Recent work has shifted towards data-driven benchmarking, where instead of submitting models, participants can submit datasets to understand the impact of the data on training. A prominent example is the [Data-Centric AI Competition](https://https-deeplearning-ai.github.io/data-centric-comp/).

Similarly, [Dynabench](https://dynabench.org/) addresses problems inherent to traditional benchmarks such as overfitting and saturation by providing an platform for dynamic, human-in-the-loop dataset collection and model benchmarking.

Other benchmarking efforts have focused on enabling more robust model comparisons by (1) reporting richer benchmarking data such as evaluation metrics beyond accuracy (e.g. fairness scores) and metrics of practical concern (e.g. compute) and by (2) providing tools for fine-grained model comparisons. Some examples are as follows:

- [Dynaboard](https://ai.facebook.com/blog/dynaboard-moving-beyond-accuracy-to-holistic-model-evaluation-in-nlp/): an evaluation-as-a-service interface for comparing models across a holistic set of evaluation criteria including accuracy, robustness, fairness, compute, and memory.
- [ExplainaBoard](http://explainaboard.nlpedia.ai/): an interactive leaderboard and evaluation software for fine-grained model comparisons.
- [Ludwig Benchmarking Toolkit](https://github.com/HazyResearch/ludwig-benchmarking-toolkit): a personalized benchmarking toolkit for running multi-objective, standardized, and configurable benchmarking studies.
- [Experiment Impact Tracker](https://github.com/Breakend/experiment-impact-tracker): a toolkit for tracking model energy usage, carbon emissions, and compute utilization.

<h2 id="evaluation-on-unlabeled-data">Evaluation on Unlabeled Data</h2>
A key part of robustness is *monitoring* the data in order to track when the data distribution
has shifted and take remedial action. Since the model makes predictions on unlabeled data,
standard validation cannot be used due to the absence of labels.

Instead, the problem is one of statistical estimation,
with techniques for direct performance estimation that rely on importance weighting, and active sampling
methods that use a small target sample to estimate performance.

### Estimating Target Performance

Models are often deployed on an unlabeled target dataset different from the labeled source data they were trained and validated on. To efficiently check how the model performs on the target dataset, importance weighting by the distributions' density ratio is used to correct for distribution shift.
A variety of importance weighting methods are popular in the literature.

Below are some resources on distribution shift, importance weighting, and density ratio estimation:

- [Density Ratio Estimation for Machine Learning](https://www.cambridge.org/core/books/density-ratio-estimation-in-machine-learning/BCBEA6AEAADD66569B1E85DDDEAA7648) explains the different approaches to estimate density ratios, a key technical step in computing importance weights.
- [Art Owen's notes on importance sampling](https://statweb.stanford.edu/~owen/mc/Ch-var-is.pdf) provides an overview of importance weighting with connections to Monte Carlo theory.
- [CS329D at Stanford: ML Under Distribution Shifts](https://thashim.github.io/cs329D-spring2021/) covers current research in distribution shift, ranging from covariate shift to adversarial robustness.
- [Propensity Scores](https://academic.oup.com/biomet/article/70/1/41/240879) are used in observational studies for correcting disparities when evaluating treatment on a target population given that the treatment was applied to a set of potentially biased subjects.
- [Learning Bounds on Importance Weighting](https://papers.nips.cc/paper/2010/file/59c33016884a62116be975a9bb8257e3-Paper.pdf): how well importance weighting corrects for distribution shift can be attributed to the variance of the weights, or alternatively the R\'enyi divergence between source and target.
- Importance weighting works poorly when the supports of the source and target do not overlap and when data is high-dimensional. [Mandoline](https://mayeechen.github.io/files/mandoline.pdf) addresses this by reweighting based on user/model-defined ``slices'' that intend to capture relevant axes of distribution shift. Slices are often readily available as subpopulations identified by the practitioner, but can also be based on things like metadata and the trained model's scores.

<h2 id="outlier-detection">Outlier Detection</h2>

As an important task in open-world learning, out-of-distribution (OOD) detection refers to detecting whether or not an input comes from the training distribution. It has been shown that both [discriminative](https://arxiv.org/abs/1412.1897) and [generative](https://arxiv.org/abs/1810.09136) models can assign high confience predictions for OOD data. Here are some recent works that try to tackle different aspects of OOD detection:

- New OOD scores: besides using the [maximum softmax probability](https://arxiv.org/abs/1610.02136) from a pre-trained network for OOD detection, recent works propose new OOD scores to improve the OOD uncertainty estimation such as [the calibrated softmax score](https://arxiv.org/abs/1706.02690), [generalized ODIN](https://arxiv.org/abs/2002.11297), [the Mahalanobis distance-based confidence score](https://arxiv.org/abs/1807.03888), and [energy score](https://arxiv.org/abs/2010.03759).

- Robustness of OOD detection: [This paper](https://arxiv.org/pdf/1909.12180.pdf) provides the first provable guarantees for wort-case OOD detection on some balls around uniform noise. [ATOM](https://arxiv.org/abs/2006.15207) shows that by effectively mining informative auxiliary OOD data to improve the decision boundary of the OOD detector, one can boost OOD detection performance as well as adversarial robustness.

- Computationally efficient and Large-scale OOD detection: In contrast to most works that rely on final-layer outputs for OOD detection, [MOOD](https://arxiv.org/abs/2104.14726) first exploits intermediate classifier outputs for dynamic and efficient OOD inference. [MOS](https://arxiv.org/abs/2105.01879) proposes a group-based OOD detection framework with a new OOD scoring function for large-scale image classification. The main idea is to decompose the large semantic space into smaller groups with similar concepts , which allows simplifying the decision boundaries between in- vs. OOD data for effective OOD detection.

### Active Sampling and Labeling

_This section is a stub. You can help by improving it._
