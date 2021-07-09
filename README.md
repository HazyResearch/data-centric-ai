# Data Centric AI (v0.0.1)

We're collecting (an admittedly opinionated) list of resources and progress made
in data-centric AI, with exciting directions past and present.

While AI has been pretty focused on models, the real-world experience of those who put
models into production is that the data often matters more.

---

## Contributing

We want this resource to grow with contributions from readers and data enthusiasts.
Make a pull request if you want to add resources.

Instructions for adding resources:

[comment]: <> (0. Add potential emoji to section header &#40;syntax `:emoji:`&#41;. )
[comment]: <> ( The emoji can be made up and may not exist &#40;yet&#41;.)

1. Write a sentence summarizing the content of the section you're writing.
   Feel free to change the header name to something more appropriate
   and/or split your section across multiple headers if that makes sense to you.
2. Add a few critical links (citations, paper links, blogs, videos, code, workshops,
   classes, tutorials, figures, pictures, recipes, tweets, books), and a short
   description of what the content is about and how it relates to the section.
3. If you added any sections with `h1` or `h2` headers, add them to the table of contents.
   Use `<h1 id="my-h1-section"></h1>` or `<h2 id="my-h2-section"></h2>` to tag sections
   instead of standard markdown hashes.

---

# Table of Contents

1. [Data Programming & Weak Supervision](data-programming.md)
   1. [Key Papers](data-programming.md#data-programming-key-papers)
   2. [Techniques](data-programming.md#data-programming-techniques)
   3. [Foundations](data-programming.md#data-programming-foundations)
   4. [Other Resources](data-programming.md#data-programming-resources)
   5. [Success Stories](data-programming.md#weak-supervision-success-stories)
2. [Data Representations & Self-Supervision](#data-representations--self-supervision)
   1. [Embeddings](#embeddings)
   2. [Learning with Auxiliary Information](#learning-with-auxiliary-information)
   3. [Success Stories](#data-representation-successes)
3. [Go Big or Go Home](#go-big-or-go-home)
   1. [Universal Models](#universal-models)
   2. [Efficient Models](#efficient-models)
   3. [Interactive Machine Learning](#interactive-machine-learning)
4. [Data Augmentation](#data-augmentation)
   1. [History](#augmentation-history)
   2. [Theoretical Foundations](#augmentation-theory)
   3. [Augmentation Primitives](#augmentation-primitives)
   4. [Future Directions](#augmentation-future)
   5. [Further Reading](#augmentation-evenmore)
5. [Contrastive Learning](#contrastive-learning)
   1. [Theoretical Foundations](#contrastive-theory)
   2. [Applications](#contrastive-applications)
6. [Fine-Grained Evaluation](#fine-grained-evaluation)
   1. [Slice-Based Evaluation](#slice-based-evaluation)
   2. [Benchmarking](#benchmarking)
7. [Robustness](#robustness)
   1. [Subgroup Information](#subgroup-information)
   2. [Evaluation on Unlabeled Data](#evaluation-on-unlabeled-data)
   3. [Outlier Detection](#outlier-detection)
8. [Applications](#section-applications)
   1. [Named Entity Linking](#named-entity-linking)
   2. [Medical Imaging](#medical-imaging)
   3. [Computational Biology](#computational-biology)
   4. [Observational Supervision](#observational-supervision)

# Data Programming & Weak Supervision

[Data Programming & Weak Supervision Area Page](data-programming.md)

Many modern machine learning systems require large, labeled datasets to be successful but producing such datasets is time-consuming and expensive. Instead, weaker sources of supervision, such as [crowdsourcing](https://papers.nips.cc/paper/2011/file/c667d53acd899a97a85de0c201ba99be-Paper.pdf), [distant supervision](https://www.aclweb.org/anthology/P09-1113.pdf), and domain experts' heuristics like [Hearst Patterns](https://people.ischool.berkeley.edu/~hearst/papers/coling92.pdf) have been used since the 90s.

However, these were largely regarded by AI and AI/ML folks as ad hoc or isolated techniques. The effort to unify and combine these into a data centric viewpoint started in earnest with [data programming](https://arxiv.org/pdf/1605.07723.pdf) embodied in the [Snorkel system](http://www.vldb.org/pvldb/vol11/p269-ratner.pdf), now an [open-source project](http://snorkel.org) and [thriving company](http://snorkel.ai). In Snorkel's conception, users specify multiple labeling functions that each represent a noisy estimate of the ground-truth label. Because these labeling functions vary in accuracy, coverage of the dataset, and may even be correlated, they are combined and denoised via a latent variable graphical model. The technical challenge is thus to learn accuracy and correlation parameters in this model, and to use them to infer the true label to be used for downstream tasks.

Data programming builds on a long line of work on parameter estimation in latent variable graphical models. Concretely, a generative model for the joint distribution of labeling functions and the unobserved (latent) true label is learned. This label model permits aggregation of diverse sources of signal, while allowing them to have varying accuracies and potential correlations.

An overview of the weak supervision pipeline can be found in this [Snorkel blog post](https://www.snorkel.org/blog/weak-supervision), including how it compares to other approaches to get more labeled data and the technical modeling challenges. These [Stanford CS229 lecture notes](https://mayeechen.github.io/files/wslecturenotes.pdf) provide a theoretical summary of how graphical models are used in weak supervision.

# Data Representations & Self-Supervision

[Data Representation & Self-Supervision Area Page](data-representation.md)

# Data Augmentation

[Data Augmentation Area Page](augmentation.md)

A key challenge when training machine learning models is collecting a large, diverse dataset that sufficiently captures the variability observed in the real world. Due to the cost of collecting and labeling datasets, data augmentation has emerged as a promising alternative. 

The central idea in data augmentation is to transform examples in the dataset in order to generate additional augmented examples that can then be added to the data. These additional examples typically increase the diversity of the data seen by the model, and provide additional supervision to the model. The foundations of data augmentation originate in [tangent propagation](https://papers.nips.cc/paper/1991/file/65658fde58ab3c2b6e5132a39fae7cb9-Paper.pdf), where model invariances were expressed by adding constraints on the derivates of the learned model.

Early successes in augmentation such as [AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) focused on inducing invariances in an image classifier by generating examples that encouraged translational or rotational invariance. These examples made augmentation a de-facto part of pipelines for a wide-ranging tasks such as image, speech and text classification, machine translation, etc. 

The choice of transformations used in augmentation is an important consideration, since it dictates the behavior and invariances learned by the model. While heuristic augmentations have remained popular, it was important to be able to control and program this augmentation pipeline carefully. [TANDA](https://arxiv.org/pdf/1709.01643.pdf.) initiated a study of the problem of programming augmentation pipelines by composing a selection of data transformations. This area  has seen rapid growth in recent years with both deeper theoretical understanding and practical implementations such as [AutoAugment](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.pdf). A nascent line of work leverages conditional generative models to learn-rather than specify-these transformations, further extending this programming paradigm. 


# Contrastive Learning

<h2 id="contrastive-theory">Theoretical Foundations</h2>

Contrastive learning works by optimizing a loss function that pulls together similar points ("positive" pairs) and pushes apart dissimilar points ("negative" pairs). Compared to its empirical success in self-supervision, a theoretical understanding of contrastive learning is relatively lacking in terms of what sort of representations are learned by minimizing contrastive loss, and what these representations guarantee on downstream tasks.

- [Representations induced on the hypersphere](https://arxiv.org/pdf/2005.10242.pdf): assuming that the representations to learn are constrained to a hypersphere, the contrastive loss function is closely connected to optimizing for "alignment" (positive pairs map to the same representation) and "uniformity" (representations are "spread out" as much as possible on the hypersphere to maintain as much as information as possible).
- [Downstream performance](https://arxiv.org/pdf/1902.09229.pdf): suppose that similar pairs belong to the same latent subclass, and that the downstream task aims to classify among some of these latent subclasses. Then, downstream loss of a linear classifier constructed using mean representations can be expressed in terms of the unsupervised contrastive loss.
- [Debiasing contrastive learning](https://arxiv.org/pdf/2007.00224.pdf) and [using hard negative samples](https://openreview.net/pdf?id=CR1XOQ0UTh-): in unsupervised settings, negative pairs are constructed by selecting two points at random i.i.d. This can result in the two points actually belonging to the same latent subclass, but this can be corrected via importance weighting. Moreover, even within different latent subclasses, some negative samples can be "harder" than others and enforce better representations.

<h2 id="contrastive-applications">Applications</h2>

# Fine-Grained Evaluation

Models are typically evaluated using average performance, e.g. average accuracy or F1 scores.
These metrics hide when a model is particularly poor on an important slice of data or if it's
decisions are made for the wrong reasons, which may hurt generalization.

To better understand how models perform when deployed in real-world settings,
tools for fine-grained analysis and efficient methods for handling unlabeled
validation/test data are needed in evaluation.

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
- [SliceLine](https://mboehm7.github.io/resources/sigmod2021b_sliceline.pdf) uses a fast slice-enumeration
  method to make the process of slice discovery efficient and parallelizable.
- [GEORGE](https://arxiv.org/pdf/2011.12945.pdf), which uses standard approaches to cluster representations
  of a deep model in order to discover underperforming subgroups of data.

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
- [Experiment Impact Tracker])(https://github.com/Breakend/experiment-impact-tracker): a toolkit for tracking model energy usage, carbon emissions, and compute utilization.

[comment]: <> ([Avanika])
[comment]: <> (## Robustness [Jared, Michael])
[comment]: <> (- Hidden Stratification + GEORGE)

<h1 id="robustness">Robustness</h2>  
NEED ROBUSTNESS SETUP/HIDDEN STRAT
label drift whatever
Hidden straification point to talks! This is a huge thing, it came first--and it shouldn't be some after thought as "application".
Then, put all that work in context.
Tell the stor!

Machine learning subscribes to a simple idea: models perform well on data that “look” or “behave” similarly to data that they were trained on - in other words, the test distributions are encountered and learned during training.

- In practice, though, collecting enough training data to account for all potential deployment scenarios is infeasible. With standard training (i.e. empirical risk minimization (ERM)), this can lead to poor ML robustness; current ML systems may fail when encountering out-of-distribution data.
- More fundamentally, this lack of robustness also sheds light on the limitations with how we collect data and train models. Training only with respect to statistical averages can lead to models learning the "wrong" things, such as spurious correlations and dependencies on confounding variables that hold for most, but not all, of the data.

How can we obtain models that perform well on many possible distributions and tasks, especially in realistic scenarios that come from deploying models in practice? This is a broad question and a big undertaking. We've therefore been interested in building on both the frameworks and problem settings that allow us to model and address robustness in tractable ways, and the methods to improve robustness in these frameworks.

One area we find particularly interesting is that of subgroup robustness or [hidden](https://hazyresearch.stanford.edu/hidden-stratification) [stratification](https://www.youtube.com/watch?v=_4gn7ibByAc). With standard classification, we assign a single label for each sample in our dataset, and train a model to correctly predict those labels. However, several distinct data subsets or "subgroups" might exist among datapoints that all share the same label, and these labels may only coarsely describe the meaningful variation within the population.

- In real-world settings such as [medical](https://dl.acm.org/doi/pdf/10.1145/3368555.3384468) [imaging](https://lukeoakdenrayner.wordpress.com/2019/10/14/improving-medical-ai-safety-by-addressing-hidden-stratification/), models trained on the entire training data can obtain low average error on a similarly-distributed test set, but surprisingly high error on certain subgroups, even if these subgroups' distributions were encountered during training.
- Frequently, what also separates these underperfoming subgroups from traditional
  ones in the noisy data sense is that there exists a true dependency between the subgroup features and labels - the model just isn't learning it.

Towards overcoming hidden stratification, recent work such as [GEORGE](https://www.youtube.com/watch?v=ZXHGx52yKDM) observes that modern machine learning methods also learn these "hidden" differences between subgroups as hidden layer representations with supervised learning, even if no subgroup labels are provided.

<h2 id="subgroup-information">Improving Robustness with Subgroup Information</h2>

Framed another way, a data subset or "subgroup" may carry spurious correlations between its features and labels that do not hold for datapoints outside of the subgroup. When certain subgroups are larger than others, models trained to minimize average error are susceptible to learning these spurious correlations and performing poorly on the minority subgroups.

To obtain good performance on _all_ subgroups, in addition to the ground-truth labels we can bring in subgroup information during training.

- [Group Distributionally Robust Optimization (Group DRO)](https://arxiv.org/abs/1911.08731) assumes knowledge of which subgroup each training sample belongs to, and proposes a training algorithm that reweights the loss objective to focus on subgroups with higher error.
- [Model Patching](https://arxiv.org/abs/2008.06775) uses a generative model to synthesize samples from certain subgroups as if they belonged to another. These augmentations can then correct for subgroup imbalance, such that training on the new dataset mitigates learning correlations that only hold for the original majority subgroups.

Subgroup information also does not need to be explicitly annotated or known. Several recent works aim to first infer subgroups before using a robust training method to obtain good performance on all subgroups. A frequent heuristic is to use the above observation that models trained with empirical risk minimization (ERM) and that minimize average error may still perform poorly on minority subgroups; one can then infer minority or majority subgroups depending on if the trained ERM model correctly predicts the datapoints.

- [Learning from Failure (LfF)](https://arxiv.org/abs/2007.02561) trains two models in tandem. Each model trains on the same data batches, where for each batch, datapoints that the first model gets incorrect are upweighted in the loss objective for the second model.
- [Just Train Twice (JTT)]() trains an initial ERM model for a few epochs, identifies the datapoints this model gets incorrect after training, and trains a new model with ERM on the same dataset but with the incorrect points upsampled.
- [Correct-N-Contrast (CNC)]() also trains an initial ERM model, but uses supervised contrastive learning to train a new model to learn similar representations for datapoints with the same class but different trained ERM model predictions.

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

[comment]: <> (## Video)

[comment]: <> ([Dan])

<h1 id="section-applications">Applications</h1>
Data-centric approaches have had a wide-ranging impact wherever machine learning is used and deployed, whether in academia, industry or other organizations. Impact spans modalities such as structured data, text, images, videos, graphs and others, while areas include text and image processing, medical imaging, computational biology, autonomous driving, etc.
