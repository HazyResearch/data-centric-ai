# Robustness

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