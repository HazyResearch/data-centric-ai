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
