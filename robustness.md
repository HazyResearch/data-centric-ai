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


## Certified Robustness against Adversarial Perturbations

Outside of subpopulation or domain shift, models based on standard training (i.e. empirical risk minimization (ERM)) are known to be vulnerable to carefully crafted adversarial perturbations. This vulnerability leads to security concerns for safety-critical AI applications such as autonomous vehicles (AV), where a malicious attacker can generate and add [imperceptible physical perturbations] (https://arxiv.org/abs/1707.08945) to the input data, leading to severe consequences (e.g., an AV that recognizes a stop sign as a speed limit sign and doesn't stop). 

There has been an arms race between attacks and [defenses](https://robustbench.github.io/). Empirical defenses that claim to train robust models are often broken by subsequent adaptive attacks ([example](https://arxiv.org/abs/2002.08347)). Certified defenses guarantee a lower bound on the performance of a model under certain perturbation constraints (e.g., the perturbation magnitude is bounded by a certain norm distance).

Certified robustness is jointly realized by both training and verification methods. For the common setting of a classification model against an Lp norm bounded attacker, as an example: the verification method takes individual data x0 from the test set with ground-truth label y0, and verifies the lower bound of the probability that for any perturbation \delta (||delta||_p <= eps), F(x0 + delta) = y0 always holds; the certified training method aims to train a model to improve such lower bounds.
 
Existing methods on certified robustness can be divided into two groups: deterministic methods and probabilitic approaches.

### Deterministic Approaches

Deterministic approaches usually apply relaxations to the non-linear activation functions in neural networks for verification. Common relaxations include [interval bounds](https://arxiv.org/abs/1810.12715), [linear bounds](​​https://arxiv.org/abs/1711.00851), [Zonotopes](https://files.sri.inf.ethz.ch/website/papers/sp2018.pdf), [linear programming (LP)](https://arxiv.org/abs/1902.08722) relaxation, and [semidefinite programming (SDP)](https://arxiv.org/abs/2010.11645) relaxation, ranked in ascending order of tightness and descending order of scalability for large model size. Taking the interval bounds as an example, given an input perturbation [x0 - eps, x0 + eps], the method propagates the interval [l, u] as the possible value range for each neuron’s input and output layer by layer. Similarly, for linear bounds, we propagate the linear bound [l · x + b_l, u · x + b_u], where x is the (possibly perturbed) input. Finally, the model can be certified based on the confidence score bounds of each class in the final layer: if one can certify that the lower bound of the confidence score for the ground truth class is always higher than the upper bounds of other classes, the model is certifiably robust at input x0. To provide tighter certification, one can further [combine branch-and-bound strategy with these relaxations](https://arxiv.org/abs/2103.06624), where the branch-and-bound strategy splits the nonlinear ReLU to two states (<0 or >=0) thus linearizes some ReLUs, and then solve the resulting subproblems. The complete and precise verification has been proved [NP-complete](https://arxiv.org/abs/1702.01135) (even [const-ratio relaxation is also NP-complete](https://arxiv.org/abs/1804.09699)).

The certified training methods for deterministic approaches usually leverage the bounds from verification methods: they compute an upper bound of worst-case empirical risk under perturbations from these verification methods, and either [directly minimizes the upper bound](https://arxiv.org/abs/1711.00851) or [minimizes its combination with standard clean loss](https://arxiv.org/abs/1906.06316).

### Probabilistic Approaches

Probabilistic approaches are usually based on [randomized smoothing](https://arxiv.org/abs/1902.02918). Since robustness issues can be due to the lack of smoothness, randomized smoothing takes the [majority voted class](https://arxiv.org/abs/1902.02918) (for classification) or [median/average](https://arxiv.org/abs/2007.03730) (for regression) over a set of smoothed input with noise added as the final prediction. 
The choices of the smoothing noise distribution play a critical role in how much certified robustness we can obtain. Different noise distributions are suitable for defending against perturbations bounded by different Lp norms. [This work](https://arxiv.org/abs/2002.08118) systematically studies different noise distributions.

The certified training methods for probabilistic approaches focus on improving the model’s prediction accuracy for noise corrupted inputs, where [standard noise augmented training](https://arxiv.org/abs/1902.02918), [adversarial training](https://arxiv.org/abs/1906.04584), [consistency regularization](https://arxiv.org/abs/2006.04062), and [certified radius maximization regularization](https://arxiv.org/abs/2001.02378) are popular methods.

Compared with deterministic methods, the probabilistic approaches are more flexible as it does not require knowledge about detailed neural network structure. Indeed, the probabilistic approaches have been extended to defend against various threat models (as we will introduce later) or have been further improved to be deterministic (e.g., [against patch attack](https://arxiv.org/abs/2002.10733), [deterministic L1 robustness](https://arxiv.org/abs/2103.10834)). On the other hand, for the commonly used L-infinite norm, [an intrinsic barrier](https://arxiv.org/abs/2002.08118) of these methods has been shown, and it shows that the deterministic approaches are usually tighter under certain constraints.

### Certified Robustness against Different Threat Models

Besides Lp-norm bounded attacks, other (unrestricted) threat models may be more realistic in practice, and the certified robustness against these threat models is an ongoing hot topic. Here we list some important progress.

- [TSS: against semantic transformations](https://arxiv.org/abs/2002.12398)

  TSS provides scalable certified robustness, including both certification and robust training methods, for common image transformations such as rotation, scaling, brightness change, construct change, blurring, and some of their combinations. TSS customizes the randomized smoothing for different transformations and significantly improves the certified robustness.

- [Against vector field deformations](https://arxiv.org/abs/2009.09318)

  An extension of linear relaxation provides certification against attacks that perform vector field deformations.

- [Wasserstein smoothing](https://arxiv.org/abs/1910.10783): 

  Alexander Levine and Soheil Feizi extend randomized smoothing to certifiably defend against Wasserstein distance bounded attacks.
 

- [RAB](https://arxiv.org/abs/2003.08904), [DPA](https://arxiv.org/abs/2006.14768): against poisoning and backdoor attacks

  In poisoning (backdoor) attacks, an attacker aims to manipulate the training dataset so as to either deteriorate the model performance on clean test dataset or mislead the model to be triggered by input with some specific backdoor patterns. The high-level idea of randomized smoothing, aggregating multiple model predictions, can also be applied to certifiably defend against poisoning or backdoor attacks. This time, the multiple model predictions come from multiple models trained with smoothed training dataset (RAB case) or different portions of training dataset (DPA case). 


For a detailed overview of the field of certified robustness, we refer interested readers to a [survey](https://arxiv.org/abs/2009.04131). For the current state-of-the-art, please refer to this [leaderboard](https://github.com/AI-secure/Provable-Training-and-Verification-Approaches-Towards-Robust-Neural-Networks).

### Certified Robustness in Different Learning Paradigms

- Certified Robustness in Ensemble

  Beyond single neural network models, there have also been efforts on improving the certified robustness via model ensemble, such as [RobBoost](https://arxiv.org/abs/1910.14655) and [DRT](https://arxiv.org/abs/2107.10873).

  [DRT](https://arxiv.org/abs/2107.10873) connects the certifiably robust conditions of randomized smoothing with base models’ diversity in the ensemble - when the base models are diverse enough, the adversarial transferability between base models are limited, and aggregating the base models’ predictions can lead to higher certified robustness. 
In addition, limiting the adversarial transferability between base models can also lead to higher empirical robustness against common attacks (e.g., [TRS](https://arxiv.org/abs/2104.00671) which provides the lower and upper bounds of adversarial transferability, and [DVERGE](https://arxiv.org/abs/2009.14720)). 

- Certified Robustness in Reinforcement Learning

  The certified robustness in reinforcement learning (RL) is a relatively open area. [CARRL](https://arxiv.org/abs/2004.06496) is inspired by linear relaxation based certified training and improves the empirical robustness of RL models against existing attacks. [CROP](https://arxiv.org/abs/2106.09292) systematically studies different certification criteria in RL and provides certification algorithms for each criterion correspondingly.

- Certified Robustness in Federated Learning

  Training-time attacks raise great concerns in federated learning (FL) since the local data and training process are entirely controlled by the local users. 
[CRFL](https://arxiv.org/abs/2106.08283) extends randomized smoothing to model parameter smoothing and provides the first framework to train certifiably robust FL models against backdoors. Its certifications are on three levels: feature, sample and client. 

  Another line of work is Byzantine-Robust FL where the adversarial behavior of users is modeled as Byzantine failure. Byzantine-Robust aggregation methods leverage different robust statistics, including [coordinate-wise median and trimmed mean](https://arxiv.org/abs/1803.01498), [geometric median of means](https://arxiv.org/abs/1705.05491), [approximate geometric median](https://arxiv.org/abs/1912.13445), [repeated median estimator](https://arxiv.org/abs/1912.11464) since median-based computations are more resistant to outliers than the default mean-based aggregation [FedAvg](https://arxiv.org/abs/1602.05629). These algorithms are provably robust with a focus on guaranteeing convergence rate under the Byzantine attackers.

## Robustness Through Simulated Data and Data Augmentation

Since models can degrade in the face of unusual events, we need to train models on more unusual scenarios. However, because such scenarios are rare, acquiring enough real data is infeasible. Furthermore, the future presents novel scenarios unlike those in the past, and to anticipate the future models must be tested in unusual scenarios beyond what is included in data from the past. While the Internet is vast, it does not cover all rare events nor future events. Autonomous driving companies use a mixture of real data and simulated data because large fleets do not provide enough real data to cover myriad future events. Moreover, even platforms where millions of users upload images of animals only have images of about [17%](https://www.inaturalist.org/blog/42626-we-passed-300-000-species-observed-on-inaturalist
) of all named species, demonstrating that even the Internet does not cover all visual concepts.
To address the limitation that real data poorly represents unusual, extreme, or heavy tail scenarios, simulated data and data augmentation can help.

When researchers test robustness to distribution shift, they may expose models to [corruptions](https://github.com/hendrycks/robustness/) such as snowfall or novel [object renditions](https://github.com/hendrycks/imagenet-r) such as cartoons. Some data augmentation techniques make models more robust to these types of distribution shifts:

-	[AugMix](https://arxiv.org/abs/1912.02781) augments data by randomly composing simple augmentations such as rotations and translations, and then it takes a convex combination of these simple augmentations. These simple operations are enough to create examples that are diverse enough to teach models to be more robust to complicated corruptions such as snow or even object renditions. Consequently some forms data augmentation can provide enough variability and stressors that can teach models to generalize to distinct distribution shifts.

-	[DeepAugment](https://arxiv.org/abs/2006.16241) uses neural networks to create more diverse and complicated data augmentations. Rather than perturb the image itself with primitives such as rotations, DeepAugment augments images by perturbing internal representations of deep networks that encode the image. The procedure is to feed an image through an image-to-image network (such as a denoising neural network) but perturb the hidden features of the network. As a consequence of perturbing the internal perturbations, the image that the model outputs is distorted and can serve as augmented data. This augmentation technique teaches models to generalize better to unseen corruption and rendition distribution shifts.

Adversarial robustness can also be enhanced by data augmentation and simulated data. For example, AugMix can improve robustness to [unexpected adversarial attacks](https://arxiv.org/abs/1908.08016) more effectively than a 1000x increase in training data. Data augmentation, when carefully applied, can also [improve](https://arxiv.org/abs/2103.01946) l_infty adversarial robustness. Moreover, completely simulated data from diffusion models can markedly [boost adversarial robustness](https://arxiv.org/abs/2103.01946). These findings establish simulated data and data augmentation as a primary tool for improving model robustness.
