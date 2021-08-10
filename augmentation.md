<h1 id="sec:data-augmentation"> Data Augmentation </h1>

A key challenge when training machine learning models is collecting a large, diverse dataset that sufficiently captures the variability observed in the real world. Due to the cost of collecting and labeling datasets, data augmentation has emerged as a promising alternative. 

The central idea in data augmentation is to transform examples in the dataset in order to generate additional augmented examples that can then be added to the data. These additional examples typically increase the diversity of the data seen by the model, and provide additional supervision to the model. The foundations of data augmentation originate in [tangent propagation](https://papers.nips.cc/paper/1991/file/65658fde58ab3c2b6e5132a39fae7cb9-Paper.pdf), where model invariances were expressed by adding constraints on the derivates of the learned model.

Early successes in augmentation such as [AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) focused on inducing invariances in an image classifier by generating examples that encouraged translational or rotational invariance. These examples made augmentation a de-facto part of pipelines for a wide-ranging tasks such as image, speech and text classification, machine translation, etc. 

The choice of transformations used in augmentation is an important consideration, since it dictates the behavior and invariances learned by the model. While heuristic augmentations have remained popular, it was important to be able to control and program this augmentation pipeline carefully. [TANDA](https://arxiv.org/pdf/1709.01643.pdf) initiated a study of the problem of programming augmentation pipelines by composing a selection of data transformations. This area  has seen rapid growth in recent years with both deeper theoretical understanding and practical implementations such as [AutoAugment](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.pdf). A nascent line of work leverages conditional generative models to learn-rather than specify-these transformations, further extending this programming paradigm. 

This document provides a detailed breakdown of resources in data augmentation.

<h2 id="augmentation-history">History</h2>

Augmentation has been instrumental to achieving high-performing models since the original
[AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
paper on ILSVRC, which used random crops, translation & reflection of images for training,
and test-time augmentation for prediction.

Since then, augmentation has become a de-facto part of image training pipelines and
an integral part of text applications such as machine translation.

<h2 id="augmentation-theory">Theoretical Foundations</h2>

- [Tangent Propagation](https://papers.nips.cc/paper/1991/file/65658fde58ab3c2b6e5132a39fae7cb9-Paper.pdf) expresses desired model invariances induced by a data augmentation as tangent constraints on the directional derivatives of the learned model.
- [Kernel Theory of Data Augmentation](http://proceedings.mlr.press/v97/dao19b/dao19b.pdf) connects the tangent propagation view of data augmentation to kernel-based methods.
- [A Group-Theoretic Framework for Data Augmentation](https://arxiv.org/abs/1907.10905) develops a theoretical framework to study data augmentation, showing how it can reduce variance and improve generalization.
- [On the Generalization Effects of Linear Transformations in Data Augmentation](https://arxiv.org/abs/2005.00695) studies an over-parameterized linear regression setting and studies the generalization effect of applying a familar of linear transformations in this setting.

<h2 id="augmentation-primitives">Augmentation Primitives</h2>

### Hand-Crafted Primitives

A large body of work utilizes hand-crafted data augmentation primitives in order to improve
model performance. These hand-crafted primitives are designed based on domain knowledge
about data properties, e.g. rotating an image preserves the content of the image, and should
typically not change the class label. 

The next few sections provide a sampling of work across several different
modalities (images, text, audio) that take this approach.

#### Images

Heuristic transformations are commonly used in image augmentations, such as rotations, flips or crops
(e.g. [AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), [Inception](https://arxiv.org/abs/1409.4842.pdf)).

Recent work has proposed more sophisticated hand-crafted primitives:

- [Cutout](https://arxiv.org/abs/1708.04552) randomly masks patches of the input image during training. 
- [Mixup](https://arxiv.org/pdf/1710.09412.pdf) augments a training dataset with convex combinations of training examples. There is substantial empirical [evidence](https://papers.nips.cc/paper/2019/file/36ad8b5f42db492827016448975cc22d-Paper.pdf) that Mixup can improve generalization and adversarial robustness. A recent [theoretical analysis](https://arxiv.org/abs/2010.04819) helps explain these gains, showing that the Mixup loss can be approximated by standard ERM loss with regularization terms.  
- [CutMix](https://arxiv.org/abs/1905.04899.pdf) combines the two approaches above: instead of summing two input images (like Mixup), CutMix pastes a random patch from one image onto the other and updates the label to be weighted sum of the two image labels proportional to the size of the cutouts.
- [MixMatch](https://arxiv.org/pdf/1905.02249.pdf) and [ReMixMatch](https://arxiv.org/abs/1911.09785.pdf) extend the utility of these techniques to semi-supervised settings.

While these primitives have culminated in compelling performance gains, they can often produce unnatural images and distort image semantics. However, data augmenation techniques such as [AugMix](https://arxiv.org/abs/1912.02781) can mix together various unnatural augmentations and lead to images that appear more natural.

#### Text

Heuristic transformations for text typically involve paraphrasing text in order to produce more diverse samples.

- On a token level, synonym substitution methods replace words with their synonyms. Synonyms might be chosen based on
   - a knowledge base such as a thesaurus: e.g. [Character-level Convolutional Networks for Text Classification](https://arxiv.org/pdf/1509.01626.pdf) and [An Analysis of Simple Data Augmentation for Named Entity Recognition](https://aclanthology.org/2020.coling-main.343/)
   - neighbors in a word embedding space: e.g. [That’s So Annoying!!!](https://www.aclweb.org/anthology/D15-1306/) 
   - probable words according to a language model that takes the sentence context into account: e.g. 
  [Model-Portability Experiments for Textual Temporal Analysis](https://www.aclweb.org/anthology/P11-2047/),
  [Data Augmentation for Low-Resource Neural Machine Translation](https://www.aclweb.org/anthology/P17-2090/) and
  [Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations](https://www.aclweb.org/anthology/N18-2072/)
- Sentence parts can be reordered by manipulating the syntax tree of a sentence: e.g. [Data augmentation via dependency tree morphing for low-resource languages](https://aclanthology.org/D18-1545/)
- The whole sentence can be modified via [Backtranslation](https://aclanthology.org/P16-1009/). There a round-trip translation from a source to target language and back is used to generate a paraphrase. Examples of use include [QANet](https://arxiv.org/abs/1804.09541) and [Unsupervised Data Augmentation for Consistency Training](https://proceedings.neurips.cc/paper/2020/hash/44feb0096faa8326192570788b38c1d1-Abstract.html).


[comment]: <> (- Noising)
[comment]: <> (- Grammar induction)
[comment]: <> (- Text editing)
[comment]: <> (- Other heuristics)

#### Audio

- Vocal Tract Length Warping approaches, such as [Audio Augmentation for Speech Recognition](https://www.danielpovey.com/files/2015_interspeech_augmentation.pdf) and [Vocal Tract Length Perturbation (VTLP) improves speech recognition](http://www.cs.toronto.edu/~ndjaitly/jaitly-icml13.pdf)
- Stochastic Feature Mapping approaches, such as in [Data Augmentation for Deep Neural Network Acoustic Modeling](https://www.semanticscholar.org/paper/Data-Augmentation-for-Deep-Neural-Network-Acoustic-Cui-Goel/c083dc15b5e169e02e208b576d6991d93955b4eb)
  and [Continuous Probabilistic Transform for Voice Conversion](https://www.ee.columbia.edu/~dpwe/papers/StylCM98-vxtfm.pdf)

### Assembled Pipelines

An interesting idea is to learn augmentation pipelines, a study initiated by [TANDA](https://arxiv.org/pdf/1709.01643.pdf). This area has seen rapid growth in recent years with both deeper theoretical understanding and practical implementations like [AutoAugment](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.pdf).

The idea is to determine the right subset of augmentation primitives, and the order in which they should be applied.
These pipelines are primarily built on top of a fixed set of generic transformations.
Methods vary by the learning algorithm used, which can be

- reinforcement learning approaches led by the [TANDA](https://arxiv.org/pdf/1709.01643.pdf) work, and extended by [AutoAugment](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.pdf);
- computationally efficient algorithms for learning augmentation policies have also been proposed such as [Population-Based Augmentation](https://arxiv.org/pdf/1905.05393.pdf), [Fast AutoAugment](https://arxiv.org/pdf/1905.00397.pdf), and [Faster AutoAugment](https://arxiv.org/pdf/1911.06987.pdf);
- random sampling such as in [RandAugment](https://arxiv.org/pdf/1909.13719.pdf) and an uncertainty-based random sampling scheme such as in [Dauphin](https://arxiv.org/abs/2005.00695).

### Learned Primitives

There is substantial prior work in learning transformations that produce semantic, rather than superficial changes to an input.

One paradigm is to learn a semantically meaningful data representation, and manipulate embeddings in this representation to produce a desired transformation.

- several methods express these transformations as vector operations over embeddings, such as in
  [Deep Visual Analogy Making](https://papers.nips.cc/paper/2015/hash/e07413354875be01a996dc560274708e-Abstract.html),
  [Deep feature interpolation for image content changes](https://arxiv.org/pdf/1611.05507.pdf)
- other methods look towards manifold traversal techniques such as
  [Deep Manifold Traversal: Changing Labels with Convolutional Features](https://arxiv.org/pdf/1511.06421.pdf),
  [Learning to disentangle factors of variation with manifold interaction](http://proceedings.mlr.press/v32/reed14.pdf)
- other methods, such as [DeepAugment](https://arxiv.org/abs/2006.16241), simply use existing image-to-image models and manipulate embeddings randomly to produce diverse image outputs

Another class of approaches relies on training conditional generative models, that learn a mapping between two or more data distributions.
A prominent use case focuses on imbalanced datasets, where learned augmentations are used to generate examples for underrepresented classes or domains.
Examples of these approaches include
[BaGAN](https://arxiv.org/abs/1803.09655.pdf), [DAGAN](https://arxiv.org/abs/1711.04340.pdf), [TransferringGAN](https://arxiv.org/abs/1805.01677.pdf),
[Synthetic Examples Improve Generalization for Rare Classes](https://arxiv.org/pdf/1904.05916.pdf),
[Learning Data Manipulation for Augmentation and Weighting](https://arxiv.org/pdf/1910.12795.pdf),
[Generative Models For Deep Learning with Very Scarce Data](https://arxiv.org/abs/1903.09030.pdf),
[Adversarial Learning of General Transformations for Data Augmentation](https://arxiv.org/abs/1909.09801.pdf),
[DADA](https://arxiv.org/abs/1809.00981) and
[A Bayesian Data Augmentation Approach for Learning Deep Models](https://arxiv.org/pdf/1710.10564.pdf)

Recent approaches use a combination of learned domain translation models with consistency training to further
improve performance e.g. [Model Patching](https://arxiv.org/pdf/2008.06775.pdf).

<h2 id="augmentation-future">Future Directions</h2>

Several open questions remain in data augmentation and synthetic data generation.

- While augmentation has been found to have a strong positive effect on performance:
  what kind of augmentations maximize model robustness? How should such augmentations be specified or learned?
- Augmentation adds several sources of noise to training. The inputs are transformed or
  corrupted, and may no longer be likely to occur in the data distribution.
  The common assumption that augmentation leaves the label unmodified is often
  violated in discrete data such as text, where small changes can
  make a large impact on the label.
  What is the effect of the noise added by data augmentation?
  Can we tolerate larger amounts of noise to improve performance further?

<h2 id="augmentation-evenmore">Further Reading</h2>

- The ["Automating the Art of Data Augmentation"](https://hazyresearch.stanford.edu/data-aug-part-1)
  series of blog posts by [Sharon Li](http://pages.cs.wisc.edu/~sharonli/) provide an overview of data augmentation.
