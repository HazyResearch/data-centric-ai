# Data Centric AI (v0.0.1)

We're collecting (an admittedly opinionated) list of resources and progress made
in data-centric AI, with exciting directions past and present.

While AI has been pretty focused on models, the real-world experience of those who put
models into production is that the data often matters more. The goal of this repository
is to consolidate this experience in a single place that can be accessed by anyone that 
wants to understand and contribute to this area.

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
2. [Data Augmentation](#data-representations--self-supervision)
3. [Self-Supervision](#data-representations--self-supervision)
4. [The End of Modelitus](#go-big-or-go-home)
5. [Fine-Grained Evaluation](#fine-grained-evaluation)
6. [Robustness](#robustness)
7. [Applications](#section-applications)

<h1 id="sec:data-programming">Data Programming & Weak Supervision</h1>

[Data Programming & Weak Supervision Area Page](data-programming.md)

Many modern machine learning systems require large, labeled datasets to be successful but producing such datasets is time-consuming and expensive. Instead, weaker sources of supervision, such as [crowdsourcing](https://papers.nips.cc/paper/2011/file/c667d53acd899a97a85de0c201ba99be-Paper.pdf), [distant supervision](https://www.aclweb.org/anthology/P09-1113.pdf), and domain experts' heuristics like [Hearst Patterns](https://people.ischool.berkeley.edu/~hearst/papers/coling92.pdf) have been used since the 90s.

However, these were largely regarded by AI and AI/ML folks as ad hoc or isolated techniques. The effort to unify and combine these into a data centric viewpoint started in earnest with [data programming](https://arxiv.org/pdf/1605.07723.pdf) embodied in the [Snorkel system](http://www.vldb.org/pvldb/vol11/p269-ratner.pdf), now an [open-source project](http://snorkel.org) and [thriving company](http://snorkel.ai). In Snorkel's conception, users specify multiple labeling functions that each represent a noisy estimate of the ground-truth label. Because these labeling functions vary in accuracy, coverage of the dataset, and may even be correlated, they are combined and denoised via a latent variable graphical model. The technical challenge is thus to learn accuracy and correlation parameters in this model, and to use them to infer the true label to be used for downstream tasks.

Data programming builds on a long line of work on parameter estimation in latent variable graphical models. Concretely, a generative model for the joint distribution of labeling functions and the unobserved (latent) true label is learned. This label model permits aggregation of diverse sources of signal, while allowing them to have varying accuracies and potential correlations.

An overview of the weak supervision pipeline can be found in this [Snorkel blog post](https://www.snorkel.org/blog/weak-supervision), including how it compares to other approaches to get more labeled data and the technical modeling challenges. These [Stanford CS229 lecture notes](https://mayeechen.github.io/files/wslecturenotes.pdf) provide a theoretical summary of how graphical models are used in weak supervision.


<h1 id="sec:augmentation">Data Augmentation</h1>

[Data Augmentation Area Page](augmentation.md)

A key challenge when training machine learning models is collecting a large, diverse dataset that sufficiently captures the variability observed in the real world. Due to the cost of collecting and labeling datasets, data augmentation has emerged as a cheap, promising alternative. 

The central idea in data augmentation is to transform examples in an existing dataset to generate additional augmented examples that can then be added to the dataset. These additional examples typically increase the diversity of the data seen by the model, and provide additional supervision to the model. The foundations of data augmentation originate in [tangent propagation](https://papers.nips.cc/paper/1991/file/65658fde58ab3c2b6e5132a39fae7cb9-Paper.pdf), which introduced techniques to make a learned model invariant with respect to some transformation of the data.

Early successes in augmentation such as [AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) focused on inducing invariances in an image classifier by generating examples that encouraged translational or rotational invariance. These successes made augmentation a de-facto part of pipelines for a wide-ranging set of tasks such as image, speech and text classification, machine translation, etc. 

The choice of transformations used in augmentation is an important consideration, since it dictates the invariances learned by the model, and its behavior when encountering a diversity of test examples. While heuristic augmentations have remained popular, it is important to be able to control and program the augmentation pipeline more carefully. [TANDA](https://arxiv.org/pdf/1709.01643.pdf.) initiated a study of the problem of programming augmentation pipelines by composing a selection of data transformations. This area has since seen rapid growth with both deeper theoretical understanding and practical implementations such as [AutoAugment](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.pdf). A nascent line of work has leveraged conditional generative models to learn-rather than specify-these transformations, further extending this programming paradigm. 


<h1 id="sec:representation">Self-Supervision</h1>

[Self-Supervision Area Page](self-supervision.md)

The need for large, labeled datasets has motivated methods to pre-train latent representations of the input space using unlabeled data and use the now knowledge-rich representations in downstream tasks. As the representations allow for knowledge transfer to downstream tasks, these tasks require less labeled data. For example, language models can be pre-trained to predict the next token in a textual input to learn representations of words or sub-tokens. These word representations are then used in downstream models such as sentiment classification. This paradigm, called "self-supervision", has revolutionized how we train (and pre-train) models. Importantly, these self-supervised pre-trained models learn without manual labels or hand curated features. This reduces the engineer effort to create and maintain features and makes models significantly easier to deploy and maintain. This shift has allowed for more data to be fed to the model and shifted the focus to understanding what data to use.


<h1 id="sec:end_modelitus">The End of Modelitus</h1>

[The End of Modelitus Area Page](end_of_modelitus.md)

With the ability to train models on unlabelled data, research is scaling up both data size and model size at an [impressive rate](https://medium.com/analytics-vidhya/openai-gpt-3-language-models-are-few-shot-learners-82531b3d3122). With access to such massive amounts of data, the question shifted from “how to construct the best model” to “how do you feed these models”. And as [Kaplan et al](https://arxiv.org/pdf/2001.08361.pdf) showed, the architecture matters less; the real lift comes from the data.

<h1 id="sec:evaluation">Evaluation</h1>

[Evaluation Area Page](evaluation.md)


<h1 id="sec:robustness">Robustness</h1>  

[Robustness Area Page](robustness.md)

Machine learning subscribes to a simple idea: models perform well on data that “look” or “behave” similarly to data that they were trained on - in other words, the test distributions are encountered and learned during training.

- In practice, collecting enough training data to account for all potential deployment scenarios is infeasible. With standard training (i.e. empirical risk minimization (ERM)), this can lead to poor robustness. Current ML systems may fail when encountering out-of-distribution data.
- More fundamentally, this lack of robustness also sheds light on the limitations with how we collect data and train models. Training only with respect to statistical averages can lead to models learning the "wrong" things, such as spurious correlations and dependencies on confounding variables that hold for most, but not all, of the data.

How can we obtain models that perform well on many possible distributions and tasks, especially in realistic scenarios that come from deploying models in practice? This is a broad question and a big undertaking. 

An important area of interest is subgroup robustness or [hidden](https://hazyresearch.stanford.edu/hidden-stratification) [stratification](https://www.youtube.com/watch?v=_4gn7ibByAc). Standard classification assigns a single label for each sample in the dataset, and trains a model to correctly predict those labels. However, several distinct data subsets or "subgroups" might exist among datapoints that all share the same label, and these labels may only coarsely describe the meaningful variation within the population.

- In real-world settings such as [medical](https://dl.acm.org/doi/pdf/10.1145/3368555.3384468) [imaging](https://lukeoakdenrayner.wordpress.com/2019/10/14/improving-medical-ai-safety-by-addressing-hidden-stratification/), models trained on the entire training data can obtain low average error on a similarly-distributed test set, but surprisingly high error on certain subgroups, even if these subgroups' distributions were encountered during training.
- Frequently, what also separates these underperfoming subgroups from traditional
  ones in the noisy data sense is that there exists a true dependency between the subgroup features and labels - the model just isn't learning it.

Towards overcoming hidden stratification, recent work such as [GEORGE](https://www.youtube.com/watch?v=ZXHGx52yKDM) observes that modern machine learning methods also learn these "hidden" differences between subgroups as hidden layer representations with supervised learning, even if no subgroup labels are provided.

<h1 id="sec:end_modelitus">Emerging Trends</h1>

[Emerging Trends Area Page](emerging.md)

Data-centric AI is still growing, and we want to capture emerging trends as they arise. Some new areas we think are forming involve interactive machine learning, massice scale models, and observational ML. Take a look at the area page.

<h1 id="sec:applications">Applications</h1>

[Applications Area Page](applications.md)

Data-centric approaches have had a wide-ranging impact wherever machine learning is used and deployed, whether in academia, industry or other organizations. Impact spans modalities such as structured data, text, images, videos, graphs and others, while areas include text and image processing, medical imaging, computational biology, autonomous driving, etc.
