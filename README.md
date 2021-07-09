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


<h1 id="sec:data-programming">Data Programming & Weak Supervision</h1>

[Data Programming & Weak Supervision Area Page](data-programming.md)

Many modern machine learning systems require large, labeled datasets to be successful but producing such datasets is time-consuming and expensive. Instead, weaker sources of supervision, such as [crowdsourcing](https://papers.nips.cc/paper/2011/file/c667d53acd899a97a85de0c201ba99be-Paper.pdf), [distant supervision](https://www.aclweb.org/anthology/P09-1113.pdf), and domain experts' heuristics like [Hearst Patterns](https://people.ischool.berkeley.edu/~hearst/papers/coling92.pdf) have been used since the 90s.

However, these were largely regarded by AI and AI/ML folks as ad hoc or isolated techniques. The effort to unify and combine these into a data centric viewpoint started in earnest with [data programming](https://arxiv.org/pdf/1605.07723.pdf) embodied in the [Snorkel system](http://www.vldb.org/pvldb/vol11/p269-ratner.pdf), now an [open-source project](http://snorkel.org) and [thriving company](http://snorkel.ai). In Snorkel's conception, users specify multiple labeling functions that each represent a noisy estimate of the ground-truth label. Because these labeling functions vary in accuracy, coverage of the dataset, and may even be correlated, they are combined and denoised via a latent variable graphical model. The technical challenge is thus to learn accuracy and correlation parameters in this model, and to use them to infer the true label to be used for downstream tasks.

Data programming builds on a long line of work on parameter estimation in latent variable graphical models. Concretely, a generative model for the joint distribution of labeling functions and the unobserved (latent) true label is learned. This label model permits aggregation of diverse sources of signal, while allowing them to have varying accuracies and potential correlations.

An overview of the weak supervision pipeline can be found in this [Snorkel blog post](https://www.snorkel.org/blog/weak-supervision), including how it compares to other approaches to get more labeled data and the technical modeling challenges. These [Stanford CS229 lecture notes](https://mayeechen.github.io/files/wslecturenotes.pdf) provide a theoretical summary of how graphical models are used in weak supervision.


<h1 id="sec:representation">Data Representations & Self-Supervision</h1>

[Data Representation & Self-Supervision Area Page](data-representation.md)


<h1 id="sec:augmentation">Data Augmentation</h1>

[Data Augmentation Area Page](augmentation.md)

A key challenge when training machine learning models is collecting a large, diverse dataset that sufficiently captures the variability observed in the real world. Due to the cost of collecting and labeling datasets, data augmentation has emerged as a promising alternative. 

The central idea in data augmentation is to transform examples in the dataset in order to generate additional augmented examples that can then be added to the data. These additional examples typically increase the diversity of the data seen by the model, and provide additional supervision to the model. The foundations of data augmentation originate in [tangent propagation](https://papers.nips.cc/paper/1991/file/65658fde58ab3c2b6e5132a39fae7cb9-Paper.pdf), where model invariances were expressed by adding constraints on the derivates of the learned model.

Early successes in augmentation such as [AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) focused on inducing invariances in an image classifier by generating examples that encouraged translational or rotational invariance. These examples made augmentation a de-facto part of pipelines for a wide-ranging tasks such as image, speech and text classification, machine translation, etc. 

The choice of transformations used in augmentation is an important consideration, since it dictates the behavior and invariances learned by the model. While heuristic augmentations have remained popular, it was important to be able to control and program this augmentation pipeline carefully. [TANDA](https://arxiv.org/pdf/1709.01643.pdf.) initiated a study of the problem of programming augmentation pipelines by composing a selection of data transformations. This area  has seen rapid growth in recent years with both deeper theoretical understanding and practical implementations such as [AutoAugment](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.pdf). A nascent line of work leverages conditional generative models to learn-rather than specify-these transformations, further extending this programming paradigm. 


<h1 id="sec:contrastive">Contrastive Learning</h1>

[Constrastive Learning Area Page](contrastive.md)


<h1 id="sec:evaluation">Evaluation</h1>

[Evaluation Area Page](evaluation.md)


<h1 id="sec:robustness">Robustness</h1>  

[Robustness Area Page](robustness.md)


<h1 id="sec:applications">Applications</h1>

Data-centric approaches have had a wide-ranging impact wherever machine learning is used and deployed, whether in academia, industry or other organizations. Impact spans modalities such as structured data, text, images, videos, graphs and others, while areas include text and image processing, medical imaging, computational biology, autonomous driving, etc.
