# Data Centric AI

We're collecting (an admittedly opinionated) list of resources and progress made 
in data-centric AI, with exciting directions past and present.

AI has been pretty focused on models, while in reality, the experience of those who put 
models into production or use them in real applications is that data matters a lot (maybe even more!). 
Often, it's the data (which is unique to you) that can make or break model performance.

---

## Contributing
We want this resource to grow with contributions from readers and data enthusiasts.
Make a pull request if you want to add resources.

Instructions for adding resources:

0. Add potential emoji to section header (syntax `:emoji:`). 
   The emoji can be made up and may not exist (yet).
1. Write a sentence summarizing the content of the section you're writing. 
   Feel free to change the header name to something more appropriate 
   and/or split your section across multiple headers if that makes sense to you.
2. Add a few critical links (citations, paper links, blogs, videos, code, workshops, 
   classes, tutorials, figures, pictures, recipes, tweets, books), and a short 
   description of what the content is about and how it relates to the section.

---

# Data Programming & Weak Supervision
[Snorkel: Rapid Training Data Creation with Weak Supervision](http://www.vldb.org/pvldb/vol11/p269-ratner.pdf) was the seminal work on data prgoramming, the ability to label data through programmatic labelling functions.

## The Theory of Weak Supervision

The theory behind weak supervision and data programming relies on latent variable estimation in graphical models.

- [Wainwright and Jordan textbook](https://people.eecs.berkeley.edu/~wainwrig/Papers/WaiJor08_FTML.pdf): provides an overview of graphical models
- [Structured inverse covariance matrices](https://arxiv.org/pdf/1212.0478.pdf): the (augmented) inverse covariance matrix of a graphical model will have 0s in locations where the row and column indices are independent conditional on all other random variables. Under sufficient sparsity of the graphical model, this property can be used in weak supervision to learn the correlations between latent variables (i.e. the unobserved ground-truth label). 
- Beyond using the inverse covariance matrix, certain families of graphical models can use method-of-moments---computing correlations among triplets of conditional independent variables---to estimate latent parameters. In the scalar setting, [FlyingSquid](https://arxiv.org/pdf/2002.11955.pdf) applies this method to weak supervision, and more generally [tensor decomposition](https://www.jmlr.org/papers/volume15/anandkumar14b/anandkumar14b.pdf) can be used in latent variable estimation.
- In most weak supervision settings, labeling functions are assumed to be conditionally independent, or the dependencies are known. However, when they are not, [robust PCA](https://arxiv.org/pdf/1903.05844.pdf) can be applied to recover the structure.
- [Comparing labeled versus unlabeled data](https://arxiv.org/pdf/2103.02761.pdf): generative classifiers based on graphical models (e.g. in weak supervision) can accept both labeled and unlabeled data, but unlabeled input is linearly more susceptible to misspecification of the dependency structure. However, this can be corrected using a general median-of-means estimator on top of method-of-moments.  

## Applications

## Success Stories
- Gmail
- Google Ads

# Contrastive Learning

## Theoretical Foundations
Contrastive learning works by optimizing a typically unsupervised loss function that pulls together similar points ("positive" pairs) and pushes apart dissimilar points ("negative" pairs). A theoretical understanding is lacking on what sort of representations are learned under contrastive loss, and what these representations guarantee on downstream tasks.

- [Representations induced on the hypersphere](https://arxiv.org/pdf/2005.10242.pdf): assuming that the representations to learn are constrained to a hypersphere, the contrastive loss function is closely connected to optimizing for "alignment" (positive pairs map to the same representation) and "uniformity" (representations are ``spread out'' as much as possible on the hypersphere to maintain as much as information as possible).
- [Downstream performance](https://arxiv.org/pdf/1902.09229.pdf): suppose that similar pairs belong to the same latent subclass, and that the downstream task aims to classify among some of these latent subclasses. Then, downstream loss of a linear classifier constructed using mean representations can be expressed in terms of the contrastive loss.
- [Debiasing contrastive learning](https://arxiv.org/pdf/2007.00224.pdf) and [using hard negative samples](https://openreview.net/pdf?id=CR1XOQ0UTh-): in unsupervised settings, negative pairs are constructed by selecting two points at random i.i.d. This can result in the two points actually belonging to the same latent subclass, but this can be corrected via importance weighting. Moreover, even within different latent subclasses, some negative samples can be ``harder'' than others and enforce better representations.

## Applications


# Data Augmentation
Data augmentation is a standard approach for improving model performance, where additional 
synthetically modified versions of examples are added to training.

## History
Augmentation has been instrumental to achieving high-performing models since the original
[AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) 
paper on ILSVRC, which used random crops, translation & reflection of images for training, 
and test-time augmentation for prediction.

Since then, augmentation has become a de-facto part of image training pipelines.

## Theoretical Foundations

- 
- [Kernel Theory of Data Augmentation]
- []

## Learned vs. Specified Augmentations

## Label-Preserving 

## Applications


# Data Representations

## Embeddings
Data is represented and transferred through embeddings which encode
knowledge about the "unit" the embedding is representing. The widespread
use of embeddings is fundamentally changing how we build and understand models.

### Learning Embeddings
How you train an embedding changes what kind of knowledge and how the knowledge is represented
- Graph based approaches, such as [TransE](https://papers.nips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf), preserve link structure
- [Hyperbolic embeddings](https://homepages.inf.ed.ac.uk/rsarkar/papers/HyperbolicDelaunayFull.pdf) takes graph-structured embeddigns one step further and learns embeddings in hyporbolic space. This [blog](https://dawn.cs.stanford.edu/2019/10/10/noneuclidean/) gives a great introduction.
- Common word embedding techniques, like [word2vec](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf), train embeddings to predict surrounding words given a single word, preserving word co-occurrence patterns.
- Contextual word embeddings, like [BERT](https://www.aclweb.org/anthology/N19-1423.pdf), generate embeddings that depend on the surrounding context thereby allowing for homonyms to get different representations.
- [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html) learns euclidean embeddings of images of faces to better perform face recognition.
- [StarSpace: Embed All the Things!](https://arxiv.org/abs/1709.03856) using as one-embedding-model approach to a general purpose embedding for graphs and lang

### Knowledge Transfer
The embeddings can be use in downstream tasks as a way of transferring knowledge.
- [A Primer in BERTology: What We Know About How BERT Works](https://www.aclweb.org/anthology/2020.tacl-1.54/) explores the omniprescent use of BERT word embeddings as a way of transferring global language knowledge to downstream tasks.
- [Bootleg: Chasing the Tail with Self-Supervised Named Entity Disambiguation](https://arxiv.org/abs/2010.10363) explores how the use of entity embeddings from a Named Entity Disambiguation system can encode entity knowledge in downstream knowledge rich tasks, like relation extraction.

### :lotus_position: Stability and Compression
Stability describes the sensitivity of machine learning models (e.g. embeddings) to changes in their input. In production settings, machine learning models may be constantly retrained on up-to-date data ([sometimes every hour](https://research.fb.com/wp-content/uploads/2017/12/hpca-2018-facebook.pdf)!), making it critical to understand their stability. Recent works have shown that word embeddings can suffer from instability: 

- [Factors Influencing the Surprising Instability of Word Embeddings](https://www.aclweb.org/anthology/N18-1190.pdf) 
  evaluates the impact of word properties (e.g. part of speech), data properties (e.g. word frequencies), 
  and algorithms ([PPMI](https://link.springer.com/content/pdf/10.3758/BF03193020.pdf), [GloVe](https://www.aclweb.org/anthology/D14-1162.pdf), [word2vec](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)) on word embedding stability.
- [Evaluating the Stability of Embedding-based Word Similarities](https://www.aclweb.org/anthology/Q18-1008.pdf) shows that small changes in the training data, such as including specific documents, causes the nearest neighbors of word embeddings to vary significantly.
- [Understanding the Downstream Instability of Word Embeddings](https://arxiv.org/abs/2003.04983) demonstrates that instability can propagate to the downstream models that use word embeddings and introduces a theoretically-motivated measure to help select embeddings to minimize downstream instability.  

### Theoretical Foundations
- [word2vec, node2vec, graph2vec, X2vec: Towards a Theory of Vector Embeddings of Structured Data](https://arxiv.org/abs/2003.12590), the pdf version of a POD 2020 KeyNote talk, discusses the connection between the theory of homomorphism vectors and embedding techniques.
- These [notes](http://demo.clab.cs.cmu.edu/cdyer/nce_notes.pdf) discuss how noise contrastive estimation and negative sampling impacts static word embedding training.  

## Learning with Auxiliary Information
Shift towards guiding and coding models with (latent) metadata

### Higher-Level Signals [Laurel, Maya, Megan]
- Bootleg blog on structural resources. Uses structural resources to overcome the tail
- Just using descriptions rather than any memorization with BLINK

[comment]: <> (### Data Shaping [Simran])

[comment]: <> (### Subgroup Information [Michael])

[comment]: <> (### Observational Supervision [Khaled])



[comment]: <> (## Data-Driven Inductive Bias in Model Representations [Albert, Ines])
[comment]: <> (When you don't have enough data, inductive biases can make models much more efficient. )
[comment]: <> (- SLLSSL)
[comment]: <> (- Hyperbolics)

## Success Storeis

### Feature Stores
- Uber Michelangelo
- Feast + Tecton



# Fine-Grained Evaluation & Measurement
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

### Foundations

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
that are interpretable, task-relevant, error-prone and suspectible to distribution shift. 

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

- [Hidden Stratification]() describes the problem of subpar performance on hidden strata. 
- The [GLUE](https://openreview.net/pdf?id=rJ4km2R5t7) benchmark paper describes important
  slices for text classification, which were used to decide what may be challenging examples for GLUE. 
- The [Robustness Gym](https://arxiv.org/abs/2101.04840.pdf) paper contains slice-based analyses
across several NLP tasks (natural language inference, named entity linking, text summarization).
- Subgroup robustness work typically specify slices.  

### Evaluation Criteria
- Robust accuracy, or the worst case performance across a set of subpopulations 
has become prominent as an evaluation metric in the model robustness literature.


## Evaluation on Unlabeled Data
A key part of model evaluation is _monitoring_ the data in order to track when the data distribution
has shifted and take remedial action. Since the model makes predictions on unlabeled data,
standard validation cannot be used due to the absence of labels.

Instead, the problem is one of statistical estimation, 
with techniques for direct performance estimation that rely on importance weighting, and active sampling
methods that use a small target sample to estimate performance.

### Estimating Target Performance

Distribution shift between labeled source data and unlabeled target data can be addressed
with importance weighting. 
A variety of importance weighting methods are popular in the literature. 


- [Density Ratio Estimation for Machine Learning](https://www.cambridge.org/core/books/density-ratio-estimation-in-machine-learning/BCBEA6AEAADD66569B1E85DDDEAA7648) explains the different approaches to estimate density ratios, a key technical step in computing importance weights.
- [CS329D at Stanford: ML Under Distribution Shifts](https://thashim.github.io/cs329D-spring2021/) covers current research in distribution shift, ranging from covariate shift to adversarial robustness.
- [Propensity Scores](https://academic.oup.com/biomet/article/70/1/41/240879) are used in observational studies for correcting disparities when evaluating treatment on a target population given that the treatment was applied to a set of potentially biased subjects. 
- [Learning Bounds on Importance Weighting](https://papers.nips.cc/paper/2010/file/59c33016884a62116be975a9bb8257e3-Paper.pdf): how well importance weighting corrects for distribution shift can be attributed to the variance of the weights, or alternatively the R\'enyi divergence between source and target.
- [Mandoline](?) works poorly when the supports of the source and target do not overlap and when data is high-dimensional. Mandoline addresses this by reweighting based on user/model-defined ``slices'' that intend to capture relevant axes of distribution shift.

### Outlier Detection

### Active Sampling and Labeling
Another approach to understand 

## Benchmarking [Avanika]

[comment]: <> (## Robustness [Jared])
[comment]: <> (- Hidden Stratification + GEORGE)

# Go Big or Go Home

## :frodo-monstertruck-sauron: Universal Models [Karan, Laurel]
Shift towards one-model-to-rule-them-all paradigm.

### Data-Agnostic Architectures
The goal is to find one architecutre that can be universal, working on text, image, video, etc.
- The most common standard architecture is that of the [Transformer](https://arxiv.org/pdf/1706.03762.pdf), explained very well in this [blog](https://jalammar.github.io/illustrated-transformer/).
- Transformers have seen wide-spread-use in NLP tasks through [BERT](https://www.aclweb.org/anthology/N19-1423.pdf), [RoBERTa](https://arxiv.org/abs/1907.11692v1), and Hugging Face's [model hub](https://huggingface.co/models), where numerous Transformer style models are trained and shared.
- Recent work has shown how Transformers can even be sued in vision tasks with the [Vision Transformers](https://arxiv.org/pdf/2010.11929.pdf).
  - Transformers are no pancea and are still generally larger and slower to train that the simple model of a MLP. Recent work has explored how you can replace the Transformer architecture with a sequence of MLPs in the [gMLP](https://arxiv.org/pdf/2105.08050.pdf).

### Emphasis on Scale
With the ability to train models without needing labelled data through self-supervision, the focus became on scaling models up and training on more data.
- [GPT-3](https://arxiv.org/abs/2005.14165.pdf) was the first 170B parameter model capable of few-shot in-context learning developed by OpenAI.
- [Moore's Law for Everything](https://moores.samaltman.com) is a post about scale and its effect on AI / society.
- [Switch Transformers](https://arxiv.org/pdf/2101.03961.pdf) is a mixture of experts for training massive models beyond the scale of GPT-3.  

### Multi-Modal Applications
Models are also becoming more unviersal, capable of handling multiple modalities at once.
- [Wu Dao 2.0](https://www.engadget.com/chinas-gigantic-multi-modal-ai-is-no-one-trick-pony-211414388.html) is the Chinese 1.75T parameter MoE model with multimodal capabilities.
- [DALL-E](https://openai.com/blog/dall-e/) & [CLIP](https://openai.com/blog/clip/) are two other multi-modal models

### Industrial Trends
- Companies like [OpenAI](https://openai.com), [Anthropic](https://www.anthropic.com), [Cohere](https://cohere.ai) see building universal models as part of their core business strategy.
- Lots of companies emerging that rely on APIs from these universal model companies to build applications on top e.g. [AI Dungeon](https://play.aidungeon.io/main/landing). A long list from OpenAI at this [link](https://openai.com/blog/gpt-3-apps/).

### Data Trends
- There's been a shift towards understanding how to collect and curate truly massive amounts of data for pretraining.
    - [The Pile](https://pile.eleuther.ai) is a new massive, more diverse dataset for training language models than the standard Common Crawl.
    - [Huggingface BigScience](https://docs.google.com/document/d/1BIIl-SObx6tR41MaEpUNkRAsGhH7CQqP_oHQrqYiH00/edit) is a new effort to establish good practices in data curation.

### Theoretical Foundations
- [Limitations of Autoregressive Models and Their Alternatives](https://arxiv.org/abs/2010.11939) explores the theoretical limitations of autoregressive language models in the inability to represent "hard" language distributions.

[comment]: <> (### Other Links)
[comment]: <> (- Stanford class [upcoming])

## Efficient Models and Sparsity [Beidi, Tri]


[comment]: <> (## Interactive Machine Learning [Karan, Sabri, Arjun, Laurel])

[comment]: <> (- Forager [Fait])

[comment]: <> (- Mosaic DataPanels)

# Applications

### Named Entity Linking [Laurel, Maya, Megan]

### Video [Dan]

### Medical Imaging [Sarah, Arjun]

### Image Segmentation [Sarah]

### Computational Biology [Sabri, Maya]
