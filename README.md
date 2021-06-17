# Data Centric AI

We're collecting (an admittedly opinionated) list of resources and progress made 
in data-centric AI, with exciting directions from our own lab and collaborators.

AI has been too focused on the models, while in reality, the experience of those who put 
models into production or use them in real applications is that data matters most. 

It's the data that's unique, not the models. 

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

# Open Problems




# Emerging Directions
New directions in data-centric AI.



_Themes_:
- how to engineer the data to 
- how does data influence the learned representations?

## Embeddings
How data is represented and transferred

### "Atoms" of an embedding
- Patches for images
- Subword tokens, BPE, No tokens

### Knowledge Transfer
- BERTology
- Bootleg
- Omniprescence in industrial ecosystems

### Continuous (Fuzzy??) Distances
- Epoxy [Dan]

### Stability and Compression [Megan, Simran]

### Embedding Patching [Laurel]
- Goodwill Hunting
- Model Patching

## Interactive Machine Learning [Karan, Sabri, Arjun, Laurel]

- Forager [Fait]
- Mosaic DataPanels

## :sleeping: Fine-Grained Evaluation & Measurement [Karan]

To better understand how models perform when deployed in real-world settings, tools for fine-grained analysis and efficient methods for handling unlabeled validation/test data are needed in evaluation.

### Robustness Gym

### Mandoline: Model Evaluation under Distribution Shift

Distribution shift between labeled source data and unlabeled target data can be addressed with importance weighting, but this method works poorly when the supports of the source and target do not overlap and when data is high-dimensional. Mandoline addresses this by reweighting based on user/model-defined ``slices'' that intend to capture relevant axes of distribution shift.
- [Density Ratio Estimation for Machine Learning](https://www.cambridge.org/core/books/density-ratio-estimation-in-machine-learning/BCBEA6AEAADD66569B1E85DDDEAA7648) explains the different approaches to estimate density ratios, a key technical step in computing importance weights.
- [CS329D at Stanford: ML Under Distribution Shifts](https://thashim.github.io/cs329D-spring2021/) covers current research in distribution shift, ranging from covariate shift to adversarial robustness.
- [Propensity Scores](https://academic.oup.com/biomet/article/70/1/41/240879) are used in observational studies for correcting disparities when evaluating treatment on a target population given that the treatment was applied to a set of potentially biased subjects. 
- [Learning Bounds on Importance Weighting](https://papers.nips.cc/paper/2010/file/59c33016884a62116be975a9bb8257e3-Paper.pdf): how well importance weighting corrects for distribution shift can be attributed to the variance of the weights, or alternatively the R\'enyi divergence between source and target. 

### Active Validation [Vishnu]

## Data-Driven Inductive Bias in Model Representations [Albert, Ines]
When you don't have enough data, inductive biases can make models much more efficient. 
- SLLSSL
- Hyperbolics

## Robustness [Jared]
- Hidden Stratification + GEORGE



## Learning with Auxiliary Information
Shift towards guiding and coding models with (latent) metadata

### Higher-Level Signals [Laurel, Maya, Megan]
- Bootleg blog on structural resources. Uses structural resources to overcome the tail
- Just using descriptions rather than any memorization with BLINK

### Data Shaping [Simran]

### Subgroup Information [Michael]

### Observational Supervision [Khaled]


## Applications

### Named Entity Linking [Laurel, Maya, Megan]
- Shift towards simple Transformer models with BLINK and CrossEncoder
- Using more data-driven changes with Zero-Shot Description and Bootleg

### Video [Dan]

### Medical Imaging [Sarah, Arjun]

### Image Segmentation [Sarah]

### Computational Biology [Sabri, Maya]


## :frodo-monstertruck-sauron: Universal Models [Karan, Laurel]
Shift towards one-model-to-rule-them-all paradigm.


## Benchmarking [Avanika]


### Emphasis on Scale

- [GPT-3 Paper](https://arxiv.org/abs/2005.14165.pdf)
  - 170B parameter model capable of few-shot in-context learning.
- [Moore's Law for Everything](https://moores.samaltman.com)
  - A post about scale and its effect on AI / society.
- [Switch Transformers](https://arxiv.org/pdf/2101.03961.pdf)
  - Mixture of experts for training massive models.  

### Multi-Modal Applications

- [Wu Dao 2.0](https://www.engadget.com/chinas-gigantic-multi-modal-ai-is-no-one-trick-pony-211414388.html)
  - 1.75T parameter MoE model with multimodal capabilities.
- [DALL-E](https://openai.com/blog/dall-e/) & [CLIP](https://openai.com/blog/clip/)

  
### Data-Agnostic Architectures

- Transformers: 
  [NLP Transformers](https://arxiv.org/pdf/1706.03762.pdf), 
  [Vision Transformers](https://arxiv.org/pdf/2010.11929.pdf)
  - Explanatory [Blog](https://jalammar.github.io/illustrated-transformer/)
- Multi-Layer Perceptrons
  - [gMLP](https://arxiv.org/pdf/2105.08050.pdf)


### Industrial Trends
- Companies like [OpenAI](https://openai.com), [Anthropic](https://www.anthropic.com), [Cohere](https://cohere.ai) see building universal models as part of their core business strategy.
- Lots of companies emerging that rely on APIs from these universal model companies to build applications on top e.g. [AI Dungeon](https://play.aidungeon.io/main/landing). A long list from OpenAI at this [link](https://openai.com/blog/gpt-3-apps/).

### Data Trends
- There's been a shift towards understanding how to collect and curate truly massive amounts of data for pretraining.
    - [The Pile](https://pile.eleuther.ai) is a new dataset for training language models.
    - [Huggingface BigScience](https://docs.google.com/document/d/1BIIl-SObx6tR41MaEpUNkRAsGhH7CQqP_oHQrqYiH00/edit) is a new effort to establish good practices in data curation.

[comment]: <> (### Other Links)
[comment]: <> (- Stanford class [upcoming])

## Theoretical Foundations

### Contrastive Learning

Contrastive learning works by optimizing a typically unsupervised loss function that pulls together similar points (``positive'' pairs) and pushes apart dissimilar points (``negative'' pairs). A theoretical understanding is lacking on what sort of representations are learned under contrastive loss, and what these representations guarantee on downstream tasks.

- [Representations induced on the hypersphere](https://arxiv.org/pdf/2005.10242.pdf): assuming that the representations to learn are constrained to a hypersphere, the contrastive loss function is closely connected to optimizing for ``alignment'' (positive pairs map to the same representation) and ``uniformity'' (representations are ``spread out'' as much as possible on the hypersphere to maintain as much as information as possible).
- [Downstream performance](https://arxiv.org/pdf/1902.09229.pdf): suppose that similar pairs belong to the same latent subclass, and that the downstream task aims to classify among some of these latent subclasses. Then, downstream loss of a linear classifier constructed using mean representations can be expressed in terms of the contrastive loss.
- [Debiasing contrastive learning](https://arxiv.org/pdf/2007.00224.pdf) and [using hard negative samples](https://openreview.net/pdf?id=CR1XOQ0UTh-): in unsupervised settings, negative pairs are constructed by selecting two points at random i.i.d. This can result in the two points actually belonging to the same latent subclass, but this can be corrected via importance weighting. Moreover, even within different latent subclasses, some negative samples can be ``harder'' than others and enforce better representations.

### Weak Supervision

The theory behind weak supervision and data programming relies on latent variable estimation in graphical models.

- [Wainwright and Jordan textbook](https://people.eecs.berkeley.edu/~wainwrig/Papers/WaiJor08_FTML.pdf): provides an overview of graphical models
- [Structured inverse covariance matrices](https://arxiv.org/pdf/1212.0478.pdf): the (augmented) inverse covariance matrix of a graphical model will have 0s in locations where the row and column indices are independent conditional on all other random variables. Under sufficient sparsity of the graphical model, this property can be used in weak supervision to learn the correlations between latent variables (i.e. the unobserved ground-truth label). 
- Beyond using the inverse covariance matrix, certain families of graphical models can use method-of-moments---computing correlations among triplets of conditional independent variables---to estimate latent parameters. In the scalar setting, [FlyingSquid](https://arxiv.org/pdf/2002.11955.pdf) applies this method to weak supervision, and more generally [tensor decomposition](https://www.jmlr.org/papers/volume15/anandkumar14b/anandkumar14b.pdf) can be used in latent variable estimation.
- In most weak supervision settings, labeling functions are assumed to be conditionally independent, or the dependencies are known. However, when they are not, [robust PCA](https://arxiv.org/pdf/1903.05844.pdf) can be applied to recover the structure.
- [Comparing labeled versus unlabeled data](https://arxiv.org/pdf/2103.02761.pdf): generative classifiers based on graphical models (e.g. in weak supervision) can accept both labeled and unlabeled data, but unlabeled input is linearly more susceptible to misspecification of the dependency structure. However, this can be corrected using a general median-of-means estimator on top of method-of-moments.  

### Data Augmentation [Tri, Hongyang, Sen]
- Sharon's Blog Post Series

### Sparsity [Beidi, Tri]

### Structured Matrices [Albert, Tri]



# Successes in Data-Centric AI [Nancy, Karan, Laurel]
Where data-centric AI has already succeeded!

## Stories
- GMail
- Google Ads
- Tesla Data Engine
- Instacart/all the blog posts of turning to logs for embedding pretraining


## Industry
- Snorkel
- Ludwig
- DataRobot
- Karpathy's Blog(s)


# Tools

## Monitoring
- Weights & Biases
- CometML

## Feature Stores
- Uber Michelangelo
- Feast + Tecton

## Data Exploration Tools
- Pandas
- Mosaic

## Data Repositories
- Huggingface Datasets
- Kaggle
- MIMIC 
- WILDS
- Tensorflow Datasets
- Torch Datasets
- Scikit-Learn
- UCI Datasets Repository  
- data.gov

## Data Collection
- Snorkel
- Crowdsourcing stuff

## Zero Code ML
- Ludwig
- Overton


# History of Data-Centric AI

_Themes:_ more data, noisier data, less supervision.

1. I want more labeled data.
2. I can handle more noise, so it's easier to collect labeled data.
3. I don't even need labels (or very few labels).


## Early Approaches to Data-Centric AI [25 BC - 1980]
- Stone Tablets
- Egyptians with Papyrus
- Printing Press & Gutenberg
- The Personal Computer


## More Labeled Data [2008? - 2015?]

### Data Collection
Any and all mechanisms for gathering clean-ish data and manually cleaning it. 
Methods for collecting more data e.g. web scraping, crowdsourcing, Wikipedia annotations.

#### Modern Approaches

### Data Labeling & Crowdsourcing



## Noisier Data [2015 - 2018?]

### Noise-Aware Approaches to Data Collection

### Data Programming & Weak Supervision [Snorkel, Dan]
- General approach of data programming.
- Incorporating noisy data sources.


### Data Augmentation [Sen, Hongyang]


## Less Supervision [2018 - present]

### Pretraining with Large, Unlabeled Data

### Scaling Up [Beidi]
Transformers, GPT

### Multi-Modality

### Few-Shot & Meta Learning


## Large Datasets
Data-centric AI was pushed forward by the availability of large
datasets.

- ImageNet [totally supervised]
- MS Coco [rich annotations]
- Language Modeling Datasets (GPT, Pile) [pretraining]
- Huggingface Datasets

### Feature Engineering [Snorkel]
- Tell the story of how people used to do careful feature engineering, 
hand-crafted features.
- Still done in industry with feature stores alongside embeddings.

## Data Extraction & Preparation

### Data Cleaning
### Data Integration
### Knowledge Base Construction

