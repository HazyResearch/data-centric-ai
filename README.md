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

Widespread use: shift in how information is represented.
- The take over of Word2Vec and BERT
- Epoxy [Dan]
- Replacing influence functions via NN
- Hidden Stratification + GEORGE
- VLDB Tutorial [upcoming]

### Stability [Megan]

### Compression [Simran]

### Embedding Updates [Laurel]

### Label Propagation [Dan]
- Epoxy

### Hidden Stratification [Jared]
- Hidden Stratification + GEORGE

## Interactive Machine Learning [Karan, Sabri, Arjun, Laurel]

- Forager [Fait]
- Mosaic DataPanels

## Fine-Grained Evaluation & Measurement [Karan, Mayee]

- Robustness Gym 
- Mandoline
- Active Validation [Vishnu]

## Data-Driven Inductive Bias in Model Representations [Albert, Ines]
When you don't have enough data, inductive biases can make models much more efficient. 
- SLLSSL
- Hyperbolics

## Robustness [Jared]
- Hidden Stratification + GEORGE



## Learning with Auxiliary Information

### Higher-Level Signals [Laurel, Maya, Megan]

### Data Shaping [Simran]

### Subgroup Information [Michael]

### Observational Supervision [Khaled]


## Applications

### Named Entity Linking [Laurel, Maya, Megan]

### Video [Dan]

### Medical Imaging [Sarah, Arjun]

### Image Segmentation [Sarah]

### Computational Biology [Sabri, Maya]


## :frodo-monstertruck-sauron: Universal Models [Karan, Laurel]
Shift towards one-model-to-rule-them-all paradigm.


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

### Contrastive Learning [Mayee]

### Weak Supervision [Mayee, Fred]

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

