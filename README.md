# Data Centric AI

We're collecting (an admittedly opinionated) list of resources and progress made 
in data-centric AI, with exciting directions from our own lab and collaborators.

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

### :lotus_position: Stability and Compression [Megan, Simran]
Stability describes the sensitivity of machine learning models (e.g. embeddings) to changes in their input. In production settings, machine learning models may be constantly retrained on up-to-date data ([sometimes every hour](https://research.fb.com/wp-content/uploads/2017/12/hpca-2018-facebook.pdf)!), making it critical to understand their stability. Recent works have shown that word embeddings can suffer from instability: 

- [Factors Influencing the Surprising Instability of Word Embeddings](https://www.aclweb.org/anthology/N18-1190.pdf) evaluates the impact of word properties (e.g. part of speech), data properties (e.g. word frequencies), and algorithms ([PPMI](https://link.springer.com/content/pdf/10.3758/BF03193020.pdf), [GloVe](https://www.aclweb.org/anthology/D14-1162.pdf), [word2vec](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)) on word embedding stability. 
- [Evaluating the Stability of Embedding-based Word Similarities](https://www.aclweb.org/anthology/Q18-1008.pdf) shows that small changes in the training data, such as including specific documents, causes the nearest neighbors of word embeddings to vary significantly.
- [Understanding the Downstream Instability of Word Embeddings](https://arxiv.org/abs/2003.04983) demonstrates that instability can propagate to the downstream models that use word embeddings and introduces a theoretically-motivated measure to help select embeddings to minimize downstream instability.  
  
### Embedding Patching [Laurel]
- Goodwill Hunting
- Model Patching

## :joystick: Interactive Machine Learning [Karan, Sabri, Arjun, Laurel]

- **Forager** [Fait]
- **[Mosaic](https://github.com/robustness-gym/mosaic)** makes it easier for ML practitioners to interact with high-dimensional, multi-modal data. It provides simple abstractions for data inspection, model evaluation and model training supported by efficient and robust IO under the hood. Mosaic's core contribution is the DataPanel, a simple columnar data abstraction. The Mosaic DataPanel can house columns of arbitrary type – from integers and strings to complex, high-dimensional objects like videos, images, medical volumes and graphs.
   - [Introducing Mosaic](https://www.notion.so/Introducing-Mosaic-64891aca2c584f1889eb0129bb747863) (blog post)
   - [Working with Images in Mosaic](https://drive.google.com/file/d/15kPD6Kym0MOpICafHgO1pCt8T2N_xevM/view?usp=sharing) (Google Colab)
   - [Working with Medical Images in Mosaic](https://colab.research.google.com/drive/1UexpPqyXdKp6ydBf87TW7LtGIoU5z6Jy?usp=sharing) (Google Colab)
- **Explanatory interactive learning** Can we, by interacting with models during training, encourage their explanations to line up with our priors on what parts of the input are relevant?
   - [Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations](https://arxiv.org/pdf/1703.03717.pdf)
   - [Explanatory Interactive Machine Learning](https://ml-research.github.io/papers/teso2019aies_XIML.pdf)
   - [Making deep neural networks right for the right scientific reasons by interacting with their explanations](https://www.nature.com/articles/s42256-020-0212-3)

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

### Learning with Structured Data [Laurel, Maya, Megan]

Structured data, such as the types associated with an entity, can provide useful signals for training models, alongside unstructured text corpora.  

- [Bootleg](https://hazyresearch.stanford.edu/bootleg/) is a system that leverages structured data in the form of type and knowledge graph relations to improve named entity disambiguation over 40 F1 points for rare entities. 
- This [blog]([README.md](https://hazyresearch.stanford.edu/bootleg_blog)) describes how Bootleg uses both structured and unstructured data to learn reasoning patterns, such as certain words should be associated with a type.  

### Data Shaping [Simran]
Standard language models struggle to reason over the long-tail of entity-based knowledge and significant recent work tackles this challenge by providing the model with external knowledge signals. Prior methods modify the model architecture and/or algorithms to introduce the knowledge. In contrast, data shaping involves introduces external knowledge to the raw data inputted to a language model. While it may be difficult to efficiently deploy the specialized and sophisticated models as proposed in prior work, data shaping simply uses the standard language model with no modifications whatsoever to achieve competitive performance.
- Recent work on knowledge-aware language modeling, involving a modified architecture and/or learning algorithm: [E-BERT](https://arxiv.org/abs/1911.03681) (Poerner, 2020), [K-Adapter](https://arxiv.org/abs/2002.01808) (Wang, 2020), [KGLM](https://arxiv.org/abs/1906.07241) (Logan, 2019), [KnowBERT](https://arxiv.org/abs/1909.04164) (Peters, 2019), [LUKE](https://www.aclweb.org/anthology/2020.emnlp-main.523.pdf) (Yamada, 2020), [ERNIE](https://arxiv.org/abs/1905.07129) (Zhang, 2019)
- Examples demonstrating how to introduce inductive biases through the data for knowledge-based reasoning: [TEK](https://arxiv.org/pdf/2004.12006.pdf) (Joshi 2020),  [Data Noising](https://arxiv.org/pdf/1703.02573.pdf) (Xie, 2017), [DeepType](https://arxiv.org/abs/1802.01021) (Raiman, 2018)
- Recent theoretical analyses of LM generalization reason about the data distributions. Information theory is an important foundataional topic here: [Information Theory and Statistics](http://web.stanford.edu/class/stats311/) (Stanford STAT 311)

### Subgroup Information [Michael]
Similar to observations brought up with hidden stratification, a data subset or "subgroup" may carry spurious correlations between its features and labels that do not hold for datapoints outside of the subgroup. When certain subgroups are larger than others, models trained to minimize average error are susceptible to learning these spurious correlations and performing poorly on the minority subgroups. To obtain good performance on *all* subgroups, in addition to the ground-truth labels we can bring in subgroup information during training. 
- [Group Distributionally Robust Optimization (Group DRO)](https://arxiv.org/abs/1911.08731) assumes knowledge of which subgroup each training sample belongs to, and proposes a training algorithm that reweights the loss objective to focus on subgroups with higher error.  
- [Model Patching](https://arxiv.org/abs/2008.06775) uses a generative model to synthesize samples from certain subgroups as if they belonged to another. These augmentations can then correct for subgroup imbalance, such that training on the new dataset mitigates learning correlations that only hold for the original majority subgroups.

Subgroup information also does not need to be explicitly annotated or known. Several recent works aim to first infer subgroups before using a robust training method to obtain good performance on all subgroups. A frequent heuristic is to use the above observation that models trained with empirical risk minimization (ERM) and that minimize average error may still perform poorly on minority subgroups; one can then infer minority or majority subgroups depending on if the trained ERM model correctly predicts the datapoints.  
- [Learning from Failure (LfF](https://arxiv.org/abs/2007.02561) trains two models in tandem. Each model trains on the same data batches, where for each batch, datapoints that the first model gets incorrect are upweighted in the loss objective for the second model. 
- [Just Train Twice (JTT)]() trains an initial ERM model for a few epochs, identifies the datapoints this model gets incorrect after training, and trains a new model with ERM on the same dataset but with the incorrect points upsampled.  
- [Correct-N-Contrast (CNC)]() also trains an initial ERM model, but uses supervised contrastive learning to train a new model to learn similar representations for datapoints with the same class but different trained ERM model predictions.

### Observational Supervision [Khaled]


## Applications

### Named Entity Linking [Laurel, Maya, Megan]

Named entity linking (NEL) is the task of linking ambiguous mentions in text to entities in a knowledge base. NEL is a core preprocessing step in downstream applications, including search and question answering. 

- [Shift towards simple Transformer models for NEL with bi-encoders and cross-encoders](https://arxiv.org/abs/1911.03814): recent state-of-the-art models such as BLINK rely on a simple two-stage architecture for NEL. First a bi-encoder retrieves candidate entitites by embedding the query and entities. Then a cross-encoder re-ranks the candidate entities.  
- [Data-driven improvements in NEL through weak labeling](https://arxiv.org/pdf/2010.10363.pdf): Bootleg uses weak labeling of the training data to noisily assign entity links to mentions, increasing performance over rare entities. 

### Video [Dan]

### :xray: Medical Imaging [Sarah, Arjun]

- :seesaw: Sensitive to inputs, not models
    - The varient of imaging configurations (e.g. [site locations](https://arxiv.org/pdf/2002.11379.pdf)), hardware, and processing techniques (e.g. [CT windowing](https://pubs.rsna.org/doi/abs/10.1148/ryai.2021200229)) lead to large performance shifts
    - Recent medical imaging challenges (segmentation: [knee](https://arxiv.org/pdf/2004.14003.pdf), [brain](https://arxiv.org/pdf/1811.02629.pdf), reconstruction: [MRI](https://arxiv.org/abs/2012.06318)), found that, to a large extent, the choice of model is less important than the underlying distribution of data (e.g. disease extent)

-  :mixing-pot: Towards multi-modal data fusion
    - :report: Radiologist reports (and more generally text) have been used to improve learned visual representations (e.g. [ConVIRT](https://arxiv.org/abs/2010.00747)) and to source weak labels in annotation-scarce settings (e.g. ([PET/CT](https://www-nature-com.stanford.idm.oclc.org/articles/s41467-021-22018-1)))
    - :ekg: :test-tube: Auxiliary features from other rich, semi-structured data, such as [electronic health records (EHRs)](https://www-nature-com.stanford.idm.oclc.org/articles/s41746-020-00341-z), successfully complemented standard image representations



### Image Segmentation [Sarah]

### :dna: Computational Biology [Sabri, Maya]
- :pill: Collecting the right data for training and evalution can require wetlab work – especially in computational drug discovery. 
   - [A Deep Learning Approach to Antibiotic Discovery](https://www.cell.com/cell/pdf/S0092-8674(20)30102-1.pdf))
   - [Network medicine framework for identifying drug-repurposing opportunities for COVID-19](https://www.pnas.org/content/118/19/e2025581118))
   
- :jigsaw: Non-standard data modalities are common in computational biology. 
   - Biological Interaction Networks (_e.g._ [Network-based in silico drug efficacy screening](https://www.nature.com/articles/ncomms10331), [Identification of disease treatment mechanisms through the multiscale interactome](https://www.nature.com/articles/s41467-021-21770-8))
   - Chemical Graphs (_e.g._ [Strategies for pre-training graph neural networks](https://arxiv.org/pdf/1905.12265.pdf))
   - DNA, RNA and Amino Acid sequences (_e.g._[Sequential regulatory activity prediction across chromosomes with convolutional neural networks](https://genome.cshlp.org/content/28/5/739.short))
   - 3D structures (_e.g._ [Learning from protein structure with geometric vector perceptrons](https://openreview.net/pdf?id=1YLJDvSx6J4))

- In order to facilitate the extraction of relevant signal from large biological datasets, methods have been designed to prune irrelevant features and integrate knowledge across datasets.  
   - [AMELIE](https://stm.sciencemag.org/content/scitransmed/12/544/eaau9113.full.pdf) helps improve diagnosis of Mendelian disorders by integrating information from a patient’s phenotype and genotype and automatically identifying relevant references to literature.
   - [This](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1004754) article discusses the importance of creating effective feature selection methods to filter irrelevant features from large whole genome datasets. Other works (such as [this one](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03693-1) and [this one](https://www.worldscientific.com/doi/abs/10.1142/9789813279827_0024)) discuss approaches for identifying putative genetic variants by incorporating information from interaction networks or utilizing independent control datasets.
   - Approaches for extracting biological information from medical literature (such as [chemical-disease relation extraction](https://link.springer.com/article/10.1186/s13321-016-0165-z) and [genotype-phenotype association extraction](https://www.nature.com/articles/s41467-019-11026-x)) have benefitted from data programming techniques as well as the incorporation of weakly labeled data. 

## Benchmarking [Avanika]

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

