# Data Centric AI (v0.0.1)

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

[comment]: <> (0. Add potential emoji to section header &#40;syntax `:emoji:`&#41;. )
[comment]: <> (   The emoji can be made up and may not exist &#40;yet&#41;.)

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

1. [Data Programming & Weak Supervision](#data-programming--weak-supervision)
   1. [Key Papers](#data-programming-key-papers)
   2. [Techniques](#data-programming-techniques)
   3. [Foundations](#data-programming-foundations)
   3. [Other Resources](#data-programming-resources)
   4. [Success Stories](#weak-supervision-success-stories)
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
7. [Robustess](#robustness)
   1. [Subgroup Information](#subgroup-information)
   2. [Evaluation on Unlabeled Data](#evaluation-on-unlabeled-data)
9. [Applications](#section-applications)
   1. [Named Entity Linking](#named-entity-linking) 
   2. [Medical Imaging](#medical-imaging)
   3. [Computational Biology](#computational-biology)
   4. [Observational Supervision](#observational-supervision)


# Data Programming & Weak Supervision
Many modern machine learning systems require large, labeled datasets to be successful but producing such datasets is time-consuming and expensive. Instead, weaker sources of supervision, such as [crowdsourcing](https://papers.nips.cc/paper/2011/file/c667d53acd899a97a85de0c201ba99be-Paper.pdf), [distant supervision](https://www.aclweb.org/anthology/P09-1113.pdf), and domain experts' heuristics like [Hearst Patterns](https://people.ischool.berkeley.edu/~hearst/papers/coling92.pdf) have been used since the 90s! 
 
However, these were largely regarded by AI and AI/ML folks as ad hoc or isolated techniques. The effort to unify and combine these into a data centric viewpoint started in earnest with [data programming](https://arxiv.org/pdf/1605.07723.pdf) embodied in the [Snorkel system](http://www.vldb.org/pvldb/vol11/p269-ratner.pdf), now an [open-source project](http://snorkel.org) and [thriving company](http://snorkel.ai). In Snorkel's conception, users specify multiple labeling functions that each represent a noisy estimate of the ground-truth label. Because these labeling functions vary in accuracy, coverage of the dataset, and may even be correlated, they are combined and denoised via a latent variable graphical model. The technical challenge is thus to learn accuracy and correlation parameters in this model, and to use them to infer the true label to be used for downstream tasks.

<h2 id="data-programming-link">A Link to the Classics via Graphical Models</h2>

**Learning the parameters of a latent variable graphical model:** At the heart of weak supervision lies the label model--a generative model for the joint distribution of labeling functions and the unobserved (latent) true label. This concept enables the modeling of labeling functions with varying accuracies and potential correlations. Learning the label model permits the use of diverse sources of signal. In fact, we do not need the sources to be equally accurate or to be independent of one another. In the original data programming approach, the distribution is learned by solving a maximum likelihood problem using stochastic gradient descent with Gibbs sampling. More recent work in weak supervision instead exploit the structure of the graphical model more in learning the distribution, yielding efficient algorithms discussed below.

We first provide an overview of fundamental literature in learning from multiple/noisy sources and graphical models, and then present some recent work on weak supervision and how to learn its respective graphical model.


<h2 id="data-programming-foundations">Foundational Papers</h2>


- **Crowdsourcing and noisy labelers:** the closely-related problem of dealing with many labelers of different quality, especially in the context of crowdsourcing, is foundational. The classic work of [Dawid and Skene](https://www.jstor.org/stable/2346806) uses expectation maximization to learn the confusion matrices for each labeler. In a more recent seminal work, [Karger, Oh, and Shah](https://papers.nips.cc/paper/2011/file/c667d53acd899a97a85de0c201ba99be-Paper.pdf) show how to iteratively improve estimates of reliability---reducing the total amount of required sources. [Joglekar et al](https://dl.acm.org/doi/10.1145/2487575.2487595) learn confidence intervals for labeler reliability. [Raykar et al](https://www.jmlr.org/papers/volume11/raykar10a/raykar10a.pdf) use a Bayesian approach with the EM algorithm for learning gold labels from multiple noisy sources. While these crowdsourcing models and weak supervision both address the setting of multiple noisy sources, weak supervision additionally allows for one to model dependencies between labeling functions.

- **Learning with noisy labels:** Even if we know the reliability of noisy labelers, when and how can we train a model on noisy examples? The possibility of learning with noisy labels has been studied starting with classic work such as from [Angluin and Laird](http://homepages.math.uic.edu/~lreyzin/papers/angluin88b.pdf), [Lugosi](https://www.sciencedirect.com/science/article/pii/0031320392900087), and [Bylander](https://dl.acm.org/doi/pdf/10.1145/180139.181176). More recently, for models trained with any surrogate loss function, [Scott](http://web.eecs.umich.edu/~cscott/pubs/asymsurrEJS.pdf), and [Natarajan et al](https://proceedings.neurips.cc/paper/2013/file/3871bd64012152bfb53fdf04b401193f-Paper.pdf) opened up a new area by showing that a simple re-weighting modification of the loss enables unbiased learning in the presence of noise. In weak supervision, although each labeling function can be considered a noisy labeler, there are many noisy labelers that have different unknown error rates and complex dependencies among them.

- **Learning latent-variable graphical models:** First, the problem of learning general (non-latent) graphical models and performing inference on them has been well-studied, with many books and notes such as those by [Koller](https://ai.stanford.edu/~koller/Papers/Koller+al:SRL07.pdf), [Lauritzen](http://www.statslab.cam.ac.uk/~qz280/teaching/causal-2019/reading/Lauritzen_1996_Graphical_Models.pdf), and [Wainwright and Jordan](https://people.eecs.berkeley.edu/~wainwrig/Papers/WaiJor08_FTML.pdf). Oftentimes, the graphical model has latent variables, such as in Gaussian mixture models, hidden markov models, and latent Dirichlet allocation, which is a more challenging setting. While the EM algorithm remains an option, other techniques exploit structure in matrices and tensors of moments. [Loh and Wainwright](https://arxiv.org/pdf/1212.0478.pdf) show that the augmented inverse covariance matrix of a graphical model has sparsity in a pattern that matches with conditional independence of variables, which can be used to learn parameters corresponding to the latent variables via matrix completion. Another core technique is tensor decomposition pioneered by [Anandkumar et al](https://www.jmlr.org/papers/volume15/anandkumar14b/anandkumar14b.pdf), applied to graphical models in [Chaganty and Liang](http://proceedings.mlr.press/v32/chaganty14.html). 

- **Structure learning of latent-variable graphical models:** using extensions of robust PCA ([Candes et al](https://arxiv.org/pdf/0912.3599.pdf)) permits the recovery of latent-variable Gaussian graphical models, as in [Chandrasekaran et al](https://arxiv.org/pdf/1008.1290.pdf). Discrete models share certain similar properties, i.e., having structured inverse covariance matrices.

- **Effective rank:** to characterize the learning rate for dependency structures, the [effective rank](https://arxiv.org/pdf/1011.3027.pdf) of a matrix is used for stronger [concentration inequalities](https://projecteuclid.org/journals/bernoulli/volume-21/issue-2/On-the-sample-covariance-matrix-estimator-of-reduced-effective-rank/10.3150/14-BEJ602.full) for estimation of the covariance matrix, and illustrates particular cluster patterns in graphical models for which optimal rates are obtained. In particular, it is the ratio of the trace versus the largest eigenvalue of a matrix and thus can be much smaller than the true rank.


<h2 id="data-programming-techniques">Techniques</h2>

**Exploiting dependency structure:**
- The [MeTaL](https://arxiv.org/pdf/1810.02840.pdf) paper explains that the link between structure of the inverse covariance matrix of the sources is closely related to the dependency structure of the graphical model, which provides enough information to learn the parameters of the label model as explained in these [lecture notes](https://mayeechen.github.io/files/wslecturenotes.pdf). The performance of MeTaL surpasses conventional data programming, which is slow with Gibbs sampling.

- [FlyingSquid](https://arxiv.org/pdf/2002.11955.pdf): When the structure of the model can be factorized into triplets of conditionally independent labeling functions, FlyingSquid works with simple 3x3 covariance sub-matrices to produce closed-form solutions for the parameters of the label model. This bypasses the need for stochastic gradient descent and offers even more speedup over MeTaL and enables new online and streaming applications. Moreover, this factorization is a special case of method-of-moments with tensor decomposition described above.


**Learning the structure of a latent variable graphical model:**
- [Learning dependencies](https://arxiv.org/pdf/1903.05844.pdf): in most weak supervision settings, labeling functions are assumed to be conditionally independent, or the dependencies are known. However, when they are not, robust PCA can be applied to recover the structure.
- [Learning the Structure of Generative Models without Labeled Data](https://export.arxiv.org/pdf/1703.00854): To learn the label function dependencies with limited data, we can use a structure estimation method that is 100x faster than maximum likelihood approaches.

**Using labeled and unlabeled data in weak supervision:**
- [Comparing labeled versus unlabeled data](https://arxiv.org/pdf/2103.02761.pdf): the generative models used in weak supervision can accept both labeled and unlabeled data (since usually practitioners have a small amount of labeled data available), but unlabeled input is linearly more susceptible to misspecification of the dependency structure. However, this can be corrected using a general median-of-means estimator on top of method-of-moments. 

<h2 id="data-programming-resources">Other Resources</h2>

** THESE SHOULD FEATURE MORE UPFRONT AND USE THEM IN THE STORY. YOU SHOULD BE ABLE TO POINT TO PEOPLE'S TALKS and things **

- This [Snorkel blog post](https://www.snorkel.org/blog/weak-supervision) provides an overview of the weak supervision pipeline, including how it compares to other approaches to get more labeled data and the technical modeling challenges.
- [These Stanford CS229 lecture notes](https://mayeechen.github.io/files/wslecturenotes.pdf) provide a more theoretical summary of how graphical models are used in weak supervision.


<h2 id="weak-supervision-success-stories"> Success Stories </h2>

- [Google](https://arxiv.org/pdf/1812.00417.pdf) used a weak supervision system in Ads and YouTube based on Snorkel. In just tens of minutes, this system utilizes diverse organizational resources to create classifiers with performance equivalent to those trained on tens of thousands of hand-labeled examples over millions of datapoints.

- [Intel](https://ajratner.github.io/assets/papers/Osprey_DEEM.pdf) expanded weak supervision interfaces for non-programmers and was able to replace six months of crowdworker labels while improving precision by double digits.
- [G-mail](http://cidrdb.org/cidr2020/papers/p31-sheng-cidr20.pdf) migrated their privacy-safe rule-based information extraction system Juicer to a Software 2.0 design with weak supervision, which surpassed the previous system in terms of precision and recall of the extractions and was also found to be much easier to maintain.

- [Facebook](https://ai.facebook.com/blog/billion-scale-semi-supervised-learning/) achieved new state-of-the-art performance on academic benchmarks for image and video classification by weakly supervising training labels in the billions.

** Please watch a talk for how I talk about htis... **

- [Stanford Radiology](https://arxiv.org/pdf/1903.11101.pdf) used a cross-modal weak supervision approach to weakly supervise training labels from text reports and then train an image model for the associated radiology images.
- This [Software 2.0 blog post](https://hazyresearch.stanford.edu/software2) summarizes other successes for data programming.


# Data Representations & Self-Supervision
The need for large, labeled datasets has also motivated methods for training data on naturally, or automatically, labelled datasets, allowing the model to learn latent structure in the data. For example, language models can be trained to predict the next token in a textual input. The paradigm, called "self-supervision", has revolutionized how we train (or pre-train) models. Importantly, these self-supervised models learn without manual labels or hand curated features. This reduces the engineer effort to create and maintain features and makes models significantly easier to deploy and maintain. This shift put more importance on the underlying training data and how it is represented to the model.


<h2 id="embeddings">Embeddings</h2>
Self-supervised models commonly rely on embeddings as core inputs and outputs during training. For example, language models like [BERT](https://www.aclweb.org/anthology/N19-1423.pdf) and [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) tokenize a sentence into sub-tokens. Each sub-token gets an embedding that is fed through the model and updated during training. These token embeddings encode knowledge about what the is and how it interacts with other tokens in the sentence.

### Embedding Atoms and Learning Embeddings
How you "tokenize" your intput into different atomic units and how you train an embedding changes what kind of knowledge and how the knowledge is represented.
- Graph based approaches, such as [TransE](https://papers.nips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf), represent entities (people, place, and things) and are trained to preserve link structure in a Knowledge Base.
- [Hyperbolic embeddings](https://homepages.inf.ed.ac.uk/rsarkar/papers/HyperbolicDelaunayFull.pdf) takes graph-structured embeddings one step further and learns embeddings in hyporbolic space. This [blog](https://dawn.cs.stanford.edu/2019/10/10/noneuclidean/) gives a great introduction.
- Common word embedding techniques, like [word2vec](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf), train embeddings for each word in a fixed vocabulary to predict surrounding words given a single word, preserving word co-occurrence patterns. This [chapter](http://web.stanford.edu/~jurafsky/slp3/6.pdf) from Speech and Language Processing gives a nice overview of word embeddings.
- Contextual word embeddings, like [BERT](https://www.aclweb.org/anthology/N19-1423.pdf), split words into sub-tokens and generate embeddings for each sub-token that depend on the surrounding context thereby allowing for homonyms to get different representations.
- [Selfie](https://arxiv.org/pdf/1906.02940.pdf) learns embeddings for image patches, sub-spaces of the image, to be trained using Transformer architecture.
- [StarSpace: Embed All the Things!](https://arxiv.org/abs/1709.03856) using as one-embedding-model approach to a general purpose embedding for graph nodes, words, and entities.

### Knowledge Transfer
As self-supervised embeddings are trained in a general-purpose manner, they are frequently used as core inputs into downstream tasks to be fine-tuned on a specific task. These embeddings transfer general knowledge into the downstream tasks for improved quality.

- [A Primer in BERTology: What We Know About How BERT Works](https://www.aclweb.org/anthology/2020.tacl-1.54/) explores the omniprescent use of BERT word embeddings as a way of transferring global language knowledge to downstream tasks.
- Systems like [KnowBERT](https://arxiv.org/pdf/1909.04164.pdf) and [Bootleg](https://arxiv.org/abs/2010.10363) both explore how the use of entity embeddings from a Named Entity Disambiguation system can encode entity knowledge in downstream knowledge rich tasks like relation extraction.
- [Selfie](https://arxiv.org/pdf/1906.02940.pdf) and [BEiT](https://arxiv.org/abs/2106.08254) pretrain image patch embeddings to be used in downstream image classifications tasks.

### Stability and Compression
Stability describes the sensitivity of machine learning models (e.g. embeddings) to changes in their input. In production settings, machine learning models may be constantly retrained on up-to-date data ([sometimes every hour](https://research.fb.com/wp-content/uploads/2017/12/hpca-2018-facebook.pdf)!), making it critical to understand their stability. Recent works have shown that word embeddings can suffer from instability: 

- [Factors Influencing the Surprising Instability of Word Embeddings](https://www.aclweb.org/anthology/N18-1190.pdf) 
  evaluates the impact of word properties (e.g. part of speech), data properties (e.g. word frequencies), 
  and algorithms ([PPMI](https://link.springer.com/content/pdf/10.3758/BF03193020.pdf), [GloVe](https://www.aclweb.org/anthology/D14-1162.pdf), [word2vec](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)) on word embedding stability.
- [Evaluating the Stability of Embedding-based Word Similarities](https://www.aclweb.org/anthology/Q18-1008.pdf) shows that small changes in the training data, such as including specific documents, causes the nearest neighbors of word embeddings to vary significantly.
- [Understanding the Downstream Instability of Word Embeddings](https://arxiv.org/abs/2003.04983) demonstrates that instability can propagate to the downstream models that use word embeddings and introduces a theoretically-motivated measure to help select embeddings to minimize downstream instability.  

### Theoretical Foundations
- [word2vec, node2vec, graph2vec, X2vec: Towards a Theory of Vector Embeddings of Structured Data](https://arxiv.org/abs/2003.12590), the pdf version of a POD 2020 KeyNote talk, discusses the connection between the theory of homomorphism vectors and embedding techniques.
- These [notes](http://demo.clab.cs.cmu.edu/cdyer/nce_notes.pdf) discuss how noise contrastive estimation and negative sampling impacts static word embedding training. 
- [A Mathematical Exploration of Why Language Models Help Solve Downstream Tasks](https://arxiv.org/pdf/2010.03648.pdf) seeks to explain why pretraining on next token prediction self-supervised tasks improves downstream performance.

<h2 id="learning-with-auxiliary-information">Learning with Auxiliary Information</h2>
In a self-supervised regime, large, unlabeled datasets make it difficult for engineers to inject domain specific knowledge into the model. One approach to have this fine-grained control over a model, while keeping the model as simple as possible, is to inject (latent) metadata into the model.

### Overcoming the Long Tail with Structured Data
Rare are entities (named entities, products, words, ...) are uncommon or non-existent in training data yet appear when a model is deployed. Models often struggle to resolve these rare entities as the diversity of signals required to understand them is not represented in training data. One approachs, is to rely on structural information, such as the types associated with an entity.  

- The [Bootleg](https://hazyresearch.stanford.edu/bootleg/) system leverages structured data in the form of type and knowledge graph relations to improve Named Entity Disambiguation over 40 F1 points for rare entities.
- This [zero-shot NED system](https://arxiv.org/pdf/1906.07348.pdf) uses entity descriptions to improve rare entity linking performance.
- The [TEK](https://arxiv.org/pdf/2004.12006.pdf) framework injects entity descriptions for improved reading comprehension and QA. 

### Data Shaping
Prior methods to inject domain knowledge often modify the model architecture and/or algorithms. In contrast, data shaping involves introduces external knowledge to the raw data inputted to a language model. While it may be difficult to efficiently deploy the specialized and sophisticated models as proposed in prior work, data shaping simply uses the standard language model with no modifications whatsoever to achieve competitive performance.

- Recent work on knowledge-aware language modeling, involving a modified architecture and/or learning algorithm: 
  [E-BERT](https://arxiv.org/abs/1911.03681) (Poerner, 2020), [K-Adapter](https://arxiv.org/abs/2002.01808) (Wang, 2020), 
  [KGLM](https://arxiv.org/abs/1906.07241) (Logan, 2019), [KnowBERT](https://arxiv.org/abs/1909.04164) (Peters, 2019), 
  [LUKE](https://www.aclweb.org/anthology/2020.emnlp-main.523.pdf) (Yamada, 2020), [ERNIE](https://arxiv.org/abs/1905.07129) (Zhang, 2019)
- Examples demonstrating how to introduce inductive biases through the data for knowledge-based reasoning: 
  [TEK](https://arxiv.org/pdf/2004.12006.pdf) (Joshi 2020),  [Data Noising](https://arxiv.org/pdf/1703.02573.pdf) (Xie, 2017), 
  [DeepType](https://arxiv.org/abs/1802.01021) (Raiman, 2018)
- Recent theoretical analyses of LM generalization reason about the data distributions. 
  Information theory is an important foundational topic here: [Information Theory and Statistics](http://web.stanford.edu/class/stats311/) (Stanford STAT 311)

<h2 id="data-representation-successes">Success Stories</h2>

### Feature Stores
Feature Store (FS) systems were developed to help engineers build, share, and manage data features (including pretrained embeddings) for model training and deployment. 
- Uber's [Michelangelo](https://eng.uber.com/michelangelo-machine-learning-platform/) was the first of its kind Feature Store deployed at Uber.
- [Feast](https://www.tecton.ai/feast/) by Tecton is one of the only open source Feature Store.
- There's an entire [website](https://www.featurestore.org/) devoted to Feature Stores.

### Industry and the Embedding Ecosystem
Self-supervised embeddings are core inputs to numerous downstream user-facing systems in industry. We term the embeddings and models that use them a "embedding ecosystem". A few examples of these ecosystems at work are:
- [Pinterest's](https://medium.com/pinterest-engineering/pinnersage-multi-modal-user-embedding-framework-for-recommendations-at-pinterest-bfd116b49475) multi-model user embeddings for recommendation.
- [Spotify](https://research.atspotify.com/contextual-and-sequential-user-embeddings-for-music-recommendation/) uses embeddings for user music recommendations.
- [Netflix](https://netflixtechblog.com/supporting-content-decision-makers-with-machine-learning-995b7b76006f) also uses embeddings for movie recommendations. 


# Go Big or Go Home
With the ability to train models on unlabelled data, research is scaling up both data size and model size at an [impressive rate](https://medium.com/analytics-vidhya/openai-gpt-3-language-models-are-few-shot-learners-82531b3d3122). This both raises questions of how to scale and how to make models more efficient to alleviate the costs of training.

<h2 id="universal-models">Universal Models</h2>

[comment]: <> ([Karan, Laurel])
As models get larger, researchers are seeing emergent trends of impressive zero-shot behavior. This is driving a one-model-to-rule-them-all paradigm that would alleviate the need for any downstream fine-tuning.

### Task-Agnostic, Simple Architectures
The goal is to find one architecture that can be universal, working on text, image, video, etc. Further, rather than leaning into something complex, recent work in [scaling laws](https://arxiv.org/pdf/2001.08361.pdf) suggest that architectures matter less than data. The implication is that very standard, commoditized architectures can be universal.
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


<h2 id="interactive-machine-learning">Interactive Machine Learning</h2>
With models getting larger and costing more to train, there's a growing need to interact with the model and quickly iterate on its performance before a full training run.

**I DO NOT UNDERSTAND THIS **

- **Explanatory interactive learning** Can we, by interacting with models during training, encourage their explanations to line up with our priors on what parts of the input are relevant?
   - [Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations](https://arxiv.org/pdf/1703.03717.pdf)
   - [Explanatory Interactive Machine Learning](https://ml-research.github.io/papers/teso2019aies_XIML.pdf)
   - [Making deep neural networks right for the right scientific reasons by interacting with their explanations](https://www.nature.com/articles/s42256-020-0212-3)
    
- **[Mosaic](https://github.com/robustness-gym/mosaic)** makes it easier for ML practitioners to interact with high-dimensional, multi-modal data. It provides simple abstractions for data inspection, model evaluation and model training supported by efficient and robust IO under the hood. Mosaic's core contribution is the DataPanel, a simple columnar data abstraction. The Mosaic DataPanel can house columns of arbitrary type – from integers and strings to complex, high-dimensional objects like videos, images, medical volumes and graphs.
   - [Introducing Mosaic](https://www.notion.so/Introducing-Mosaic-64891aca2c584f1889eb0129bb747863) (blog post)
   - [Working with Images in Mosaic](https://drive.google.com/file/d/15kPD6Kym0MOpICafHgO1pCt8T2N_xevM/view?usp=sharing) (Google Colab)
   - [Working with Medical Images in Mosaic](https://colab.research.google.com/drive/1UexpPqyXdKp6ydBf87TW7LtGIoU5z6Jy?usp=sharing) (Google Colab)


# Data Augmentation
Data augmentation is a standard approach for improving model performance, where additional 
synthetically modified versions of examples are added to training.


<h2 id="augmentation-history">History</h2>

Augmentation has been instrumental to achieving high-performing models since the original
[AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) 
paper on ILSVRC, which used random crops, translation & reflection of images for training, 
and test-time augmentation for prediction.

Since then, augmentation has become a de-facto part of image training pipelines and 
an integral part of text applications such as machine translation.



<h2 id="augmentation-theory">Theoretical Foundations</h2>

- [Tangent Propagation](https://papers.nips.cc/paper/1991/file/65658fde58ab3c2b6e5132a39fae7cb9-Paper.pdf) expresses desired model invariances induced by a data augmentation as tangent constraints on the directional derivatives of the learned model
- [Kernel Theory of Data Augmentation](http://proceedings.mlr.press/v97/dao19b/dao19b.pdf) connects the tangent propagation view of data augmentation to kernel-based methods.
- [On the Generalization Effects of Linear Transformations in Data Augmentation](https://arxiv.org/abs/2005.00695) studies an over-parametrized linear regression setting and study the generalization effect of applying a familar of linear transformations in this setting.



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

Recent work has hand-crafted more sophisticated primitives, such as 
- [Cutout](https://arxiv.org/abs/1708.04552)
- [Mixup](https://arxiv.org/pdf/1710.09412.pdf)
- [CutMix](https://arxiv.org/abs/1905.04899.pdf) 
- [MixMatch](https://arxiv.org/pdf/1905.02249.pdf) and [ReMixMatch](https://arxiv.org/abs/1911.09785.pdf) 
  
While these primitives have culminated in compelling performance gains, they can often produce unnatural images and distort image semantics.

#### Text
Heuristic transformations for text, typically involve paraphrasing text in order to produce more diverse samples.

- [Backtranslation](https://arxiv.org/abs/1511.06709) uses a round-trip translation from a source to target language and back in order to generate a paraphrase. 
  Examples of use include [QANet](https://arxiv.org/abs/1804.09541).
- Synonym substitution methods replace words with their synonyms such as in 
  [Data Augmentation for Low-Resource Neural Machine Translation](https://www.aclweb.org/anthology/P17-2090/),
  [Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations](https://www.aclweb.org/anthology/N18-2072/),
  [Model-Portability Experiments for Textual Temporal Analysis](https://www.aclweb.org/anthology/P11-2047/)
  [That’s So Annoying!!!](https://www.aclweb.org/anthology/D15-1306/) and
  [Character-level Convolutional Networks for Text Classification](https://arxiv.org/pdf/1509.01626.pdf)

[comment]: <> (- Noising)
[comment]: <> (- Grammar induction)
[comment]: <> (- Text editing)
[comment]: <> (- Other heuristics)

#### Audio

- Vocal Tract Length Warping approaches, such as [Audio Augmentation for Speech Recognition](https://www.danielpovey.com/files/2015_interspeech_augmentation.pdf) and [Vocal Tract Length Perturbation (VTLP) improves speech recognition](http://www.cs.toronto.edu/~ndjaitly/jaitly-icml13.pdf)
- Stochastic Feature Mapping approaches, such as in [Data Augmentation for Deep Neural Network Acoustic Modeling](https://www.semanticscholar.org/paper/Data-Augmentation-for-Deep-Neural-Network-Acoustic-Cui-Goel/c083dc15b5e169e02e208b576d6991d93955b4eb)
    and [Continuous Probabilistic Transform for Voice Conversion](https://www.ee.columbia.edu/~dpwe/papers/StylCM98-vxtfm.pdf)


### Assembled Pipelines 
An interesting idea is to learn augmentation pipelines, a study initiated by [TANDA](https://arxiv.org/pdf/1709.01643.pdf). This area has seen rapid growth in recent years with both deeper theoretical understanding and practical implementations, like AutoAugment.. 

** PLEASE CLEAN UP *** 

to determine the right subset of augmentation primitives, and the order in which they should be applied. 
These pipelines are primarily built on top of a fixed set of generic transformations.
Methods vary by the learning algorithm used, which can be

- random sampling such as in [RandAugment](https://arxiv.org/pdf/1909.13719.pdf) and an uncertainty-based random sampling scheme such as in [Dauphin](https://arxiv.org/abs/2005.00695).
- reinforcement learning approaches led by the work, and extended by [AutoAugment](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.pdf) 
- computationally efficient algorithms for learning augmentation policies have also been proposed such as [Population-Based Augmentation](https://arxiv.org/pdf/1905.05393.pdf) and [Fast AutoAugment](https://arxiv.org/pdf/1905.00397.pdf)


### Learned Primitives
There is substantial prior work in learning transformations that produce semantic, rather than superficial changes to an input. 

One paradigm is to learn a semantically meaningful data representation, and manipulate embeddings in this representation to produce a desired transformation.

- several methods express these transformations as vector operations over embeddings, such as in 
  [Deep Visual Analogy Making](https://papers.nips.cc/paper/2015/hash/e07413354875be01a996dc560274708e-Abstract.html),
  [Deep feature interpolation for image content changes](https://arxiv.org/pdf/1611.05507.pdf)
- other methods look towards manifold traversal techniques such as 
  [Deep Manifold Traversal: Changing Labels with Convolutional Features](https://arxiv.org/pdf/1511.06421.pdf),
  [Learning to disentangle factors of variation with manifold interaction](http://proceedings.mlr.press/v32/reed14.pdf)

Another class of approaches relies on training conditional generative models, that learn a mapping between two or more data distributions. 
A prominent use case focuses on imbalanced datasets, where learned augmentations are used to generate examples for underrepresented classes or domains.
Examples of these approaches include
[BaGAN](https://arxiv.org/abs/1803.09655.pdf), [DAGAN](https://arxiv.org/abs/1711.04340.pdf), [TransferringGAN](https://arxiv.org/abs/1805.01677.pdf), 
[Synthetic Examples Improve Generalization for Rare Classes](https://arxiv.org/pdf/1904.05916.pdf),
[Learning Data Manipulation for Augmentation and Weighting](https://arxiv.org/pdf/1910.12795.pdf),
[Generative Models For Deep Learning with Very Scarce Data](https://arxiv.org/abs/1903.09030.pdf),
[Adversarial learning of general transformations for data augmentation](https://arxiv.org/abs/1909.09801.pdf),
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

- the ["Automating the Art of Data Augmentation"](https://hazyresearch.stanford.edu/data-aug-part-1) 
  series of blog posts by [Sharon Li](http://pages.cs.wisc.edu/~sharonli/) provide an overview of data augmentation.


# Contrastive Learning

<h2 id="contrastive-theory">Theoretical Foundations</h2>

Contrastive learning works by optimizing a loss function that pulls together similar points ("positive" pairs) and pushes apart dissimilar points ("negative" pairs). Compared to its empirical success in self-supervision, a theoretical understanding of contrastive learning is relatively lacking in terms of what sort of representations are learned by minimizing contrastive loss, and what these representations guarantee on downstream tasks.

- [Representations induced on the hypersphere](https://arxiv.org/pdf/2005.10242.pdf): assuming that the representations to learn are constrained to a hypersphere, the contrastive loss function is closely connected to optimizing for "alignment" (positive pairs map to the same representation) and "uniformity" (representations are "spread out" as much as possible on the hypersphere to maintain as much as information as possible).
- [Downstream performance](https://arxiv.org/pdf/1902.09229.pdf): suppose that similar pairs belong to the same latent subclass, and that the downstream task aims to classify among some of these latent subclasses. Then, downstream loss of a linear classifier constructed using mean representations can be expressed in terms of the unsupervised contrastive loss.
- [Debiasing contrastive learning](https://arxiv.org/pdf/2007.00224.pdf) and [using hard negative samples](https://openreview.net/pdf?id=CR1XOQ0UTh-): in unsupervised settings, negative pairs are constructed by selecting two points at random i.i.d. This can result in the two points actually belonging to the same latent subclass, but this can be corrected via importance weighting. Moreover, even within different latent subclasses, some negative samples can be "harder" than others and enforce better representations.


<h2 id="contrastive-applications">Applications</h2>



# Fine-Grained Evaluation
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

[comment]: <> (### Foundations)

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
that are interpretable, task-relevant, error-prone and susceptible to distribution shift. 

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

- [Hidden Stratification](https://arxiv.org/pdf/1909.12475.pdf) describes the problem of subpar performance on hidden strata. 
- The [GLUE](https://openreview.net/pdf?id=rJ4km2R5t7) benchmark paper describes important
  slices for text classification, which were used to decide what may be challenging examples for GLUE. 
- The [Robustness Gym](https://arxiv.org/abs/2101.04840.pdf) paper contains slice-based analyses
across several NLP tasks (natural language inference, named entity linking, text summarization).
- Subgroup robustness work typically specify slices.  


[comment]: <> (Another approach to understand )

## Benchmarking

Benchmarking is a common practice for quantifying progress in the machine learning community and 
comparing models across tasks and applications.

Recent work has shifted towards data-driven benchmarking, where instead of submitting models, participants can submit datasets to understand the impact of the data on training. A prominent example is the [Data-Centric AI Competition](https://https-deeplearning-ai.github.io/data-centric-comp/).

Similarly, [Dynabench](https://dynabench.org/) addresses problems inherent to traditional benchmarks such as overfitting and saturation by providing an platform for dynamic, human-in-the-loop dataset collection and model benchmarking. 

Other benchmarking efforts have focused on enabling more robust model comparisons by (1) reporting richer benchmarking data such as evaluation metrics beyond accuracy (e.g. fairness scores) and metrics of practical concern (e.g. compute) and by (2) providing tools for fine-grained model comparisons. Some examples are as follows:
- [Dynaboard](https://ai.facebook.com/blog/dynaboard-moving-beyond-accuracy-to-holistic-model-evaluation-in-nlp/): an evaluation-as-a-service interface for comparing models across a holistic set of evaluation criteria including accuracy, robustness, fairness, compute, and memory. 
- [ExplainaBoard](http://explainaboard.nlpedia.ai/): an interactive leaderboard and evaluation software for fine-grained model comparisons.
- [Ludwig Benchmarking Toolkit](https://github.com/HazyResearch/ludwig-benchmarking-toolkit): a personalized benchmarking toolkit for running multi-objective, standardized, and configurable benchmarking studies.




[comment]: <> ([Avanika])
[comment]: <> (## Robustness [Jared])
[comment]: <> (- Hidden Stratification + GEORGE)

# Robustness
NEED ROBUSTNESS SETUP/HIDDEN STRAT
label drift whatever
Hidden straification point to talks! This is a huge thing, it came first--and it shouldn't be some after thought as "application".
Then, put all that work in context.
Tell the stor!

<h2 id="subgroup-information">Subgroup Information</h2>

A data subset or "subgroup" may carry spurious correlations between its features and labels that do not hold for datapoints outside of the subgroup. When certain subgroups are larger than others, models trained to minimize average error are susceptible to learning these spurious correlations and performing poorly on the minority subgroups. 

To obtain good performance on *all* subgroups, in addition to the ground-truth labels we can bring in subgroup information during training.

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
- Importance weighting works poorly when the supports of the source and target do not overlap and when data is high-dimensional. [Mandoline](https://mayeechen.github.io/files/mandoline.pdf)  addresses this by reweighting based on user/model-defined ``slices'' that intend to capture relevant axes of distribution shift. Slices are often readily available as subpopulations identified by the practitioner, but can also be based on things like metadata and the trained model's scores.

### Outlier Detection
_This section is a stub. You can help by improving it._

### Active Sampling and Labeling
_This section is a stub. You can help by improving it._

[comment]: <> (## Video)

[comment]: <> ([Dan])


<h1 id="section-applications">Applications</h1>

<h2 id="named-entity-linking">Named Entity Linking</h2>

Named entity linking (NEL) is the task of linking ambiguous mentions in text to entities in a knowledge base. NEL is a core preprocessing step in downstream applications, including search and question answering.
- Pre-deep-learning approaches to NEL have been [rule-based](https://www.aclweb.org/anthology/X96-1053.pdf) or leverage statistical techniques and manual feature engineering to filter and rank candidates ([survey paper](https://arxiv.org/abs/1910.11470)).
- In recent years, deep learning systems have become the new standard ([overview paper](https://dl.acm.org/doi/10.1145/3183713.3196926) of deep learning approaches to entity disambiguation and entity matching problems). The most recent state-of-the-art models generally rely on deep contextual word embeddings with entity embeddings. For example, [Pre-training of Deep Contextualized Embeddings of Words and Entities for Named Entity Disambiguation](https://arxiv.org/pdf/1909.00426v1.pdf) and [Empirical Evaluation of Pretraining Strategies for Supervised Entity Linking](https://arxiv.org/pdf/2005.14253.pdf).
- We've seen a recent shift in simplifying the model even more to just use tranformers without explicit entity embeddings with models like [BLINK](https://arxiv.org/pdf/1911.03814.pdf) (uses a bi-encoder) and the [Dual and Cross-Attention Encoders](https://arxiv.org/pdf/2004.03555.pdf) (uses cross-encoder).
- Other trends have been to enhance the training data further. The system [Bootleg](https://arxiv.org/pdf/2010.10363.pdf) system uses weak labeling of the training data to noisily assign entity links to mentions, increasing performance over rare entities.
- Ikuya Yamada has a wonderful GitHub [survey](https://github.com/izuna385/Entity-Linking-Recent-Trends) of recent trends in Entity Linking

<h2 id="medical-imaging">Medical Imaging</h2>

- Sensitive to inputs, not models
    - The varient of imaging configurations (e.g. [site locations](https://arxiv.org/pdf/2002.11379.pdf)), hardware, and processing techniques (e.g. [CT windowing](https://pubs.rsna.org/doi/abs/10.1148/ryai.2021200229)) lead to large performance shifts
    - Recent medical imaging challenges (segmentation: [knee](https://arxiv.org/pdf/2004.14003.pdf), [brain](https://arxiv.org/pdf/1811.02629.pdf), reconstruction: [MRI](https://arxiv.org/abs/2012.06318)), found that, to a large extent, the choice of model is less important than the underlying distribution of data (e.g. disease extent)

- Towards multi-modal data fusion
    - Radiologist reports (and more generally text) have been used to improve learned visual representations (e.g. [ConVIRT](https://arxiv.org/abs/2010.00747)) and to source weak labels in annotation-scarce settings (e.g. ([PET/CT](https://www-nature-com.stanford.idm.oclc.org/articles/s41467-021-22018-1)))
    - Auxiliary features from other rich, semi-structured data, such as [electronic health records (EHRs)](https://www-nature-com.stanford.idm.oclc.org/articles/s41746-020-00341-z), successfully complemented standard image representations

[comment]: <> (## Image Segmentation)

[comment]: <> ([Sarah])

<h2 id="computational-biology">Computational Biology</h2>

- Collecting the right data for training and evalution can require wetlab work – especially in computational drug discovery. 
   - [A Deep Learning Approach to Antibiotic Discovery](https://www.cell.com/cell/pdf/S0092-8674(20)30102-1.pdf)
   - [Network medicine framework for identifying drug-repurposing opportunities for COVID-19](https://www.pnas.org/content/118/19/e2025581118)
   
- Non-standard data modalities are common in computational biology. 
   - Biological Interaction Networks (_e.g._ [Network-based in silico drug efficacy screening](https://www.nature.com/articles/ncomms10331), [Identification of disease treatment mechanisms through the multiscale interactome](https://www.nature.com/articles/s41467-021-21770-8)
   - Chemical Graphs (_e.g._ [Strategies for pre-training graph neural networks](https://arxiv.org/pdf/1905.12265.pdf)
   - DNA, RNA and Amino Acid sequences (_e.g._[Sequential regulatory activity prediction across chromosomes with convolutional neural networks](https://genome.cshlp.org/content/28/5/739.short)
   - 3D structures (_e.g._ [Learning from protein structure with geometric vector perceptrons](https://openreview.net/pdf?id=1YLJDvSx6J4)

- In order to facilitate the extraction of relevant signal from large biological datasets, methods have been designed to prune irrelevant features and integrate knowledge across datasets.  
   - [AMELIE](https://stm.sciencemag.org/content/scitransmed/12/544/eaau9113.full.pdf) helps improve diagnosis of Mendelian disorders by integrating information from a patient’s phenotype and genotype and automatically identifying relevant references to literature.
   - [This](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1004754) article discusses the importance of creating effective feature selection methods to filter irrelevant features from large whole genome datasets. Other works (such as [this one](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03693-1) and [this one](https://www.worldscientific.com/doi/abs/10.1142/9789813279827_0024)) discuss approaches for identifying putative genetic variants by incorporating information from interaction networks or utilizing independent control datasets.
   - Approaches for extracting biological information from medical literature (such as [chemical-disease relation extraction](https://link.springer.com/article/10.1186/s13321-016-0165-z) and [genotype-phenotype association extraction](https://www.nature.com/articles/s41467-019-11026-x)) have benefitted from data programming techniques as well as the incorporation of weakly labeled data.


<h2 id="observational-supervision">Observational Supervision</h2>

The way experts interact with their data (e.g. a radiologist’s eye movements) contains rich information about the task (e.g. classification difficulty), and the expert (e.g. drowsiness level).
With the current trend of wearable technology (e.g. AR with eye tracking capability), the hardware needed to collect such human-data interactions is expected to become more ubiquitous, affordable, and standardized. 
In observational supervision, we investigate how to extract the rich information embedded in the human-data interaction, to either supervise models from scratch, or to improve model robustness.

Interesting works have collected observational signals such as:
- Eye tracking data in medicine (chest x-ray [dataset](https://www.nature.com/articles/s41597-021-00863-5.pdf))
- Eye tracking plus brain activity in NLP (Zuco [dataset](https://www.nature.com/articles/sdata2018291.pdf)) 
- We have also collaborated with Stanford radiologists to curate an additional two medical datasets with eye tracking data [coming soon!].

Critical papers in observational supervision:
- Some of the pioneering work on using gaze data. N. Hollenstein and C. Zhang showed how to use gaze data to improve NLP models [paper](https://arxiv.org/pdf/1902.10068.pdf).
- Improving zero-shot learning with gaze by N. Karasseli et al. [paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Karessli_Gaze_Embeddings_for_CVPR_2017_paper.pdf) 
- Improving sample complexity with gaze by K. Saab et al. [paper](https://openreview.net/pdf?id=r1gPtjcH_N)
- Our recent work on supervising medical models from scratch [coming soon!]
