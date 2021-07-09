# Data Representations & Self-Supervision

The need for large, labeled datasets has also motivated methods for training data on naturally, or automatically, labeled datasets, allowing the model to learn latent structure in the data. For example, language models can be trained to predict the next token in a textual input. The paradigm, called "self-supervision", has revolutionized how we train (or pre-train) models. Importantly, these self-supervised models learn without manual labels or hand curated features. This reduces the engineering effort to create and maintain features and makes models significantly easier to deploy and maintain. This shift put more importance on the underlying training data and how it is represented to the model.

<h2 id="embeddings">Embeddings</h2>
Self-supervised models commonly rely on embeddings as core inputs and outputs during training. For example, language models like [BERT](https://www.aclweb.org/anthology/N19-1423.pdf) and [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) tokenize a sentence into sub-tokens. Each sub-token gets an embedding that is fed through the model and updated during training. These token embeddings encode knowledge about what the is and how it interacts with other tokens in the sentence.

### Embedding Atoms and Learning Embeddings

How you "tokenize" your intput into different atomic units and how you train an embedding changes what kind of knowledge and how the knowledge is represented.

- Graph based approaches, such as [TransE](https://papers.nips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf), represent entities (people, place, and things) and are trained to preserve link structure in a Knowledge Base.
- [Hyperbolic embeddings](https://homepages.inf.ed.ac.uk/rsarkar/papers/HyperbolicDelaunayFull.pdf) takes graph-structured embeddings one step further and learns embeddings in hyperbolic space. This [blog](https://dawn.cs.stanford.edu/2019/10/10/noneuclidean/) gives a great introduction.
- Common word embedding techniques, like [word2vec](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf), train embeddings for each word in a fixed vocabulary to predict surrounding words given a single word, preserving word co-occurrence patterns. This [chapter](http://web.stanford.edu/~jurafsky/slp3/6.pdf) from Speech and Language Processing gives a nice overview of word embeddings.
- Contextual word embeddings, like [BERT](https://www.aclweb.org/anthology/N19-1423.pdf), split words into sub-tokens and generate embeddings for each sub-token that depend on the surrounding context thereby allowing for homonyms to get different representations.
- [Selfie](https://arxiv.org/pdf/1906.02940.pdf) learns embeddings for image patches, sub-spaces of the image, to be trained using Transformer architecture.
- [StarSpace: Embed All the Things!](https://arxiv.org/abs/1709.03856) using as one-embedding-model approach to a general purpose embedding for graph nodes, words, and entities.

### Knowledge Transfer

As self-supervised embeddings are trained in a general-purpose manner, they are frequently used as core inputs into downstream tasks to be fine-tuned on a specific task. These embeddings transfer general knowledge into the downstream tasks for improved quality.

- [A Primer in BERTology: What We Know About How BERT Works](https://www.aclweb.org/anthology/2020.tacl-1.54/) explores the omniprescent use of BERT word embeddings as a way of transferring global language knowledge to downstream tasks.
- Systems like [KnowBERT](https://arxiv.org/pdf/1909.04164.pdf) and [Bootleg](https://arxiv.org/abs/2010.10363) both explore how the use of entity embeddings from a Named Entity Disambiguation system can encode entity knowledge in downstream knowledge-rich tasks like relation extraction.
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

Prior methods to inject domain knowledge often modify the model architecture and/or algorithms. In contrast, data shaping involves introducing external knowledge to the raw data inputted to a language model. While it may be difficult to efficiently deploy the specialized and sophisticated models as proposed in prior work, data shaping simply uses the standard language model with no modifications whatsoever to achieve competitive performance.

- Recent work on knowledge-aware language modeling, involving a modified architecture and/or learning algorithm:
  [E-BERT](https://arxiv.org/abs/1911.03681) (Poerner, 2020), [K-Adapter](https://arxiv.org/abs/2002.01808) (Wang, 2020),
  [KGLM](https://arxiv.org/abs/1906.07241) (Logan, 2019), [KnowBERT](https://arxiv.org/abs/1909.04164) (Peters, 2019),
  [LUKE](https://www.aclweb.org/anthology/2020.emnlp-main.523.pdf) (Yamada, 2020), [ERNIE](https://arxiv.org/abs/1905.07129) (Zhang, 2019)
- Examples demonstrating how to introduce inductive biases through the data for knowledge-based reasoning:
  [TEK](https://arxiv.org/pdf/2004.12006.pdf) (Joshi 2020), [Data Noising](https://arxiv.org/pdf/1703.02573.pdf) (Xie, 2017),
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

- [Pinterest's](https://medium.com/pinterest-engineering/pinnersage-multi-modal-user-embedding-framework-for-recommendations-at-pinterest-bfd116b49475) multi-modal user embeddings for recommendation.
- [Spotify](https://research.atspotify.com/contextual-and-sequential-user-embeddings-for-music-recommendation/) uses embeddings for user music recommendations.
- [Netflix](https://netflixtechblog.com/supporting-content-decision-makers-with-machine-learning-995b7b76006f) also uses embeddings for movie recommendations.
