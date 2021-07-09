# Self-Supervision
The need for large, labeled datasets has motivated methods to pre-train latent representations of the input space using unlabeled data and use the now knowledge-rich representations in downstream tasks. As the representations allow for knowledge transfer to downstream tasks, these tasks require less labeled data. For example, language models can be pre-trained to predict the next token in a textual input to learn representations of words or sub-tokens. These word representations are then used in downstream models such as sentiment classification. This paradigm, called "self-supervision", has revolutionized how we train (and pre-train) models. Importantly, these self-supervised pre-trained models learn without manual labels or hand curated features. This reduces the engineer effort to create and maintain features and makes models significantly easier to deploy and maintain. This shift has allowed for more data to be fed to the model and shifted the focus to understanding what data to use.


<h2 id="representations-and-knowledge-transfer">Representations and Knowledge Transfer</h2>
Self-supervised models train representations, or embeddings, to encode latent structural information about the input. For example, language models like [BERT](https://www.aclweb.org/anthology/N19-1423.pdf) and [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) learn an embedding per sub-token in a sentence. These sub-token embeddings encode knowledge about what the token is and how it interacts with other tokens in the sentence.

How you "tokenize" your intput into different atomic units and how you train an embedding changes what kind of knowledge and how the knowledge is represented.


- Graph based approaches, such as [TransE](https://papers.nips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf), represent entities (people, place, and things) and are trained to preserve link structure in a Knowledge Base.
- [Hyperbolic embeddings](https://homepages.inf.ed.ac.uk/rsarkar/papers/HyperbolicDelaunayFull.pdf) takes graph-structured embeddings one step further and learns embeddings in hyporbolic space. This [blog](https://dawn.cs.stanford.edu/2019/10/10/noneuclidean/) gives a great introduction.
- Common word embedding techniques, like [word2vec](https://papers.nips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf), train embeddings for each word in a fixed vocabulary to predict surrounding words given a single word, preserving word co-occurrence patterns. This [chapter](http://web.stanford.edu/~jurafsky/slp3/6.pdf) from Speech and Language Processing gives a nice overview of word embeddings.
- Contextual word embeddings, like [ELMo](https://arxiv.org/pdf/1802.05365.pdf), split words into sub-tokens and generate embeddings for each sub-token that depend on the surrounding context thereby allowing, for example, homonyms to get different representations.
- While self-supervised representations were more common for language and graph based tasks, recent work of [Selfie](https://arxiv.org/pdf/1906.02940.pdf) learns embeddings for image patches, sub-spaces of the image, to be trained using Transformer architecture.


As self-supervised embeddings are trained in a general-purpose manner, they are used as core inputs into downstream tasks to be fine-tuned. Critically, as these embeddings transfer general knowledge, the downstream tasks need less labeled data.


- [A Primer in BERTology: What We Know About How BERT Works](https://www.aclweb.org/anthology/2020.tacl-1.54/) explores the omniprescent use of BERT word embeddings as a way of transferring global language knowledge to downstream tasks.
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683v1.pdf) and the tutorial of [Transfer Learning in Natural Language Processing](https://www.aclweb.org/anthology/N19-5004.pdf) give an overview of the different factors contributing to the success of transfer learning methods in natural language processing.
- [A Mathematical Exploration of Why Language Models Help Solve Downstream Tasks](https://arxiv.org/pdf/2010.03648.pdf) seeks to explain why pretraining on next token prediction self-supervised tasks improves downstream performance.

<h2 id="learning-with-structured-knowledge">Learning with Structured Knowledge</h2>
In a self-supervised regime, large, unlabeled datasets make it difficult for engineers to inject domain specific knowledge into the model. One approach to have this fine-grained control over a model, while keeping the model as simple as possible, is to inject (latent) metadata into the model. For example, works like [TEK](https://arxiv.org/pdf/2004.12006.pdf) inject entity descriptions for improved reading comprehension and QA.

The real benefit from knowledge injection is over the long tail of rare things (named entities, products, words, ...) that are uncommon or non-existent in training data yet appear when a model is deployed. Models often struggle to resolve these rare things as the diversity of signals required to understand them is not represented in training data. One approach is to rely on structural information, such as the types associated with an entity.  

- The [Bootleg](https://hazyresearch.stanford.edu/bootleg/) system leverages structured data in the form of type and knowledge graph relations to improve Named Entity Disambiguation over 40 F1 points for rare entities.
- This [zero-shot NED system](https://arxiv.org/pdf/1906.07348.pdf) uses entity descriptions to improve rare entity linking performance.
- The word sense disambiguation model by [Belvins and Zettlemoyer](https://arxiv.org/pdf/2005.02590v1.pdf) use word glosses to improve performance for rare words.

<h2 id="contrastive-learning">Contrastive Learning</h2>

Contrastive learning works by optimizing a loss function that pulls together similar points ("positive" pairs) and pushes apart dissimilar points ("negative" pairs). Compared to its empirical success in self-supervision, a theoretical understanding of contrastive learning is relatively lacking in terms of what sort of representations are learned by minimizing contrastive loss, and what these representations guarantee on downstream tasks.


- [Representations induced on the hypersphere](https://arxiv.org/pdf/2005.10242.pdf): assuming that the representations to learn are constrained to a hypersphere, the contrastive loss function is closely connected to optimizing for "alignment" (positive pairs map to the same representation) and "uniformity" (representations are "spread out" as much as possible on the hypersphere to maintain as much as information as possible).
- [Downstream performance](https://arxiv.org/pdf/1902.09229.pdf): suppose that similar pairs belong to the same latent subclass, and that the downstream task aims to classify among some of these latent subclasses. Then, downstream loss of a linear classifier constructed using mean representations can be expressed in terms of the unsupervised contrastive loss.
- [Debiasing contrastive learning](https://arxiv.org/pdf/2007.00224.pdf) and [using hard negative samples](https://openreview.net/pdf?id=CR1XOQ0UTh-): in unsupervised settings, negative pairs are constructed by selecting two points at random i.i.d. This can result in the two points actually belonging to the same latent subclass, but this can be corrected via importance weighting. Moreover, even within different latent subclasses, some negative samples can be "harder" than others and enforce better representations.

<h2 id="self-supervision-successes">Success Stories</h2>


- Self-supervised langauge models like BERT are being used in Google [search](https://www.blog.google/products/search/search-language-understanding-bert/)
- Recommender pipelines like those at [Pinterest's](https://medium.com/pinterest-engineering/pinnersage-multi-modal-user-embedding-framework-for-recommendations-at-pinterest-bfd116b49475), [Spotify](https://research.atspotify.com/contextual-and-sequential-user-embeddings-for-music-recommendation/), and [Netflix](https://netflixtechblog.com/supporting-content-decision-makers-with-machine-learning-995b7b76006f) all use pretrained product or customer embeddings to improve downstream recommendations. 