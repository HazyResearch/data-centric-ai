# Go Big or Go Home

With the ability to train models on unlabelled data, research is scaling up both data size and model size at an [impressive rate](https://medium.com/analytics-vidhya/openai-gpt-3-language-models-are-few-shot-learners-82531b3d3122). This both raises questions of how to scale and how to make models more efficient to alleviate the costs of training.

<h2 id="universal-models">Universal Models</h2>

[comment]: <> ([Karan, Laurel])
As models get larger, researchers are seeing emergent trends of impressive zero-shot behavior. This is driving a one-model-to-rule-them-all paradigm that would alleviate the need for any downstream fine-tuning.

### Task-Agnostic, Simple Architectures

The goal is to find one architecture that can be universal, working on text, image, video, etc. Further, rather than leaning into something complex, recent work in [scaling laws](https://arxiv.org/pdf/2001.08361.pdf) suggest that architectures matter less than data. The implication is that very standard, commoditized architectures can be universal.

- The most common standard architecture is that of the [Transformer](https://arxiv.org/pdf/1706.03762.pdf), explained very well in this [blog](https://jalammar.github.io/illustrated-transformer/).
- Transformers have seen wide-spread-use in NLP tasks through [BERT](https://www.aclweb.org/anthology/N19-1423.pdf), [RoBERTa](https://arxiv.org/abs/1907.11692v1), and Hugging Face's [model hub](https://huggingface.co/models), where numerous Transformer style models are trained and shared.
- Recent work has shown how Transformers can even be used in vision tasks with the [Vision Transformers](https://arxiv.org/pdf/2010.11929.pdf).
- Transformers are no panacea and are still generally larger and slower to train that the simple model of a MLP. Recent work has explored how you can replace the Transformer architecture with a sequence of MLPs in the [gMLP](https://arxiv.org/pdf/2105.08050.pdf).

### Emphasis on Scale

With the ability to train models without needing labelled data through self-supervision, the focus became on scaling models up and training on more data.

- [GPT-3](https://arxiv.org/abs/2005.14165.pdf) was the first 170B parameter model capable of few-shot in-context learning developed by OpenAI.
- [Moore's Law for Everything](https://moores.samaltman.com) is a post about scale and its effect on AI / society.
- [Switch Transformers](https://arxiv.org/pdf/2101.03961.pdf) is a mixture of experts for training massive models beyond the scale of GPT-3.

### Multi-Modal Applications

Models are also becoming more universal, capable of handling multiple modalities at once.

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

<h2 id="efficient-models">Efficient Models</h2>
As models get larger, there is an increasing need to make training less computational expensive, especially in larger Transformer networks that have a quadratic complexity (w.r.t. input sequence length) in attention layers.

### Exploit Sparsity

- For efficient MLP Training, [SLIDE](https://arxiv.org/pdf/1903.03129.pdf) uses Locality Sensitive Hashing(LSH) for approximating dense matrix multiplications in linear layers preceding a softmax (identify sparsity patterns).
- For efficient Transformers Training, [Reformer](https://arxiv.org/pdf/2001.04451.pdf) uses similar techniques, LSH, to reduce quadratic computations in attention layers.
- [MONGOOSE](https://openreview.net/pdf?id=wWK7yXkULyh) uses a scheduling algorithm and learnable hash functions to address LSH overhead and further speed-up MLP and Transformer training.
- [Routing Transformers](https://arxiv.org/pdf/2003.05997.pdf) and [Smyrf](https://arxiv.org/pdf/2010.05315.pdf) use other similar techniques, such as k-means or asymmetric LSH to identify sparsity patterns.

### Exploit Low-rank Properties

- This [paper](http://www.vikas.sindhwani.org/lowRank.pdf) uses low-rank matrix factorization to reduce the computation for linear layers.
- For efficient attention computation in Transformers, [Performer](https://arxiv.org/pdf/2009.14794.pdf), [Linformer](http://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf), [Linear Transformer](https://arxiv.org/pdf/2006.04768.pdf) use either low-rank or kernel approximation techniques.

### Sparse+Low-rank and other References

- Inspired by the classical robust-PCA algorithm for sparse and low-rank decomposition, [Scatterbrain]() unifies sparse (via LSH) and low-rank (via kernel feature map) attention for accurate and efficient approximation.
- This [survey paper](https://arxiv.org/pdf/2009.06732.pdf) covers most of the efficient Transformer variants.
- [Long Range Arena](https://arxiv.org/pdf/2011.04006.pdf) is a systematic and unified benchmark, focused on evaluating model quality under long-context scenarios.

<h2 id="interactive-machine-learning">Interactive Machine Learning</h2>
With models getting larger and costing more to train, there's a growing need to interact with the model and quickly iterate on its performance before a full training run.

- **Explanatory interactive learning** Can we, by interacting with models during training, encourage their explanations to line up with our priors on what parts of the input are relevant?
  - [Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations](https://arxiv.org/pdf/1703.03717.pdf)
  - [Explanatory Interactive Machine Learning](https://ml-research.github.io/papers/teso2019aies_XIML.pdf)
  - [Making deep neural networks right for the right scientific reasons by interacting with their explanations](https://www.nature.com/articles/s42256-020-0212-3)
- **[Meerkat](https://github.com/robustness-gym/meerkat)** makes it easier for ML practitioners to interact with high-dimensional, multi-modal data. It provides simple abstractions for data inspection, model evaluation and model training supported by efficient and robust IO under the hood. Mosaic's core contribution is the DataPanel, a simple columnar data abstraction. The Mosaic DataPanel can house columns of arbitrary type – from integers and strings to complex, high-dimensional objects like videos, images, medical volumes and graphs.
  - [Introducing Meerkat](https://www.notion.so/Introducing-Mosaic-64891aca2c584f1889eb0129bb747863) (blog post)
  - [Working with Images in Mosaic](https://drive.google.com/file/d/15kPD6Kym0MOpICafHgO1pCt8T2N_xevM/view?usp=sharing) (Google Colab)
  - [Working with Medical Images in Mosaic](https://colab.research.google.com/drive/1UexpPqyXdKp6ydBf87TW7LtGIoU5z6Jy?usp=sharing) (Google Colab)