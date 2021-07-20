# The End of Modelitis

Two trends are determining the end of "New Modelitis", the tendency of researchers and practitioners to focus on new model architectures with mostly marginal gains, rather than other, potentially more impactful, aspects of the machine learning pipeline, such as data quality and evaluation.

Model building platforms like [Ludwig](https://eng.uber.com/introducing-ludwig/) and [Overton](https://www.cs.stanford.edu/~chrismre/papers/overton-tr.pdf) enforce commoditized architectures, and move towards ML systems that can be created declaratively [Molino and Ré 2021](https://arxiv.org/abs/2107.08148). These platforms show that commoditiy models can perform even better than their tuned predecessors!

With the ability to train models on unlabeled data, research is scaling up both data and model size at an [impressive rate](https://medium.com/analytics-vidhya/openai-gpt-3-language-models-are-few-shot-learners-82531b3d3122). With access to such massive amounts of data, the question has shifted from “how to construct the best model” to “how do you feed these models”.

Both trends are supported by results from [Kaplan et al.](https://arxiv.org/abs/2001.08361), who show that the architecture matters less, and the real lift comes from the data.


## Commoditized Architectures to One-Model-to-Rule-them-Al

- Over the last few years, the natural language processing community has landed on the [Transformer](https://arxiv.org/pdf/1706.03762.pdf), explained very well in this [blog](https://jalammar.github.io/illustrated-transformer/), as being the *commoditized* language architecture. The vision community landed on the convolutional neural network, explained more in this [blog](https://towardsdatascience.com/convolutional-neural-networks-explained-9cc5188c4939).
- [Ramachandran et al](https://arxiv.org/pdf/1906.05909.pdf) showed that the CNN and self-attention block in Transformers could actual be the same, and this was capitalized with the [Vision Transformers](https://arxiv.org/pdf/2010.11929.pdf) that used a Transformer to train an image classification model to achieve near or above state-of-the-art results.
- These architecutres are still complex and expensive to use. Researchers missed the good-old-days of MLP layers. Exciting recent work has shown that even the Transformer can be replaced by a sequence of MLPs in [gMLP](https://arxiv.org/pdf/2105.08050.pdf) and [MLP-Mixer](https://arxiv.org/pdf/2105.01601.pdf).

## More Focus on the Data Being Fed

As the goal is to feed as much knowledge to these commoditized models as possible, recent work has explored multi-modal applications that use both vision and text data.
- [Wu Dao 2.0](https://www.engadget.com/chinas-gigantic-multi-modal-ai-is-no-one-trick-pony-211414388.html) is the Chinese 1.75T parameter MoE model with multimodal capabilities.
- [DALL-E](https://openai.com/blog/dall-e/) & [CLIP](https://openai.com/blog/clip/) are two other multi-modal models.

Other groups are trying to curate a better pretraining dataset
- [The Pile](https://pile.eleuther.ai) is a new massive, more diverse dataset for training language models than the standard Common Crawl.
- [Huggingface BigScience](https://docs.google.com/document/d/1BIIl-SObx6tR41MaEpUNkRAsGhH7CQqP_oHQrqYiH00/edit) is a new effort to establish good practices in data curation.

More focus is being put into the kind of tokenization strategies can be used to further unify these models. While language tasks typically deal with [WordPiece](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/37842.pdf) tokens or [BytePairEncoding (BPE)](https://arxiv.org/pdf/1508.07909.pdf) tokens, recent work explores [byte-to-byte](https://arxiv.org/pdf/2105.13626.pdf) strategies that work on individual bytes or characters, requiring no tokenization. In vision, tokens are usually [patches](https://arxiv.org/pdf/2010.11929.pdf) in the image.

## Theoretical Foundations

- [Limitations of Autoregressive Models and Their Alternatives](https://arxiv.org/abs/2010.11939) explores the theoretical limitations of autoregressive language models in the inability to represent "hard" language distributions.
- [Provable Limitations of Acquiring Meaning from Ungrounded Form: What Will Future Language Models Understand?](https://arxiv.org/pdf/2104.10809.pdf) explores the theoretical limitations of autoregressive language models in the inability to represent "hard" language distributions.

## Success Stories

- Companies like [OpenAI](https://openai.com), [Anthropic](https://www.anthropic.com), [Cohere](https://cohere.ai) see building universal models as part of their core business strategy.
- Lots of companies emerging that rely on APIs from these universal model companies to build applications on top e.g. [AI Dungeon](https://play.aidungeon.io/main/landing). A long list from OpenAI at this [link](https://openai.com/blog/gpt-3-apps/).
