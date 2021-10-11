<h1 id="emerging">Emerging Trends</h1>

<h2 id="interactive-machine-learning">Interactive Machine Learning</h2>
With models getting larger and costing more to train, there's a growing need to interact with the model and quickly iterate on its performance before a full training run.

- **Explanatory interactive learning** Can we, by interacting with models during training, encourage their explanations to line up with our priors on what parts of the input are relevant?

   - [Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations](https://arxiv.org/pdf/1703.03717.pdf)
   - [Explanatory Interactive Machine Learning](https://ml-research.github.io/papers/teso2019aies_XIML.pdf)
   - [Making deep neural networks right for the right scientific reasons by interacting with their explanations](https://www.nature.com/articles/s42256-020-0212-3)
    
- **[Mosaic](https://github.com/robustness-gym/mosaic)** makes it easier for ML practitioners to interact with high-dimensional, multi-modal data. It provides simple abstractions for data inspection, model evaluation and model training supported by efficient and robust IO under the hood. Mosaic's core contribution is the DataPanel, a simple columnar data abstraction. The Mosaic DataPanel can house columns of arbitrary type – from integers and strings to complex, high-dimensional objects like videos, images, medical volumes and graphs.

   - [Introducing Mosaic](https://www.notion.so/Introducing-Mosaic-64891aca2c584f1889eb0129bb747863) (blog post)
   - [Working with Images in Mosaic](https://drive.google.com/file/d/15kPD6Kym0MOpICafHgO1pCt8T2N_xevM/view?usp=sharing) (Google Colab)
   - [Working with Medical Images in Mosaic](https://colab.research.google.com/drive/1UexpPqyXdKp6ydBf87TW7LtGIoU5z6Jy?usp=sharing) (Google Colab)

<h2 id="massive-scale-ml">Massive Scale Models</h2> 

With the ability to train models without needing labelled data through self-supervision, the focus became on scaling models up and training on more data.

- [GPT-3](https://arxiv.org/abs/2005.14165.pdf) was the first 170B parameter model capable of few-shot in-context learning developed by OpenAI.
- [Moore's Law for Everything](https://moores.samaltman.com) is a post about scale and its effect on AI / society.
- [Switch Transformers](https://arxiv.org/pdf/2101.03961.pdf) is a mixture of experts for training massive models beyond the scale of GPT-3.

<h2 id="observational-supervision">Observational Supervision</h2>

The way experts interact with their data (e.g. a radiologist’s eye movements) contains rich information about the task (e.g. classification difficulty), and the expert (e.g. drowsiness level).
With the current trend of wearable technology (e.g. AR with eye tracking capability), the hardware needed to collect such human-data interactions is expected to become more ubiquitous, affordable, and standardized.
In observational supervision, we investigate how to extract the rich information embedded in the human-data interaction, to either supervise models from scratch, or to improve model robustness.

Interesting works have collected observational signals such as:

- Eye tracking data in medicine:
  - Chest X-Rays, dictation audio, bounding boxes, and gaze on 3,032 images by R. Lanfredi et al. [dataset](https://www.physionet.org/content/reflacx-xray-localization/1.0.0/) 
  - Chest X-Rays, reports, dictation audio, and gaze on 1,083 images by A. Karargyris et al. [dataset](https://physionet.org/content/egd-cxr/1.0.0/)
  - Two medical datasets on Chest X-Rays and brain MRI by K. Saab et al. [dataset](https://github.com/HazyResearch/observational/tree/main/gaze_data)
- Eye tracking plus brain activity in NLP (Zuco [dataset](https://www.nature.com/articles/sdata2018291.pdf))


Critical papers in observational supervision:

- Some of the pioneering work on using gaze data. N. Hollenstein and C. Zhang showed how to use gaze data to improve NLP models [paper](https://arxiv.org/pdf/1902.10068.pdf).
- Improving zero-shot learning with gaze by N. Karasseli et al. [paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Karessli_Gaze_Embeddings_for_CVPR_2017_paper.pdf)
- Weak supervision and multi-task learning with gaze by K. Saab et al. [paper](https://web.stanford.edu/~ksaab/media/MICCAI_2021.pdf)
