<h1 id="applications">Applications</h1>

<h2 id="named-entity-linking">Named Entity Linking</h2>

Named entity linking (NEL) is the task of linking ambiguous mentions in text to entities in a knowledge base. NEL is a core preprocessing step in downstream applications, including search and question answering.

- Pre-deep-learning approaches to NEL have been [rule-based](https://www.aclweb.org/anthology/X96-1053.pdf) or leverage statistical techniques and manual feature engineering to filter and rank candidates ([survey paper](https://arxiv.org/abs/1910.11470)).
- In recent years, deep learning systems have become the new standard ([overview paper](https://dl.acm.org/doi/10.1145/3183713.3196926) of deep learning approaches to entity disambiguation and entity matching problems). The most recent state-of-the-art models generally rely on deep contextual word embeddings with entity embeddings. For example, [Pre-training of Deep Contextualized Embeddings of Words and Entities for Named Entity Disambiguation](https://arxiv.org/pdf/1909.00426v1.pdf) and [Empirical Evaluation of Pretraining Strategies for Supervised Entity Linking](https://arxiv.org/pdf/2005.14253.pdf).
- We've seen a recent shift in simplifying the model even more to just use tranformers without explicit entity embeddings with models like [BLINK](https://arxiv.org/pdf/1911.03814.pdf) (uses a bi-encoder) and the [Dual and Cross-Attention Encoders](https://arxiv.org/pdf/2004.03555.pdf) (uses cross-encoder).
- Other trends have been to enhance the training data further. The system [Bootleg](https://arxiv.org/pdf/2010.10363.pdf) system uses weak labeling of the training data to noisily assign entity links to mentions, increasing performance over rare entities.
- Ikuya Yamada has a wonderful GitHub [survey](https://github.com/izuna385/Entity-Linking-Recent-Trends) of recent trends in Entity Linking

<h2 id="medical-imaging">Medical Imaging</h2>

- Sensitive to inputs, not models

  - The variation of imaging configurations (e.g. [site locations](https://arxiv.org/pdf/2002.11379.pdf)), hardware, and processing techniques (e.g. [CT windowing](https://pubs.rsna.org/doi/abs/10.1148/ryai.2021200229)) lead to large performance shifts
  - Recent medical imaging challenges (segmentation: [knee](https://arxiv.org/pdf/2004.14003.pdf), [brain](https://arxiv.org/pdf/1811.02629.pdf), reconstruction: [MRI](https://arxiv.org/abs/2012.06318)), found that, to a large extent, the choice of model is less important than the underlying distribution of data (e.g. disease extent)

- Towards multi-modal data fusion
  - Radiologist reports (and more generally text) have been used to improve learned visual representations (e.g. [ConVIRT](https://arxiv.org/abs/2010.00747)) and to source weak labels in annotation-scarce settings (e.g. ([PET/CT](https://www-nature-com.stanford.idm.oclc.org/articles/s41467-021-22018-1)))
  - Auxiliary features from other rich, semi-structured data, such as [electronic health records (EHRs)](https://www-nature-com.stanford.idm.oclc.org/articles/s41746-020-00341-z), successfully complemented standard image representations

- [Data Models for Dataset Drift Controls in Machine Learning With Images](https://arxiv.org/abs/2211.02578)
   - Drift synthesis enables the controlled generation of physically faithful drift test cases. The experiments presented here show that the average decrease in model performance is ten to four times less severe than under post-hoc augmentation testing.
   - The gradient connection between task and data models allows for drift forensics that can be used to specify performance-sensitive data models which should be avoided during deployment of a machine learning model.
   - Drift adjustment opens up the possibility for processing adjustments in the face of drift. This can lead to speed up and stabilization of classifier training at a margin of up to 20% in validation accuracy.

[comment]: <> (## Image Segmentation)

[comment]: <> ([Sarah])

<h2 id="computational-biology">Computational Biology</h2>

- Collecting the right data for training and evalution can require wetlab work – especially in computational drug discovery.
  - [A Deep Learning Approach to Antibiotic Discovery](<https://www.cell.com/cell/pdf/S0092-8674(20)30102-1.pdf>)
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

<h2 id="remote-sensing">Remote Sensing</h2>
- [Data Models](https://arxiv.org/abs/2211.02578)
   - Tolerancing allows for for prospective validation of machine learning task model under physically faithful dataset drifts
   - Differentiable data models allow the optimization of the data generating process
