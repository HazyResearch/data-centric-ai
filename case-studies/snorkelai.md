<!--
Name: Snorkel AI case study on data-centric AI.
Author: Snorkel AI Team
--> 

# Snorkel AI


## About Snorkel AI

Snorkel AI started as a research project in the Stanford AI Lab in 2015. Initially set out to explore a higher-level interface to machine learning through training data. There are over 50 peer-reviewed publications such as ICML, Nature, ICLR, IEEE, NeurIPS, and [many more](https://snorkel.ai/technology/), powering the core technology behind Snorkel Flow. Snorkel’s technology has developed with and deployed at Google, Intel, Apple, two of the three top US banks, the U.S. Department of Justice, and other leading organizations.


## Snorkel Flow, the first data-centric AI platform

Snorkel Flow is an AI development platform powered by weak supervision [[2](https://ai.stanford.edu/blog/weak-supervision/)] and programmatic data labeling [[3](https://bencw99.github.io/files/kdd2019_dcclworkshop.pdf)] approaches. Using Snorkel Flow, data science teams can collaborate with subject matter experts to rapidly build highly accurate AI applications. It allows users to create and manage massive amounts of training data, train models, analyze, improve performance by iterating on not just models but also training data and deploy - all in one platform.


## Where does Snorkel Flow Excel at?



* Label and build training data programmatically in hours instead of months or even years of hand-labeling.
* Integrate and manage programmatic training data from all sources, including data cleansing and data slicing.
* Train and deploy state-of-the-art machine learning models in-platform or via a Python SDK.
* Analyze and monitor model performance to rapidly identify and correct error modes in the data fast.

Learn more about the [Snorkel Flow platform](https://snorkel.ai/platform/).


## SuperGLUE Case Study

Using standard models (i.e., pre-trained BERT) and minimal tuning, the Snorkel AI team was able to leverage key abstractions for programmatically build and manage training data to achieve a **state-of-the-art result on SuperGLUE**—a newly curated benchmark; with six tasks for evaluating "general-purpose language understanding technologies.

A new SOTA was achieved using programming abstractions on the SuperGLUE Benchmark and four of its components tasks. [SuperGLUE](https://super.gluebenchmark.com/) is similar to [GLUE](https://gluebenchmark.com/) but contains "more difficult tasks, which are chosen to maximize difficulty and diversity. These tasks are selected to show a substantial headroom gap between a strong BERT-based baseline and human performance."

After reproducing the BERT++ baselines, we minimally tune these models (baseline models, default learning rate, and so on.) and find that with applications of the above programming abstractions, we notice improvements of +4.0 points on the SuperGLUE benchmark (indicating a 21% reduction of the gap to human performance).

The paper [[5](https://arxiv.org/abs/1905.00537)] also gives updates on Snorkel's industry use cases with even more applications at scale, for example, [Google in Snorkel Drybell](https://ai.googleblog.com/2019/03/harnessing-organizational-knowledge-for.html) to scientific work in [MRI classification](https://www.nature.com/articles/s41467-019-11012-3) and [automated Genome-wide association study (GWAS) curation](https://ai.stanford.edu/~kuleshov/papers/gwaskb-manuscript.pdf), both accepted in [Nature Comms](https://www.nature.com/ncomms/).


## [Industrial Case Studies](https://snorkel.ai/case-studies/)



* Google has used Snorkel to replace 100k+ hand-annotated labels in critical machine learning pipelines.  
* A top US bank uses Snorkel Flow to quickly build AI applications that classify and extract information from their documents.
* Apple built applications with an internal Snorkel-based system that answered billions of queries in multiple languages and processed trillions of records with up to 2.9x fewer errors.
* A Fortune 500 Biotech pioneer leveraged Snorkel Flow to extract critical chronic disease data from clinical trials, accurately processing 300K documents in minutes.


## References

[1] "Snorkel: Rapid Training Data Creation with Weak Supervision." Alex Ratner, Stephen H. Bach, Henry Ehrenberg, Jason Fries, Sen Wu, Chris Re, Stanford University, https://arxiv.org/pdf/1711.10160.pdf

[2] "Weak Supervision: A New Programming Paradigm For Machine Learning." Alex Ratner, Paroma Varma, Braden Hancock, Chris Ré, et al., SAIL Blog, 2019, https://ai.stanford.edu/blog/weak-supervision/

[3] "Interactive Programmatic Labeling for Weak Supervision." Benjamin Cohen-Wang, Stephen Mussmann, Alex Ratner, Chris Ré, KDD, 2019, https://bencw99.github.io/files/kdd2019_dcclworkshop.pdf

[4] "Snorkel DryBell: A Case Study in Deploying Weak Supervision at Industrial Scale." Stephen H. Bach, Daniel Rodriguez, Yintao Liu, Chong Luo, Haidong Shao, Cassandra Xia, Souvik Sen, Alexander Ratner, Braden Hancock, Houman Alborzi, Rahul Kuchhal, Christopher Ré, Rob Malkin, SIGMOD, 2019, https://arxiv.org/abs/1812.00417

[5] Wang, Alex, et al. "SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems." 2019. SuperGLUE consists of 6 datasets: the Commitment Bank (CB, De Marneffe, et al., 2019), Choice Of Plausible Alternatives (COPA, Roemmele, et al., 2011), the Multi-Sentence Reading Comprehension dataset (MultiRC, Khashabi, et al., 2018), Recognizing Textual Entailment (merged from RTE1, Dagan et al. 2006, RTE2, Bar Haim, et al., 2006, RTE3, Giampiccolo, et al., 2007, and RTE5, Bentivogli, et al., 2009), Word in Context (WiC, Pilehvar, and Camacho-Collados, 2019), and the Winograd Schema Challenge (WSC, Levesque, et al., 2012).
