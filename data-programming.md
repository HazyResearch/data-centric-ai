# Data Programming & Weak Supervision

Many modern machine learning systems require large, labeled datasets to be successful but producing such datasets is time-consuming and expensive. Instead, weaker sources of supervision, such as [crowdsourcing](https://papers.nips.cc/paper/2011/file/c667d53acd899a97a85de0c201ba99be-Paper.pdf), [distant supervision](https://www.aclweb.org/anthology/P09-1113.pdf), and domain experts' heuristics like [Hearst Patterns](https://people.ischool.berkeley.edu/~hearst/papers/coling92.pdf) have been used since the 90s.
 
However, these were largely regarded by AI and AI/ML folks as ad hoc or isolated techniques. The effort to unify and combine these into a data centric viewpoint started in earnest with [data programming](https://arxiv.org/pdf/1605.07723.pdf) embodied in the [Snorkel system](http://www.vldb.org/pvldb/vol11/p269-ratner.pdf), now an [open-source project](http://snorkel.org) and [thriving company](http://snorkel.ai). In Snorkel's conception, users specify multiple labeling functions that each represent a noisy estimate of the ground-truth label. Because these labeling functions vary in accuracy, coverage of the dataset, and may even be correlated, they are combined and denoised via a latent variable graphical model. The technical challenge is thus to learn accuracy and correlation parameters in this model, and to use them to infer the true label to be used for downstream tasks.

Data programming builds on a long line of work on parameter estimation in latent variable graphical models. Concretely, a generative model for the joint distribution of labeling functions and the unobserved (latent) true label is learned. This label model permits aggregation of diverse sources of signal, while allowing them to have varying accuracies and potential correlations.

An overview of the weak supervision pipeline can be found in this [Snorkel blog post](https://www.snorkel.org/blog/weak-supervision), including how it compares to other approaches to get more labeled data and the technical modeling challenges. These [Stanford CS229 lecture notes](https://mayeechen.github.io/files/wslecturenotes.pdf) provide a theoretical summary of how graphical models are used in weak supervision.


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

- This [Snorkel blog post](https://www.snorkel.org/blog/weak-supervision) provides an overview of the weak supervision pipeline, including how it compares to other approaches to get more labeled data and the technical modeling challenges.
- [These Stanford CS229 lecture notes](https://mayeechen.github.io/files/wslecturenotes.pdf) provide a more theoretical summary of how graphical models are used in weak supervision.


<h2 id="weak-supervision-success-stories"> Success Stories </h2>

- [Google](https://arxiv.org/pdf/1812.00417.pdf) used a weak supervision system in Ads and YouTube based on Snorkel. In just tens of minutes, this system utilizes diverse organizational resources to create classifiers with performance equivalent to those trained on tens of thousands of hand-labeled examples over millions of datapoints.

- [Intel](https://ajratner.github.io/assets/papers/Osprey_DEEM.pdf) expanded weak supervision interfaces for non-programmers and was able to replace six months of crowdworker labels while improving precision by double digits.
- [G-mail](http://cidrdb.org/cidr2020/papers/p31-sheng-cidr20.pdf) migrated their privacy-safe rule-based information extraction system Juicer to a Software 2.0 design with weak supervision, which surpassed the previous system in terms of precision and recall of the extractions and was also found to be much easier to maintain.

- [Facebook](https://ai.facebook.com/blog/billion-scale-semi-supervised-learning/) achieved new state-of-the-art performance on academic benchmarks for image and video classification by weakly supervising training labels in the billions.

- [Stanford Radiology](https://arxiv.org/pdf/1903.11101.pdf) used a cross-modal weak supervision approach to weakly supervise training labels from text reports and then train an image model for the associated radiology images.

- This [Software 2.0 blog post](https://hazyresearch.stanford.edu/software2) summarizes other successes for data programming.