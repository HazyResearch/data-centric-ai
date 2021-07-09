# Contrastive Learning

<h2 id="contrastive-theory">Theoretical Foundations</h2>

Contrastive learning works by optimizing a loss function that pulls together similar points ("positive" pairs) and pushes apart dissimilar points ("negative" pairs). Compared to its empirical success in self-supervision, a theoretical understanding of contrastive learning is relatively lacking in terms of what sort of representations are learned by minimizing contrastive loss, and what these representations guarantee on downstream tasks.

- [Representations induced on the hypersphere](https://arxiv.org/pdf/2005.10242.pdf): assuming that the representations to learn are constrained to a hypersphere, the contrastive loss function is closely connected to optimizing for "alignment" (positive pairs map to the same representation) and "uniformity" (representations are "spread out" as much as possible on the hypersphere to maintain as much as information as possible).
- [Downstream performance](https://arxiv.org/pdf/1902.09229.pdf): suppose that similar pairs belong to the same latent subclass, and that the downstream task aims to classify among some of these latent subclasses. Then, downstream loss of a linear classifier constructed using mean representations can be expressed in terms of the unsupervised contrastive loss.
- [Debiasing contrastive learning](https://arxiv.org/pdf/2007.00224.pdf) and [using hard negative samples](https://openreview.net/pdf?id=CR1XOQ0UTh-): in unsupervised settings, negative pairs are constructed by selecting two points at random i.i.d. This can result in the two points actually belonging to the same latent subclass, but this can be corrected via importance weighting. Moreover, even within different latent subclasses, some negative samples can be "harder" than others and enforce better representations.

<h2 id="contrastive-applications">Applications</h2>