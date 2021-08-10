# Data Selection
_This area is a stub, you can help by improving it._


## Data Valuation
Quantifying the contribution of each training datapoint to an end model is useful in a number of settings: 
1. in __active learning__ knowing the value of our training examples can help guide us in collecting more data
2. when __compensating__ individuals for the data they contribute to a training dataset (_e.g._ search engine users contributing their browsing data or patients contributing their medical data)
3. for __explaining__ a model's predictions and __debugging__ its behavior.

However, data valuation can be quite tricky.
The first challenge lies in selecting a suitable criterion for quantifying a datapoint's value. Most criteria aim to measure the gain in model performance attributable to including the datapoint in the training dataset. A common [approach](https://conservancy.umn.edu/handle/11299/37076), dubbed "leave-one-out", simply computes the difference in performance between a model trained on the full dataset and one trained on the full dataset minus one example. Recently, [Ghorbani _et al._](https://proceedings.mlr.press/v97/ghorbani19c/ghorbani19c.pdf) proposed a data valuation scheme based on the [Shapley value](https://en.wikipedia.org/wiki/Shapley_value), a classic solution in game theory for distributing rewards in cooperative games. Empirically, Data Shapley valuations are more effective in downstream applications (_e.g._ active learning) than "leave-one-out" valuations. Moreover, they have several intuitive properties not shared by other criteria.


Computing exact valuations according to either of these criteria requires retraining the model from scratch many times, which can be prohibitively expensive for large models. Thus, a second challenge lies in finding a good approximation for these measures. [Influence functions](https://arxiv.org/pdf/1703.04730.pdf) provide an efficient estimate of the "leave-one-out" measure that only requires on access to the model's gradients and hessian-vector products. Shapley values can be estimated with Monte Carlo samples or, for models trained via stochastic gradient descent, a simple gradient-based [approach](https://proceedings.mlr.press/v97/ghorbani19c/ghorbani19c.pdf). 
 