## About Landing AI
[Landing AI](https://landing.ai/) is building a data-centric MLOps platform for computer vision. Focusing on manufacturing visual inspection, LandingLens enables ML teams to build, deploy, and scale computer vision applications 10x faster than before, and rapidly create significant ROI. Founded by Dr. Andrew Ng, co-founder of Coursera, and founding lead of Google Brain, the team at Landing AI is uniquely positioned to help companies across the globe successfully move their AI projects from proof of concept to full-scale production.

## Data-Centric MLOps Platform
In the past, the team at Landing AI helped many manufacturing customers build and deploy machine learning solutions for visual inspection tasks. One issue we have commonly seen is, when facing a new problem, engineers tend to throw state-of-the-art models at it. When they observe a gap between the current model performance and the target, it is tempting to tune the mdoel architecture and hyperparameters through hundreds of training runs. However, we have consistently found this to be a waste of time. Instead, efforts like cleaning inconsistent labels, fixing confusions in labeling books, collecting new data to fix imbalance class distribution often lead to better performing models. 

In fact, the team at Landing AI evaluated the model-centric approach and the data-centric approach on datasets of different domains. Over time, the data-centric approach that prioritize data quality consistently out-perform the model-centric approach in those experiments, as shown in the table below. 


|               | Steel defect detection | Solar panel     | Surface inspection |
| ------------- | ---------------------- | --------------- | ------------------ |
| Baseline      | 76.2%                  | 75.68%          | 85.1%              |
| Model-centric | +0% (76.2%)            | +0.04% (75.72%) | +0.0% (85.1%)      |
| Data-centric  | +16.9% (93.1%)         | +3.06% (78.74)  | +0.4% (85.5%)      |

Therefore, Landing AI team puts lots of investment into the data-centric approach and builds [LandingLens](https://landing.ai/platform/), a MLOps platform that offers end-to-end capabilities to label data, train and deploy computer vision solutions. With data quality being a key to the success of production AI systems, LandingLens is equipped with a host of specially designed data preparation tools and workflows that help users achieve optimal data accuracy and consistency.
