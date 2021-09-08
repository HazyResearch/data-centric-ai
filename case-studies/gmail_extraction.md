## Structured Extractions in Gmail: From Heuristics to Learned Extractors

Extracting structured data from emails can enable several assistive experiences, such as reminding the user when a bill payment is due, answering queries about the location of a dentist appointment or updating users about when an online purchase is scheduled for delivery. Juicer is a privacy-safe extraction system over email serving more than a billion users worldwide ([Ying et al., 2018](https://research.google/pubs/pub46991/)).  The three key principles we followed when designing the system are: scale, simplicity, and privacy-safe.

The extraction system was first built using hand-crafted rules, which is hard to maintain and scale. The recent advances in machine learning (ML) makes it possible to build a 'software 2.0' system which focuses on training models to learn from data instead of explicitly writing code for the required behavior. In the extraction system's case, we tried to use the extractions from the existing rule-based system as training data to learn ML models that in turn replace all the machinery for the rule-based system ([Ying et al., 2020](https://research.google/pubs/pub48846/)). 

The ML-based extraction system ([Ying et al., 2018](https://research.google/pubs/pub46991/)) consists of two kinds of classifiers: 1. a multi-label classifier that tells us which category the email belongs to and thus determines the sets of fields to extract; 2. a field classifier that finds the corresponding value for each field. For each field, a set of text spans with a certain type of annotation will be selected as the candidates. For example, the candidates for the event start time field are all the text spans that are annotated as time. Then the classifier picks one of these candidates as the field value.

## Data Quality is the Key

When building an ML-based extraction system on top of the data from a rule-based extraction system, the first sets of models trained could only cover 6% of extractions from the rule-based extraction system. We tried to tune various hyperparameters of the models but that didn’t help much.

The biggest improvement came from managing data quality along 3 different aspects: 1. building high coverage candidates; 2. obtaining high-quality groundtruth; 3. high precision matching between ground truth and candidates. Since we have field training data for more than 10 fields, we instrumented a standard system that is shared by all fields to produce stats that help us understand the potential problem of the data. For example, if stats indicate 0 candidates match ground truth in many emails, it indicates either candidate coverage is low or matching between ground truth and candidates needs more work.

After various data management on candidate, groundtruth and candidate-groundtruth matching, we were able to produce high-quality training data and increase the extraction volume from 6% to  >100% of rule-based extractions with higher precision. See table below about precision and volume increase for Event and Purchase extraction tasks.

![results](https://github.com/oliviasheng/data-centric-ai/blob/patch-1/case-studies/gmail_extraction_case_study_result.png)

## Summary

By solving the challenges from data management in various stages, we built an ML-based extraction system with higher precision and recall to replace the hand-crafted rule based extraction system. The work further reinforces the argument that a critical ingredient of a real-world machine learning system is managing training data.


## References

1. [Sheng, Ying, et al. "Migrating a Privacy-Safe Information Extraction System to a Software 2.0 Design." (2020).][ref1]
2. [Sheng, Ying, et al. "Anatomy of a privacy-safe large-scale information extraction system over email." Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018.][ref2]

[ref1]: https://research.google/pubs/pub48846/
[ref2]: https://research.google/pubs/pub46991/
