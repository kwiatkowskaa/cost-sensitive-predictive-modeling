# Cost-Sensitive Campaign Modeling

This project focuses on building a cost-sensitive predictive model for marketing campaign targeting.

The main objective is not only to predict which customers are likely to accept an offer, but also to maximize the overall business profit while minimizing unnecessary marketing costs and customer fatigue.

Unlike standard classification tasks, this problem introduces two important business constraints:

- Every variable used in the model has an acquisition and processing cost.
- Only a limited number of customers can be contacted.

The project aims to find an optimal balance between:
- identifying customers with high conversion probability,
- reducing false positive targeting,
- minimizing the number of variables used by the model.

The final evaluation metric is based on a business profit function:

Score = (TP × 10) - (FP × 5) - (Variables × 200)

where:
- TP = correctly targeted customers,
- FP = incorrectly targeted customers,
- Variables = number of features used in the final model.

## Project Goals

- Build a reproducible machine learning pipeline
- Compare different feature selection strategies
- Evaluate multiple classification models
- Optimize customer targeting thresholds
- Maximize the final business score
