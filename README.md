# NLP-CS-2024-25

Code developed for CentraleSupélec Kaggle competition in the Advanced NLP course

---

<div style="display: flex; gap: 10px;">
  <a href="https://www.kaggle.com/competitions/nlp-cs-2025/overview">[Kaggle competition]</a>
  <a href="https://github.com/PierreColombo/NLP-CS-2024-25/tree/main/kaggle_project">[Instructions]</a>
  <a href="https://plmlatex.math.cnrs.fr/3164218448nqxdntshvkgm">[Report]</a>
</div>

## Competition Overview

The main goal of this competition is to build an effective text classifier.

## Usage

1. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

2. **Run the scripts**:
    - To train the any model:
        ```sh
        python <model_script>.py --train_dataset Assets/Data/train_submission.csv --train
        ```
    - To predict the language of a single text using any model:
        ```sh
        python <model_script>.py --predict "Your text here"
        ```
        Replace `<model_script>` with the appropriate script name (e.g., Camembert.py, classifier_Tfidf_LR.py, XLM-RoBERTa.py, classifier_Tfidf_Xgboost.py).
    - To generate a submission file using any model:
        ```sh
        python <model_script>.py --test_dataset Assets/Data/test_submission.csv --submission Assets/Outputs/submission.csv
        ```

We are utilizing the [DCE](https://dce.pages.centralesupelec.fr/) GPU provided by CentraleSupélec for training our models.

## Models

This project includes implementations of several models for language classification:

| Model                          | Evaluation Accuracy  | Kaggle Accuracy                                           |
|--------------------------------|-----------|-------------------|
| **CamemBERT**                  | -   | 0.70534 |
| **TF-IDF with Logistic Regression** | -   | 0.71067 |
| **XLM-RoBERTa**                | -   | 0.88269 |
| **TF-IDF with XGBoost**        | -         | - |

## To Do (just ideas)

- Drop out to reduce overfit
- Evaluate all models to compare with Kaggle Accuracy 
- Submit XGBoost
- Modularize the codes with Tfidf