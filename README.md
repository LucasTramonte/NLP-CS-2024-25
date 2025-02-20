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

2. **Run the main script**:
    - To train the model:
        ```sh
        python main.py --train_dataset Assets/Data/train_submission.csv --train
        ```
    - To evaluate the model: 
        ```sh
        python main.py --test_dataset Assets/Data/test_submission.csv --evaluate
        ```
    - To predict the language of a single text:
        ```sh
        python main.py --predict "Your text here"
        ```
    - To generate a submission file:
        ```sh
        python main.py --test_dataset Assets/Data/test_submission.csv --submission Assets/Outputs/submission.csv
        ```
