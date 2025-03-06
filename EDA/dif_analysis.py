import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# 1) Load data
###############################################################################

train_df = pd.read_csv("../Assets/Data/train_submission.csv")  # Must have at least 'Label'
test_df  = pd.read_csv("../Assets/Outputs/Submission/submission_file.csv")  # Must have 'Label'

###############################################################################
# 2) Compute percentage distribution for train and test
###############################################################################
train_counts = train_df["Label"].value_counts(normalize=True) * 100  # percentage
test_counts  = test_df["Label"].value_counts(normalize=True)  * 100  # percentage

# Combine into a single DataFrame for comparison
all_labels = list(set(train_counts.index).union(set(test_counts.index)))

df_compare = pd.DataFrame({
    "Train (%)": [train_counts.get(lbl, 0) for lbl in all_labels],
    "Test (%)":  [test_counts.get(lbl, 0)  for lbl in all_labels]
}, index=all_labels)

###############################################################################
# 3) Compute absolute difference in percentage, find average difference
###############################################################################
df_compare["Diff"] = (df_compare["Train (%)"] - df_compare["Test (%)"]).abs()

# Average difference across all languages
avg_diff = df_compare["Diff"].mean()
print(f"Average difference in language percentage distribution: {avg_diff:.2f}%")

###############################################################################
# 4) Identify the top 5 languages with the largest difference
###############################################################################
df_compare.sort_values(by="Diff", ascending=False, inplace=True)
top_5 = df_compare.head(5)
print("\nTop 5 languages with the biggest difference:")
print(top_5[["Train (%)", "Test (%)", "Diff"]])
