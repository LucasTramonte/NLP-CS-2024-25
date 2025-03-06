import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# 1) Load train and test data
###############################################################################    

train_df = pd.read_csv("../Assets/Data/train_submission.csv")  
test_df  = pd.read_csv("../Assets/Data/test_without_labels.csv")  

###############################################################################
# 2) Compute word counts for train and test
###############################################################################
train_df["word_count"] = train_df["Text"].apply(lambda x: len(str(x).split()))
test_df["word_count"]  = test_df["Text"].apply(lambda x: len(str(x).split()))

train_total = len(train_df)
test_total  = len(test_df)

###############################################################################
# 3) Bin word counts:
#    0-9, 10-19, 20-29, 30-39, 40-99, 100+
###############################################################################
bins   = [0, 10, 20, 30, 40, 100, float("inf")]
labels = ["0-9", "10-19", "20-29", "30-39", "40-99", "100+"]

train_df["wc_bin"] = pd.cut(train_df["word_count"], bins=bins, labels=labels, right=False)
test_df["wc_bin"]  = pd.cut(test_df["word_count"],  bins=bins, labels=labels, right=False)

###############################################################################
# 4) Compute the percentage in each bin for train & test
###############################################################################
train_bin_counts = train_df["wc_bin"].value_counts().reindex(labels, fill_value=0)
test_bin_counts  = test_df["wc_bin"].value_counts().reindex(labels, fill_value=0)

# Convert to percentage
train_bin_perc = (train_bin_counts / train_total) * 100
test_bin_perc  = (test_bin_counts / test_total) * 100

print("Train bin percentages (%):")
print(train_bin_perc)
print("\nTest bin percentages (%):")
print(test_bin_perc)

###############################################################################
# 5) Plot a horizontal bar chart comparing train vs. test side by side
###############################################################################
N     = len(labels)  # 6 bins
ind   = np.arange(N)
width = 0.4

plt.figure(figsize=(7,5))
for i in range(N):
    # Train bars (light blue) on the left side
    plt.barh(ind[i] - width/2, train_bin_perc.iloc[i], height=width, color="lightblue",
             label=None if i>0 else "Train")
    # Test bars (dark green) on the right side
    plt.barh(ind[i] + width/2, test_bin_perc.iloc[i], height=width, color="darkgreen",
             label=None if i>0 else "Test")

plt.yticks(ind, labels)
plt.xlabel("Percentage %")
plt.ylabel("Word Count Range")
plt.title("Word Count Distribution: Train vs. Test (in %)")
plt.legend(loc="lower right")
plt.gca().invert_yaxis()  # So the first bin appears at the top
plt.tight_layout()
plt.savefig("train_vs_test_word_count_bins_percent.png", dpi=300)
plt.close()

print("\nAnalysis completed. Word count distribution saved as 'train_vs_test_word_count_bins_percent.png'.")
