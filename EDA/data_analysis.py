import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pycountry
import os
import io

###############################################################################
# 1) Function to get language name from ISO 639-3 code
###############################################################################

def get_language_name(iso_code):
    try:
        language_name = pycountry.languages.get(alpha_3=iso_code).name
        return f"{language_name} ({iso_code})"
    except AttributeError:
        return iso_code

###############################################################################
# 2) Load the data
###############################################################################
train_df = pd.read_csv("../Assets/Data/train_submission.csv")  # Must have at least 'Label'
pred_df  = pd.read_csv("../Assets/Outputs/Submission/submission_file.csv")   # Must have at least 'ID', 'Label'

train_total = len(train_df)
test_total  = len(pred_df)

# Capture the output of df.info()
buffer = io.StringIO()
train_df.info(buf=buffer)
info_str = buffer.getvalue()

# Save dataset information to a text file
with open("../Assets/Outputs/EDA/dataset_info.txt", "w") as f:
    f.write(info_str)
    f.write("\n")
    f.write("Missing values in each column:\n")
    f.write(str(train_df.isnull().sum()))
    f.write("\n\n")
    f.write("Number of duplicated rows:\n")
    f.write(str(train_df.duplicated().sum()))
    f.write("\n\n")
    f.write("Number of unique languages:\n")
    f.write(str(train_df["Label"].nunique()))

# Print missing values, duplicated rows, and unique languages
print("Missing values in each column:",train_df.isnull().sum())
print("Number of duplicated rows:", train_df.duplicated().sum())
print("Number of unique languages:",train_df["Label"].nunique())

###############################################################################
# 3) Training distribution (Top 10) by frequency
###############################################################################
train_counts = train_df["Label"].value_counts().sort_values(ascending=False)
train_perc   = (train_counts / train_total) * 100
top_10_train_perc = train_perc.head(10)

print("Training language distribution (Top 10) in %:")
print(top_10_train_perc)

train_codes  = top_10_train_perc.index
train_langs  = [get_language_name(c) for c in train_codes]
train_values = top_10_train_perc.values


###############################################################################
# 4) Test predictions distribution (Top 10) by frequency
###############################################################################
test_counts = pred_df["Label"].value_counts().sort_values(ascending=False)
test_perc   = (test_counts / test_total) * 100
top_10_test_perc = test_perc.head(10)

print("\nTest predicted language distribution (Top 10) in %:")
print(top_10_test_perc)

test_codes  = top_10_test_perc.index
test_langs  = [get_language_name(c) for c in test_codes]
test_values = top_10_test_perc.values

###############################################################################
# 5) Compare train vs. test distribution (union of top 10 sets), side-by-side
###############################################################################
train_codes_10 = set(top_10_train_perc.index)
test_codes_10  = set(top_10_test_perc.index)
all_codes_10   = list(train_codes_10.union(test_codes_10))

# Create dataframe of percentages
compare_df = pd.DataFrame({
    "Train (%)": [train_perc.get(c, 0) for c in all_codes_10],
    "Test (%)":  [test_perc.get(c, 0) for c in all_codes_10]
}, index=all_codes_10)

# Add human-readable names for labeling
compare_df["lang_name"] = [get_language_name(c) for c in all_codes_10]

# Sort by Train (%) descending, then alphabetically by language name
compare_df.sort_values(by=["Train (%)", "lang_name"], ascending=[False, True], inplace=True)
# Limit to the top 10 languages
compare_df = compare_df.head(10)

print("\nComparison of Train vs. Test distributions (Union of Top 10 sets) in %:")
print(compare_df[["Train (%)", "Test (%)", "lang_name"]])

langs_for_compare = compare_df["lang_name"].tolist()
train_vals_comp   = compare_df["Train (%)"].values
test_vals_comp    = compare_df["Test (%)"].values

N     = len(langs_for_compare)
ind   = np.arange(N)
width = 0.4

plt.figure(figsize=(7,5))
# Side-by-side bars: light blue for Train, dark green for Test
for i in range(N):
    plt.barh(ind[i] - width/2, train_vals_comp[i], height=width, color="lightblue",
             label=None if i>0 else "Train")
    plt.barh(ind[i] + width/2, test_vals_comp[i], height=width, color="darkgreen",
             label=None if i>0 else "Test")

plt.yticks(ind, langs_for_compare)
plt.xlabel("Percentage %")
plt.ylabel("Language")
plt.title("Comparison of Language Distribution (Union of Top 10)")
plt.legend(loc="lower right")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("../Assets/Outputs/EDA/train_vs_test_distribution_top10_percent.png", dpi=300)
plt.close()

###############################################################################
# 6) Train data distribution (Top 20) by frequency
###############################################################################

# Convert ISO codes to "Language (Code)" format
train_df["Language Name"] = train_df["Label"].astype(str).apply(get_language_name)

# Count occurrences and calculate percentages
label_counts = train_df["Language Name"].value_counts(normalize=True) * 100

# Select top 20 languages
top_labels = label_counts.head(20)

# Create a horizontal bar chart
plt.figure(figsize=(14, 8))  
ax = sns.barplot(y=top_labels.index, x=top_labels.values, hue=top_labels.index, palette="viridis", legend=False)

plt.xlabel("Percentage (%)")
plt.ylabel("Language")
plt.title("Top 20 Most Frequent Languages (Percentage)")

# Add percentage labels inside bars
for i, value in enumerate(top_labels.values):
    ax.text(value - 0.05, i, f"{value:.2f}%", va="center", ha="right", fontsize=10, color="white", fontweight="bold")

# Adjust layout to prevent cut-off
plt.tight_layout()
plt.savefig("../Assets/Outputs/EDA/top_20_languages.png")
plt.show()
