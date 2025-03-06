import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

###############################################################################
# 1) Map language codes to real names (including newly requested codes)
###############################################################################
language_map = {
    "hau": "Hausa",
    "nob": "Norwegian Bokmål",
    "wln": "Walloon",
    "quh": "Quechua",
    "scn": "Sicilian",
    "uzb": "Uzbek",
    "roh": "Romansh",
    "ayr": "Aymara",
    "rmy": "Romani",
    "ven": "Venda",
    "epo": "Esperanto",
    "tgk": "Tajik",
    "gom": "Goan Konkani",
    "kat": "Georgian",
    "kaa": "Karakalpak",
    "mon": "Mongolian",
    "hin": "Hindi",
    "tat": "Tatar",
    "guj": "Gujarati",
    "crh": "Crimean Tatar",
    "som": "Somali",
    "uig": "Uyghur",
    "kur": "Kurdish",
}

def get_language_name(code):
    """Return a mapped language name if available; otherwise, return the code itself."""
    return language_map.get(code, code)

###############################################################################
# 2) Load the data
###############################################################################
train_df = pd.read_csv("../Assets/Data/train_submission.csv")  # Must have at least 'Label'
pred_df  = pd.read_csv("../Assets/Outputs/Submission/submission_file.csv")   # Must have at least 'ID', 'Label'

train_total = len(train_df)
test_total  = len(pred_df)

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
plt.savefig("train_vs_test_distribution_top10_percent.png", dpi=300)
plt.close()

print("\nAnalysis completed. All plots have been saved as PNG files.")
