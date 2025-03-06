import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import pycountry
import os
import io

# Ensure the directory exists
os.makedirs("Assets/Outputs/EDA", exist_ok=True)

df = pd.read_csv(r"Assets/Data/train_submission.csv", names=["Usage", "Text", "Label"], low_memory=False)

# Capture the output of df.info()
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()

# Save dataset information to a text file
with open("Assets/Outputs/EDA/dataset_info.txt", "w") as f:
    f.write(info_str)
    f.write("\n")
    f.write("Missing values in each column:\n")
    f.write(str(df.isnull().sum()))
    f.write("\n\n")
    f.write("Number of duplicated rows:\n")
    f.write(str(df.duplicated().sum()))
    f.write("\n\n")
    f.write("Number of unique languages:\n")
    f.write(str(df["Label"].nunique()))

# Print missing values, duplicated rows, and unique languages
print(df.isnull().sum())
print(df.duplicated().sum())
print(df["Label"].nunique())

df_sample = df.sample(n=1000, random_state=42)
df_sample["word_count"] = df_sample["Text"].astype(str).apply(lambda x: len(x.split()))

plt.figure(figsize=(8, 5))
sns.histplot(df_sample["word_count"], bins=30, kde=True)
plt.title("Distribution of Word Counts (Sampled Data)")
plt.savefig("Assets/Outputs/EDA/word_count_distribution.png")
plt.show()

# Function to get language name from ISO 639-3 code
def get_language_name(iso_code):
    try:
        language_name = pycountry.languages.get(alpha_3=iso_code).name
        return f"{language_name} ({iso_code})"
    except AttributeError:
        return iso_code

# Convert ISO codes to "Language (Code)" format
df["Language Name"] = df["Label"].astype(str).apply(get_language_name)

# Count occurrences and calculate percentages
label_counts = df["Language Name"].value_counts(normalize=True) * 100

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
plt.savefig("Assets/Outputs/EDA/top_20_languages.png")
plt.show()