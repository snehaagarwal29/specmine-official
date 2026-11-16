
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Load SpecMine data
data_dir = Path("data")
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

specs = pd.read_parquet(data_dir / "spec_files.parquet")
features = pd.read_parquet(
    data_dir / "spec_content_features.parquet"
)

# Join metadata with structural features
df = specs.merge(
    features,
    on="file_url_sha16",
    how="inner",
    validate="one_to_one"
)

print("=== RQ1: SPECIFICATION STRUCTURE ===")
print("Total specifications:", len(df))
print("Unique repositories:", df["repo_id"].nunique())
print("SDD tools:", df["spec_tool"].nunique())


# 2. Specification length
valid = df[df["n_words"].notna()].copy()
nonempty = valid[valid["n_words"] > 0].copy()

print("\n=== SPECIFICATION LENGTH ===")
print("Nonempty specs:", len(nonempty))
print("Mean word count:", round(nonempty["n_words"].mean(), 2))
print("Median word count:", nonempty["n_words"].median())
print("Minimum words:", nonempty["n_words"].min())
print("Maximum words:", nonempty["n_words"].max())


# 3. Document structure
print("\n=== DOCUMENT STRUCTURE ===")
print("Mean headings:", round(nonempty["n_headings"].mean(), 2))
print("Median headings:", nonempty["n_headings"].median())
print("Mean list items:", round(nonempty["n_list_items"].mean(), 2))
print("Mean code blocks:", round(nonempty["n_code_fences"].mean(), 2))

# 4. Specification features
print("\n=== STRUCTURAL FEATURES ===")

columns = [
    "has_requirements",
    "has_acceptance_criteria",
    "has_user_story",
    "has_gherkin",
    "has_success_criteria",
    "has_edge_cases",
    "has_unfilled_placeholder"
]

for col in columns:
    percentage = nonempty[col].fillna(False).astype(bool).mean() * 100
    print(f"{col}: {percentage:.2f}%")


# 5. Compare SDD tools
tool_stats = nonempty.groupby("spec_tool").agg(
    specifications=("file_url_sha16", "count"),
    median_words=("n_words", "median"),
    mean_words=("n_words", "mean"),
    median_headings=("n_headings", "median"),
    requirements_rate=("has_requirements", "mean"),
    acceptance_rate=("has_acceptance_criteria", "mean")
).sort_values("specifications", ascending=False)

print("\n=== COMPARISON BY SDD TOOL ===")
print(tool_stats.head(10).round(2).to_string())

tool_stats.to_csv(results_dir / "rq1_tool_comparison.csv")

# 6. Graph 1: Word count distribution
plt.figure(figsize=(9, 5))

upper = nonempty["n_words"].quantile(0.95)
subset = nonempty[nonempty["n_words"] <= upper]

plt.hist(subset["n_words"], bins=40, edgecolor="black")
plt.title("Distribution of Specification Length")
plt.xlabel("Number of Words")
plt.ylabel("Number of Specifications")
plt.tight_layout()

plt.savefig(results_dir / "rq1_word_distribution.png", dpi=200)
plt.close()


# 7. Graph 2: Length by SDD tool
top_tools = tool_stats.head(8).sort_values("median_words")

plt.figure(figsize=(9, 5))
plt.barh(top_tools.index, top_tools["median_words"])
plt.title("Median Specification Length by SDD Tool")
plt.xlabel("Median Number of Words")
plt.tight_layout()

plt.savefig(results_dir / "rq1_tool_comparison.png", dpi=200)
plt.close()


# 8. Graph 3: Structural features
feature_rates = {
    col.replace("has_", "").replace("_", " ").title():
    nonempty[col].fillna(False).astype(bool).mean() * 100
    for col in columns
}

plt.figure(figsize=(10, 5))
plt.barh(
    list(feature_rates.keys()),
    list(feature_rates.values())
)
plt.title("Prevalence of Specification Features")
plt.xlabel("Percentage of Specifications (%)")
plt.xlim(0, 100)
plt.tight_layout()

plt.savefig(results_dir / "rq1_features.png", dpi=200)
plt.close()

print("\nAnalysis complete! Results saved to results/")
