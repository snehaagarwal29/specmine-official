
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Load datasets
data_dir = Path("data")
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

specs = pd.read_parquet(data_dir / "spec_files.parquet")
links = pd.read_parquet(data_dir / "spec_links.parquet")
code_refs = pd.read_parquet(
    data_dir / "openspec_code_refs.parquet"
)
prs = pd.read_parquet(data_dir / "pull_requests.parquet")

print("=== RQ2: SPEC-TO-CODE TRACEABILITY ===")

# 2. Explore link types
print("\n=== LINK TYPES ===")
link_types = links["rel"].value_counts()
print(link_types.to_string())

link_types.to_csv(results_dir / "rq2_link_types.csv")

# 3. Explicit code references
code_links = links[links["rel"] == "code"].copy()

# Count distinct specs with code links
linked_spec_ids = set(code_links["spec_url_sha16"])

specs_unique = specs.drop_duplicates(
    subset="file_url_sha16"
).copy()

specs_unique["has_code_link"] = (
    specs_unique["file_url_sha16"].isin(linked_spec_ids)
)

print("\n=== EXPLICIT CODE REFERENCES ===")
print("Total specs:", len(specs_unique))
print("Explicit code links:", len(code_links))
print(
    "Specs with explicit code links:",
    specs_unique["has_code_link"].sum()
)
print(
    "Percentage with code links:",
    f"{specs_unique['has_code_link'].mean():.2%}"
)

# Compare by SDD tool
tool_trace = specs_unique.groupby("spec_tool").agg(
    n_specs=("file_url_sha16", "count"),
    linked_specs=("has_code_link", "sum"),
    link_rate=("has_code_link", "mean")
).sort_values("n_specs", ascending=False)

print("\n=== CODE LINKS BY TOOL ===")
print(tool_trace.head(10).to_string())

tool_trace.to_csv(results_dir / "rq2_tool_traceability.csv")

# 4. Do referenced code files exist?
print("\n=== CODE FILE EXISTENCE ===")

known = code_refs[
    code_refs["exists_at_anchor"].notna()
].copy()

known["exists_at_anchor"] = (
    known["exists_at_anchor"].astype(int)
)

print("Total OpenSpec code references:", len(code_refs))
print("References with known existence:", len(known))

if len(known) > 0:
    exists_rate = known["exists_at_anchor"].mean()

    print(
        "References to existing files:",
        int(known["exists_at_anchor"].sum())
    )
    print(
        "Code-file existence rate:",
        f"{exists_rate:.2%}"
    )

    existence_counts = pd.Series({
        "Exists": int(known["exists_at_anchor"].sum()),
        "Does not exist": int(
            (known["exists_at_anchor"] == 0).sum()
        )
    })

    plt.figure(figsize=(7, 5))
    plt.bar(
        existence_counts.index,
        existence_counts.values
    )
    plt.title("Do Referenced Code Files Exist?")
    plt.ylabel("Number of Code References")
    plt.tight_layout()
    plt.savefig(
        results_dir / "rq2_code_existence.png",
        dpi=200
    )
    plt.close()

# 5. PR co-change analysis
print("\n=== PR CO-CHANGE ===")

# One row per unique PR
unique_prs = prs.drop_duplicates(
    subset=["repo_name", "pr_number"]
).copy()

spec_prs = unique_prs[
    unique_prs["n_spec_files"] > 0
].copy()

cochange_count = int(spec_prs["touches_code"].sum())
cochange_rate = spec_prs["touches_code"].mean()

print("Spec-touching PRs:", len(spec_prs))
print("PRs also modifying code:", cochange_count)
print("Co-change rate:", f"{cochange_rate:.2%}")

# Tool-level results: preserve tool labels,
# but count unique PRs within each tool
tool_prs = prs[prs["n_spec_files"] > 0].drop_duplicates(
    subset=["tool", "repo_name", "pr_number"]
)

pr_stats = tool_prs.groupby("tool").agg(
    n_prs=("pr_number", "count"),
    cochange_rate=("touches_code", "mean")
).sort_values("n_prs", ascending=False)

print("\n=== PR CO-CHANGE BY TOOL ===")
print(pr_stats.head(10).to_string())

pr_stats.to_csv(results_dir / "rq2_pr_cochange.csv")

# 6. Graph 1: Link types
plt.figure(figsize=(9, 5))

link_types.head(10).sort_values().plot(
    kind="barh"
)

plt.title("Types of Specification Links")
plt.xlabel("Number of Links")
plt.tight_layout()
plt.savefig(
    results_dir / "rq2_link_types.png",
    dpi=200
)
plt.close()

# 7. Graph 2: Code links by tool
top_tools = tool_trace[
    tool_trace["n_specs"] >= 100
].head(10).sort_values("link_rate")

plt.figure(figsize=(9, 5))

plt.barh(
    top_tools.index,
    top_tools["link_rate"] * 100
)

plt.title("Specifications with Code Links by SDD Tool")
plt.xlabel("Specifications with Code Links (%)")
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig(
    results_dir / "rq2_tool_traceability.png",
    dpi=200
)
plt.close()

# 8. Graph 3: PR co-change by tool
top_pr_tools = pr_stats[
    pr_stats["n_prs"] >= 20
].head(10).sort_values("cochange_rate")

plt.figure(figsize=(9, 5))

plt.barh(
    top_pr_tools.index,
    top_pr_tools["cochange_rate"] * 100
)

plt.title("Spec-Code Co-change Rate by SDD Tool")
plt.xlabel("PRs Also Changing Code (%)")
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig(
    results_dir / "rq2_pr_cochange.png",
    dpi=200
)
plt.close()

print("\nRQ2 analysis complete!")
print("Results saved to results/")
