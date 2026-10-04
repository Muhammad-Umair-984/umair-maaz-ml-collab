# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Forest CoverType Exploratory Data Analysis
#
# This notebook checks the structure, quality, class balance, and principal feature patterns
# in the Forest CoverType dataset. The target has seven forest cover classes, making this a
# multiclass classification problem. Reusable loading and validation logic lives in
# `forest_cover.eda` so the analysis is reproducible outside the notebook.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from forest_cover.eda import (
    COVER_TYPE_LABELS,
    EXPECTED_COLUMN_COUNT,
    EXPECTED_ROW_COUNT,
    QUANTITATIVE_COLUMNS,
    SOIL_TYPE_COLUMNS,
    TARGET_COLUMN,
    WILDERNESS_AREA_COLUMNS,
    dataset_summary,
    load_forest_cover,
)

pd.set_option("display.max_columns", 12)

# %% [markdown]
# ## Load and validate the dataset

# %%
forest_cover = load_forest_cover()
expected_shape = (EXPECTED_ROW_COUNT, EXPECTED_COLUMN_COUNT)

print(f"Loaded shape: {forest_cover.shape}")
if forest_cover.shape != expected_shape:
    raise ValueError(f"Expected the complete DVC dataset shape {expected_shape}.")

# %% [markdown]
# ## Basic inspection

# %%
forest_cover.head()

# %%
forest_cover.info()

# %%
forest_cover.loc[:, QUANTITATIVE_COLUMNS].describe().T

# %% [markdown]
# ## Data quality

# %%
quality_summary = dataset_summary(forest_cover)
pd.Series(quality_summary, name="value").to_frame()

# %%
missing_by_column = forest_cover.isna().sum()
missing_by_column[missing_by_column > 0].sort_values(ascending=False)

# %% [markdown]
# ## Target distribution

# %%
target_counts = forest_cover[TARGET_COLUMN].value_counts().sort_index()
target_distribution = pd.DataFrame(
    {
        "class_name": target_counts.index.map(COVER_TYPE_LABELS),
        "count": target_counts,
        "proportion": target_counts / len(forest_cover),
    }
)
target_distribution

# %%
figure, axis = plt.subplots(figsize=(9, 4.5))
axis.bar(target_distribution["class_name"], target_distribution["count"], color="#4472C4")
axis.set(title="Forest CoverType class distribution", xlabel="Cover type", ylabel="Rows")
axis.tick_params(axis="x", rotation=35)
figure.tight_layout()
plt.show()

# %% [markdown]
# Classes 1 and 2 make up most observations, while classes 4 and 5 are much smaller.
# Evaluation and later train/test splitting should therefore preserve class proportions and use
# metrics that do not hide minority-class performance.

# %% [markdown]
# ## Quantitative feature exploration

# %%
forest_cover.loc[:, QUANTITATIVE_COLUMNS].hist(bins=30, figsize=(14, 10), color="#5B9BD5")
plt.suptitle("Quantitative feature distributions", y=1.01)
plt.tight_layout()
plt.show()

# %%
quantitative_correlations = forest_cover.loc[:, QUANTITATIVE_COLUMNS].corr()
figure, axis = plt.subplots(figsize=(10, 8))
image = axis.imshow(quantitative_correlations, cmap="coolwarm", vmin=-1, vmax=1)
axis.set_xticks(range(len(QUANTITATIVE_COLUMNS)), QUANTITATIVE_COLUMNS, rotation=90)
axis.set_yticks(range(len(QUANTITATIVE_COLUMNS)), QUANTITATIVE_COLUMNS)
axis.set_title("Correlation among quantitative features")
figure.colorbar(image, ax=axis, label="Pearson correlation")
figure.tight_layout()
plt.show()

# %%
selected_features = [
    "Elevation",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
]
class_medians = forest_cover.groupby(TARGET_COLUMN)[selected_features].median()
class_medians.insert(0, "class_name", class_medians.index.map(COVER_TYPE_LABELS))
class_medians

# %% [markdown]
# ## Binary indicator checks

# %%
wilderness_is_binary = forest_cover.loc[:, WILDERNESS_AREA_COLUMNS].isin((0, 1)).all().all()
soil_is_binary = forest_cover.loc[:, SOIL_TYPE_COLUMNS].isin((0, 1)).all().all()

print(f"All wilderness indicators are binary: {wilderness_is_binary}")
print(f"All soil indicators are binary: {soil_is_binary}")

# %%
wilderness_sum_counts = (
    forest_cover.loc[:, WILDERNESS_AREA_COLUMNS].sum(axis=1).value_counts().sort_index()
)
soil_sum_counts = forest_cover.loc[:, SOIL_TYPE_COLUMNS].sum(axis=1).value_counts().sort_index()

pd.DataFrame(
    {
        "wilderness_indicator_sum": wilderness_sum_counts,
        "soil_indicator_sum": soil_sum_counts,
    }
).fillna(0).astype(int)

# %% [markdown]
# ## Key findings
#
# - The restored dataset matches the documented 581,012-row, 55-column schema and contains all
#   seven target classes.
# - The data contains no missing values. Duplicate-row counts are reported above so later phases
#   can decide explicitly whether deduplication belongs in preprocessing.
# - The target is imbalanced: Lodgepole Pine and Spruce/Fir dominate, while Cottonwood/Willow and
#   Aspen are minority classes. Later evaluation should include per-class metrics.
# - Wilderness-area and soil-type fields behave as binary one-hot indicators, with one active
#   value in each group per row.
# - Elevation and distance variables span very different scales and vary by cover class. Later
#   preprocessing should be fitted only on training data, and scaling should be chosen according
#   to the model family.
