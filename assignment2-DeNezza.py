# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: CS7830 venv
#     language: python
#     name: cs7830
# ---

# %%
import polars as pl
import numpy as np
import matplotlib.pyplot as plot
from pathlib import Path
import seaborn as sns
import os


# %%
def load_dataset() -> pl.DataFrame:
    columns = [
        "ID",
        "Age",
        "Gender",
        "Education",
        "Country",
        "Ethnicity",
        "Nscore",
        "Escore",
        "Oscore",
        "Ascore",
        "Cscore",
        "Impulsive",
        "SS",
        "Alcohol",
        "Amphet",
        "Amyl",
        "Benzos",
        "Caff",
        "Cannabis",
        "Choc",
        "Coke",
        "Crack",
        "Ecstasy",
        "Heroin",
        "Ketamine",
        "Legalh",
        "LSD",
        "Meth",
        "Mushrooms",
        "Nicotine",
        "Semer",
        "VSA",
    ]
    df = pl.read_csv(
        Path("drug_consumption/drug_consumption.data"),
        has_header=False,
        new_columns=columns,
    )
    return df


df = load_dataset()
print(df.head(5))

# %%
# Descriptive Statistics:

os.makedirs(Path("figures"), exist_ok=True)
df.describe().write_csv(Path("figures/descriptive_statistics.csv"))

# %%

os.makedirs(Path("figures/frequencies"), exist_ok=True)


def save_frequencies_to_csv(column: str, df: pl.DataFrame):
    df.select(pl.col(column).value_counts()).unnest(column).write_csv(
        Path(f"figures/frequencies/{column}.csv")
    )


save_frequencies_to_csv("Alcohol", df)
save_frequencies_to_csv("Cannabis", df)
save_frequencies_to_csv("Mushrooms", df)
save_frequencies_to_csv("Nicotine", df)
save_frequencies_to_csv("LSD", df)

# %%
os.makedirs(Path("figures/correlations"), exist_ok=True)
sns.pairplot(df.to_pandas())
plot.savefig("figures/correlations/coorelation_matrix.jpg")


# %%

def send_cl_to_float(cl: str) -> float:
    """
    Classes will be of form "CLX" where X is a number from 0 to 6 inclusive.
    This function sends CL0 -> 0.0 and CL6 -> 1.0
    """
    return float(cl[-1])/6


plot.clf()
heatmap_df = df.with_columns(
    [
        pl.col("Age"),
        pl.col("Gender"),
        pl.col("Education"),
        pl.col("Country"),
        pl.col("Ethnicity"),
        pl.col("Nscore"),
        pl.col("Escore"),
        pl.col("Oscore"),
        pl.col("Ascore"),
        pl.col("Cscore"),
        pl.col("Impulsive"),
        pl.col("SS"),
        pl.col("Alcohol").map_elements(send_cl_to_float, return_dtype=pl.Float64),
        pl.col("Amphet").map_elements(send_cl_to_float, return_dtype=pl.Float64),
        pl.col("Amyl").map_elements(send_cl_to_float, return_dtype=pl.Float64),
        pl.col("Benzos").map_elements(send_cl_to_float, return_dtype=pl.Float64),
        pl.col("Caff").map_elements(send_cl_to_float, return_dtype=pl.Float64),
        pl.col("Cannabis").map_elements(send_cl_to_float, return_dtype=pl.Float64),
        pl.col("Choc").map_elements(send_cl_to_float, return_dtype=pl.Float64),
        pl.col("Coke").map_elements(send_cl_to_float, return_dtype=pl.Float64),
        pl.col("Crack").map_elements(send_cl_to_float, return_dtype=pl.Float64),
        pl.col("Ecstasy").map_elements(send_cl_to_float, return_dtype=pl.Float64),
        pl.col("Heroin").map_elements(send_cl_to_float, return_dtype=pl.Float64),
        pl.col("Ketamine").map_elements(send_cl_to_float, return_dtype=pl.Float64),
        pl.col("Legalh").map_elements(send_cl_to_float, return_dtype=pl.Float64),
        pl.col("LSD").map_elements(send_cl_to_float, return_dtype=pl.Float64),
        pl.col("Meth").map_elements(send_cl_to_float, return_dtype=pl.Float64),
        pl.col("Mushrooms").map_elements(send_cl_to_float, return_dtype=pl.Float64),
        pl.col("Nicotine").map_elements(send_cl_to_float, return_dtype=pl.Float64),
        pl.col("Semer").map_elements(send_cl_to_float, return_dtype=pl.Float64),
        pl.col("VSA").map_elements(send_cl_to_float, return_dtype=pl.Float64)
    ]
)
plot.figure(figsize=(25,25))
sns.heatmap(heatmap_df.to_pandas().corr(), cmap="coolwarm", annot=True, linewidths=0.5)
plot.savefig(Path("figures/correlations/heatmap.jpg"))
# %%
heatmap_df.corr().write_csv(Path("figures/correlations/corr_matrix.csv"))
# %%












