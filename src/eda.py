import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_ecdf(df, column, log_scale=True, show_quartiles=True):
    x = df[column].dropna()

    plt.figure(figsize=(6, 4))
    sns.ecdfplot(x=x, color="black")

    if show_quartiles:
        q1, q2, q3 = x.quantile([0.25, 0.5, 0.75])

        plt.axvline(q1, color="tab:blue",   linestyle=":", alpha=0.7, label="Q1")
        plt.axvline(q2, color="tab:orange", linestyle=":", alpha=0.7, label="Median")
        plt.axvline(q3, color="tab:green",  linestyle=":", alpha=0.7, label="Q3")
        plt.legend()

    if log_scale:
        plt.xscale("log")
        plt.xlabel(f"{column} (log scale)")
    else:
        plt.xlabel(column)

    plt.ylabel("Cumulative proportion")
    plt.title(f"ECDF of {column}")
    sns.despine()
    plt.show()

def delinquency_summary(df, column, target='SeriousDlqin2yrs'):
    return (
    df.groupby(column)[target]
      .agg(default_rate='mean', count='size')
      .sort_index()
    )

def plot_default_rate_by_count(
    df,
    column,
    bins,
    labels,
    target = 'SeriousDlqin2yrs',
    title=None,
    show_counts=True
    ):

    temp = df.copy()
    temp["_bin"] = pd.cut(
        df[column],
        bins = bins,
        labels = labels
    )

    summary = (
        temp.groupby('_bin')[target]
            .agg(default_rate='mean', count='size')
            .reset_index()
            .rename(columns={'_bin':'bin'})
    )

    ax = sns.barplot(data = summary, x='bin', y='default_rate', errorbar=None)
    ax.set_xlabel('')
    ax.set_ylabel('Default rate')
    ax.set_title(title or f"Default rate by binned {column}")
    sns.despine()

    if show_counts:
        for p, n in zip(ax.patches, summary["count"]):
            ax.annotate(
                f"n={int(n)}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points"
            )

    plt.tight_layout()
    plt.show()