import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(data_filepath: str) -> pd.DataFrame:
    """Load the IRIS dataset from the specified filepath"""
    return pd.read_excel(data_filepath)


if __name__ == "__main__":
    # Load the IRIS dataset
    os.chdir("C:\\Users\\ManosIeronymakisProb\\OneDrive - Probability\\Bureaublad")
    filepath = "IRIS.xlsx"
    iris = load_data(filepath)

    # Create scatterplots
    sns.scatterplot(x="sepal_length", y="sepal_width", hue="species", data=iris)
    sns.lmplot(x="sepal_length", y="sepal_width", hue="species", data=iris)
    sns.lmplot(x="sepal_length", y="sepal_width", data=iris)
    ax = sns.scatterplot(x="sepal_length", y="sepal_width", hue="species", data=iris)
    handles, labels = ax.get_legend_handles_labels()
    sizes = iris.groupby("species").size().values.tolist()
    labels = [f"{label} ({size})" for label, size in zip(labels, sizes)]
    ax.legend(handles, labels, title="Species (Count)")

    # Show the plots
    plt.show()