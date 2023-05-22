import os
from cycler import cycler
from pathlib import Path

import hydra
from omegaconf import DictConfig

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


cd = {
    "red": "#E24A33",
    "blue": "#348ABD",
    "purple": "#988ED5",
    "gray": "#777777",
    "green": "#8EBA42",
}
plt.style.use("ggplot")
colors = [cd["red"], cd["blue"], cd["purple"], cd["green"], cd["gray"]]
plt.rcParams["axes.prop_cycle"] = cycler(color=colors)
plt.rcParams["font.size"] = 12


def visualize(experiment: str, compared_estimators: str):
    df = pd.read_csv(f"estimation_{experiment}_{compared_estimators}.csv")

    targets = df["target_value"].unique()
    estimators = ["IPS", "IIPS", "RIPS", "AIPS", "AIPS (true)"]

    if experiment in ["data_size", "slate_size"]:
        targets_str = [str(int(targets[i])) for i in range(len(targets))]
    else:
        targets_str = [str(float(targets[i])) for i in range(len(targets))]

    if experiment == "data_size":
        xlabel = "data size"
    elif experiment == "slate_size":
        xlabel = "slate size"
    else:
        xlabel = "user behavior interpolation parameter"

    n_samples = len(df)
    n_samples_per_target = n_samples // len(targets)

    mse = np.zeros((len(estimators), len(targets), n_samples_per_target))
    squared_bias = np.zeros((len(estimators), len(targets)))
    variance = np.zeros((len(estimators), len(targets)))

    mse_df = pd.DataFrame()
    squared_bias_df = pd.DataFrame()
    variance_df = pd.DataFrame()

    for i, estimator in enumerate(estimators):
        for j, target in enumerate(targets):
            ground_truth = df[df["target_value"] == target]["ground_truth"].mean()
            estimate = df[df["target_value"] == target][estimator].to_numpy()
            mse[i, j] = ((estimate - ground_truth) / ground_truth) ** 2
            squared_bias[i, j] = ((estimate.mean() - ground_truth) / ground_truth) ** 2
            variance[i, j] = estimate.var() / (ground_truth**2)

        mse_df["mse"] = mse.flatten()
        mse_df["estimator_name"] = [
            estimators[i // n_samples] for i in range(n_samples * len(estimators))
        ]
        mse_df["target_value"] = [
            (i // n_samples_per_target) % len(targets)
            for i in range(n_samples * len(estimators))
        ]

        squared_bias_df["squared_bias"] = squared_bias.flatten()
        squared_bias_df["estimator_name"] = [
            estimators[i // len(targets)] for i in range(len(targets) * len(estimators))
        ]
        squared_bias_df["target_value"] = [
            i % len(targets) for i in range(len(targets) * len(estimators))
        ]

        variance_df["variance"] = variance.flatten()
        variance_df["estimator_name"] = [
            estimators[i // len(targets)] for i in range(len(targets) * len(estimators))
        ]
        variance_df["target_value"] = [
            i % len(targets) for i in range(len(targets) * len(estimators))
        ]

    os.chdir("../../../")

    # (relative) mse
    plt.figure()
    sns.lineplot(
        data=mse_df, x="target_value", y="mse", hue="estimator_name", marker="o"
    )
    plt.xticks(np.arange(len(targets)), targets_str)
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.yscale("log")
    plt.ylabel("mse (relative)")
    plt.legend()
    plt.savefig(f"figs/{experiment}_mse.png", dpi=300, bbox_inches="tight")

    # (relative) squared bias
    plt.figure()
    sns.lineplot(
        data=squared_bias_df,
        x="target_value",
        y="squared_bias",
        hue="estimator_name",
        marker="o",
    )
    plt.xticks(np.arange(len(targets)), targets_str)
    plt.xlabel(xlabel)
    plt.yscale("log")
    plt.ylim(8.0e-6, 2.0)
    plt.ylabel("squared bias (relative)")
    plt.legend(loc="upper right").remove()
    plt.savefig(f"figs/{experiment}_squared_bias.png", dpi=300, bbox_inches="tight")

    # (relative) variance
    plt.figure()
    sns.lineplot(
        data=variance_df,
        x="target_value",
        y="variance",
        hue="estimator_name",
        marker="o",
    )
    plt.xticks(np.arange(len(targets)), targets_str)
    plt.xlabel(xlabel)
    plt.yscale("log")
    plt.ylim(8.0e-6, 2.0)
    plt.ylabel("variance (relative)")
    plt.legend(loc="upper right").remove()
    plt.savefig(f"figs/{experiment}_variance.png", dpi=300, bbox_inches="tight")


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    print(f"The current working directory is {Path().cwd()}")
    print(f"The original working directory is {hydra.utils.get_original_cwd()}")
    print()
    # configurations
    experiment = cfg.setting.experiment
    compared_estimators = cfg.setting.compared_estimators
    visualize(experiment, compared_estimators)


if __name__ == "__main__":
    main()
