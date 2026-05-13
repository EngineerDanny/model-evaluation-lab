from __future__ import annotations

import csv
import json
import math
import random
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic_sensor_microbiome.csv"
NOTEBOOK_PATH = ROOT / "notebooks" / "model_evaluation_lab.ipynb"


def make_data(seed: int = 20260513) -> list[dict[str, float | str | int]]:
    rng = random.Random(seed)
    rows: list[dict[str, float | str | int]] = []
    sites = [
        ("estuary", 0.10, 0.75),
        ("forest", -0.30, 0.20),
        ("marsh", 0.35, 0.55),
        ("harbor", 0.65, 0.85),
        ("campus", -0.05, 0.35),
        ("airfield", 0.50, 0.70),
    ]

    for site, site_shift, humidity_base in sites:
        for day in range(1, 61):
            temp = rng.gauss(23 + 4 * site_shift, 3.2)
            humidity = min(0.98, max(0.05, rng.gauss(humidity_base, 0.08)))
            vibration = max(0.0, rng.gauss(0.8 + 0.9 * site_shift, 0.22))
            sparse_count_a = max(0, int(rng.expovariate(1 / (3.5 + 2.0 * humidity))))
            sparse_count_b = max(0, int(rng.expovariate(1 / (2.0 + vibration))))
            background = rng.gauss(site_shift * 4.5, 0.9)
            noise = rng.gauss(0, 1.5)

            risk_score = (
                0.28 * temp
                + 8.5 * humidity
                + 2.8 * vibration
                + 0.55 * math.log1p(sparse_count_a)
                - 0.35 * math.log1p(sparse_count_b)
                + background
                + noise
            )

            rows.append(
                {
                    "site": site,
                    "day": day,
                    "temperature_c": round(temp, 3),
                    "humidity": round(humidity, 3),
                    "vibration_index": round(vibration, 3),
                    "sparse_count_a": sparse_count_a,
                    "sparse_count_b": sparse_count_b,
                    "risk_score": round(risk_score, 3),
                }
            )
    return rows


def write_csv(rows: list[dict[str, float | str | int]]) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def code_cell(source: str) -> dict:
    cleaned = textwrap.dedent(source).strip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in cleaned.splitlines()],
    }


def markdown_cell(source: str) -> dict:
    cleaned = textwrap.dedent(source).strip()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in cleaned.splitlines()],
    }


def write_notebook() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    cells = [
        markdown_cell(
            """
            # Model Evaluation Lab

            This notebook shows why validation design matters. You will compare a random split with a group held out split using synthetic grouped data.
            """
        ),
        code_cell(
            """
            import pandas as pd
            import numpy as np

            DATA_PATH = "../data/synthetic_sensor_microbiome.csv"
            df = pd.read_csv(DATA_PATH)
            df.head()
            """
        ),
        markdown_cell(
            """
            ## Build one model

            We include `site` as a categorical feature to make the leakage problem visible. In the random split, each site appears in both training and test data. In the group held out split, at least one site is unseen during training.
            """
        ),
        code_cell(
            """
            target = "risk_score"
            features = [
                "site",
                "temperature_c",
                "humidity",
                "vibration_index",
                "sparse_count_a",
                "sparse_count_b",
            ]

            X_raw = df[features]
            y = df[target]
            groups = df["site"]

            def make_design_matrix(frame, known_sites, numeric_mean, numeric_std):
                numeric_columns = [c for c in features if c != "site"]
                numeric = frame[numeric_columns].astype(float)
                numeric = (numeric - numeric_mean) / numeric_std
                site_columns = {
                    f"site_{site}": (frame["site"] == site).astype(float)
                    for site in known_sites
                }
                site_df = pd.DataFrame(site_columns, index=frame.index)
                design = pd.concat([numeric, site_df], axis=1)
                design.insert(0, "intercept", 1.0)
                return design

            def fit_ridge(X_train, y_train, alpha=1.0):
                Xmat = X_train.to_numpy(float)
                yvec = y_train.to_numpy(float)
                penalty = np.eye(Xmat.shape[1]) * alpha
                penalty[0, 0] = 0.0
                return np.linalg.solve(Xmat.T @ Xmat + penalty, Xmat.T @ yvec)

            def predict(X_test, beta):
                return X_test.to_numpy(float) @ beta
            """
        ),
        code_cell(
            """
            def metrics(y_true, y_pred):
                error = y_true.to_numpy(float) - y_pred
                mae = np.mean(np.abs(error))
                rmse = np.sqrt(np.mean(error ** 2))
                ss_res = np.sum(error ** 2)
                ss_tot = np.sum((y_true - y_true.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot
                return mae, rmse, r2

            def evaluate(split_name, train_idx, test_idx):
                train = df.iloc[train_idx]
                test = df.iloc[test_idx]
                known_sites = sorted(train["site"].unique())
                numeric_columns = [c for c in features if c != "site"]
                numeric_mean = train[numeric_columns].astype(float).mean()
                numeric_std = train[numeric_columns].astype(float).std(ddof=0).replace(0, 1)
                X_train = make_design_matrix(train[features], known_sites, numeric_mean, numeric_std)
                X_test = make_design_matrix(test[features], known_sites, numeric_mean, numeric_std)
                beta = fit_ridge(X_train, train[target])
                pred = predict(X_test, beta)
                mae, rmse, r2 = metrics(test[target], pred)
                return {
                    "split": split_name,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                }
            """
        ),
        markdown_cell(
            """
            ## Random split

            This split is easy to write, but it may not match the deployment question.
            """
        ),
        code_cell(
            """
            rng = np.random.default_rng(42)
            all_idx = np.arange(len(df))
            rng.shuffle(all_idx)
            test_size = int(0.25 * len(df))
            test_idx = all_idx[:test_size]
            train_idx = all_idx[test_size:]

            random_result = evaluate("random rows", train_idx, test_idx)
            random_result
            """
        ),
        markdown_cell(
            """
            ## Group held out split

            This split asks whether the model can generalize to sites that were not represented in training.
            """
        ),
        code_cell(
            """
            held_out_sites = ["estuary", "harbor"]
            test_idx = df.index[df["site"].isin(held_out_sites)].to_numpy()
            train_idx = df.index[~df["site"].isin(held_out_sites)].to_numpy()

            group_result = evaluate("held out sites", train_idx, test_idx)

            print("Held out sites:", sorted(df.iloc[test_idx]["site"].unique()))
            group_result
            """
        ),
        code_cell(
            """
            results = pd.DataFrame([random_result, group_result])
            results.round(3)
            """
        ),
        markdown_cell(
            """
            ## Reflection

            Write a short explanation:

            1. Which split gives the stronger score?
            2. Why is that score not necessarily the right estimate for a new site?
            3. Which split would you report if the model will be used on future data from an unseen site?
            4. What limitation does this synthetic example have?
            """
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2) + "\n")


def main() -> None:
    rows = make_data()
    write_csv(rows)
    write_notebook()
    print(f"Wrote {DATA_PATH.relative_to(ROOT)} with {len(rows)} rows")
    print(f"Wrote {NOTEBOOK_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
