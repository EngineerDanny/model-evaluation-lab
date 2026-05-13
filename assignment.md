# Assignment: When Good Test Scores Lie

## Context

You are given a synthetic dataset inspired by environmental sensor readings and sparse biological count measurements. Each row is an observation from one of several collection sites. Your task is to predict a continuous response called `risk_score`.

The dataset is intentionally designed to teach a common model evaluation problem: observations from the same site can be more similar to each other than observations from different sites. If a random train and test split puts rows from every site in both sets, the model may partly learn site specific patterns rather than a relationship that generalizes to a new site.

## Your Task

Build and evaluate a regression model in two ways:

1. Random split: randomly assign observations to training and test sets.
2. Group held out split: train on some sites and test on a site the model has not seen.

Then compare the results and explain which evaluation better answers this question:

> How well will this model perform on data from a new site?

## Deliverables

Submit a short report with:

1. A table of metrics for both validation designs.
2. A paragraph explaining why the two estimates differ.
3. A recommendation for which validation design should be used before claiming the model generalizes.
4. One limitation of this synthetic experiment.

## Suggested Metrics

- mean absolute error
- root mean squared error
- coefficient of determination, or R squared

## Reflection Questions

1. What information leaks into the random split?
2. Why does holding out a site create a harder test?
3. If this were a real scientific or engineering project, what claim would be unsafe to make from the random split alone?
4. How would your validation design change if the model were intended for the same sites next semester rather than a new site?

## Extension

Replace linear regression with a random forest, gradient boosting model, or neural network. Does a more flexible model make the leakage problem smaller or larger? Explain your answer.
