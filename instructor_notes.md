# Instructor Notes

## Intended level

This lab fits a 50 to 75 minute class session, a one week homework assignment, or a short module in a data science or machine learning course.

Prerequisites:

- basic Python,
- pandas data frames,
- train and test splits,
- linear regression,
- basic error metrics.

## Main teaching point

The goal is not to teach a new algorithm. The goal is to teach students that the validation design determines what the performance estimate means.

The random split answers:

> Can the model predict held out rows that come from the same mixture of sites already represented in training?

The group held out split answers:

> Can the model predict observations from a site that was not represented in training?

Students should see that these are different questions.

## Expected outcome

The exact numbers may vary slightly after regeneration, but the pattern should be stable:

- random split performance looks substantially better,
- group held out performance is weaker,
- the difference comes from grouped observations and site level signal.

## Discussion prompts

Ask students:

1. Which result would you report to a project sponsor?
2. What would happen if this were a sensor model used at a new field site?
3. What would happen if this were a microbiome model used for a new cohort?
4. What information was available to the random split that would not be available in deployment?
5. What additional metadata would help design a better validation scheme?

## Grading focus

Strong submissions should:

- correctly identify leakage from site level grouping,
- explain the mismatch between evaluation design and deployment question,
- avoid claiming that one split is universally better,
- connect the validation strategy to a concrete prediction goal,
- include a limitation of the synthetic data.

## Adaptation ideas

For an introductory programming course, provide most of the modeling code and focus on interpretation.

For a machine learning course, ask students to compare multiple models and use grouped cross-validation.

For a scientific computing course, ask students to add visual diagnostics and reproducibility checks.

For a capstone preparation module, ask students to write a one page validation plan for their own project.
