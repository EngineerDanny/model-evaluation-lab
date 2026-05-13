# Model Evaluation Lab

This is a compact undergraduate teaching module on model evaluation and data leakage.

The lab is designed for an introductory data science, machine learning, scientific computing, or applied computer science course. Students compare a random train and test split with a group held out split, then explain why the random split can produce an overly optimistic estimate when observations from the same site or source appear in both training and test data.

## Why this project exists

Many students learn to call `train_test_split` before they learn to ask what kind of independence the test set represents. This lab gives them a concrete failure case:

1. A model looks strong under a random split.
2. The same model performs worse when tested on a held out site.
3. Students diagnose the difference as leakage from grouped observations.
4. Students propose a validation design that matches the real question.

## Files

- `index.html`: public project page for GitHub Pages.
- `assignment.md`: student facing assignment handout.
- `instructor_notes.md`: teaching notes and expected discussion points.
- `data/synthetic_sensor_microbiome.csv`: synthetic data for the lab.
- `notebooks/model_evaluation_lab.ipynb`: starter notebook using pandas and NumPy.
- `tools/generate_data_and_notebook.py`: reproducible build script.

## Learning goals

After completing the lab, students should be able to:

- explain why a random split can overestimate performance,
- choose a validation design that matches the deployment question,
- compute and compare simple regression metrics,
- connect cross-validation choices to scientific and applied computing claims,
- write a short technical interpretation of model results.

## Local use

Open `index.html` directly in a browser, or serve this folder locally:

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000`.

To regenerate the synthetic dataset and notebook:

```bash
python3 tools/generate_data_and_notebook.py
```

The data are synthetic and are not derived from St. Mary's College of Maryland, NAWCAD, or any real student, biological, or defense dataset.
