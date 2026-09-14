# Autism Spectrum Disorder (ASD) Prediction

A Streamlit web app that demonstrates ML-based ASD screening prediction, built
directly on top of an existing course notebook. **This is an educational demo,
not a medical diagnosis.**

## Project purpose

Predict whether a person is likely to screen positive for ASD (`Class/ASD` =
1) based on AQ-10 questionnaire answers and background information, using
three tree-based classifiers: Decision Tree, Random Forest, and XGBoost.

## Dataset

- 800 rows, 22 original columns.
- `ID` and `age_desc` are dropped (not predictive / constant).
- Target: `Class/ASD` (0 = No Autism, 1 = Autism).
- Final 19 input features, in this exact order:

  ```
  A1_Score, A2_Score, A3_Score, A4_Score, A5_Score, A6_Score, A7_Score,
  A8_Score, A9_Score, A10_Score, age, gender, ethnicity, jaundice, austim,
  contry_of_res, used_app_before, result, relation
  ```

## Preprocessing (from the notebook — reproduced exactly)

1. `age` cast to `int`.
2. Drop `ID`, `age_desc`.
3. Fix inconsistent country names in `contry_of_res`:
   `Viet Nam → Vietnam`, `AmericanSamoa → United States`, `Hong Kong → China`.
4. Collapse `ethnicity`: `?` / `others` → `Others`.
5. Collapse `relation`: `?`, `Relative`, `Parent`, `Health care professional`
   → `Others`.
6. `LabelEncoder` fit on every categorical (object-dtype) column — `gender`,
   `ethnicity`, `jaundice`, `austim`, `contry_of_res`, `used_app_before`,
   `relation` — and saved to `encoders.pkl`. The app **reuses these exact
   encoders** and never refits them, so category → number mappings can never
   drift between training and inference.
7. Outlier replacement (IQR rule, replaced with the column median) applied to
   `age` and `result` **on the full training dataframe before the split**.
   This step has no saved parameters (no stored quantiles/medians), so it is
   a training-set cleaning step rather than a reusable transform. **The app
   intentionally does not reapply it to a single live user input** — doing so
   would mean inventing thresholds that were never saved anywhere. This is
   the one place the deployment diverges from a literal line-by-line replay
   of the notebook, and it's called out here so it's never a silent change.

## AQ-10 scoring direction (important)

The AQ-10 is not "agree = autism trait" on every item. Per the official
Cambridge Autism Research Centre scoring key (Allison, Auyeung &
Baron-Cohen, 2012) — the same key used to build the original
`A1_Score..A10_Score` columns — **agreeing** scores a point on items
**1, 7, 8, 10**, while **disagreeing** scores a point on items
**2, 3, 4, 5, 6, 9** (these are reverse-worded to reduce response bias).
The app asks each item as Agree/Disagree and applies this per-item
direction internally (`AGREE_SCORES_1` / `DISAGREE_SCORES_1` in `app.py`)
before building the model's input row — it does not simply map every
"Agree" to 1.

Because the direction differs per item, and the tree-based models each
learn their own feature importances from the training data, different
answer patterns are weighted differently by the model — there's no fixed
"more Yes = more likely ASD" rule. The app has a "See which factors each
model relies on most" panel that shows each trained model's actual
learned `feature_importances_`.

## About the `result` feature

`result` is one of the 19 trained features, so the model needs a value
for it at every prediction — it can't just be dropped. In the textbook
AQ-10 instrument, `result` is defined as the plain sum of the ten item
scores (an integer, 0–10). But in this dataset, the `result` values shown
in the notebook preview include non-integer and even negative numbers
(e.g. `-4.777286`), which is impossible for a sum of ten 0/1 values — so
`result` in this dataset is **not** an exact function of A1–A10.

The app pre-fills `result` with the standard AQ-10 sum computed from the
user's answers (`suggested_result` in `app.py`) as a reasonable starting
value, but leaves it editable, with a note that this is only an estimate.
Run `check_result_feature.py` against your real `train.csv` to measure
how closely `result` actually tracks sum(A1..A10) in your data — if the
correlation turns out to be ~1.0, the pre-fill can be made non-editable;
if it's weaker, the manual override matters and should stay.

## Class balancing

`SMOTE(random_state=42)` is applied **only to the training split**, after
`train_test_split(X, y, test_size=0.2, random_state=42)`. SMOTE is never
applied to the test set or to a user's input at prediction time — it's a
training-time-only technique.

## Models

Three tree-based classifiers, all tuned and all shipped in the app (not just
the best one):

- Decision Tree (`sklearn.tree.DecisionTreeClassifier`)
- Random Forest (`sklearn.ensemble.RandomForestClassifier`)
- XGBoost (`xgboost.XGBClassifier`)

## Hyperparameter tuning

`RandomizedSearchCV` for each model with:

- `cv=5`
- `n_iter=20`
- `scoring='f1'`
- `random_state=42`

using the same parameter grids as the notebook (see `train_models.py`).

## Best model

**Random Forest** had the best cross-validation score (~0.93 F1) in the
notebook. Decision Tree and XGBoost are kept in the deployment regardless, so
all three are selectable in the app, plus a "Compare All Models" option.

## Project structure

```
Autism_Prediction_using_Machine_Learning/
│
├── app.py                     # Streamlit app
├── train_models.py            # Reproduces preprocessing + SMOTE + RandomizedSearchCV, saves artifacts
├── check_result_feature.py    # One-off diagnostic: does result = sum(A1..A10) in your data?
├── requirements.txt
├── README.md
│
├── models/
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
└── encoders.pkl
```

`models/*.pkl` and `encoders.pkl` are **not** included in this repo — you
generate them once by running `train_models.py` against your own `train.csv`.

## How to run locally

1. Put your original `train.csv` (800 rows, 22 columns) in the project
   folder, next to `train_models.py`.
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Generate the encoders and tuned models (only needs to be run once, or
   whenever you retrain):

   ```bash
   python train_models.py
   ```

   This creates `encoders.pkl` and the three files inside `models/`.

4. Launch the app:

   ```bash
   streamlit run app.py
   ```

5. Open the URL Streamlit prints (usually `http://localhost:8501`).

## How to deploy on Streamlit Community Cloud

1. Push this whole folder to a GitHub repository, **including the generated
   `encoders.pkl` and `models/*.pkl` files** (Community Cloud only sees what's
   in the repo — it doesn't run `train_models.py` for you).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **New app**, pick your repository, branch, and set the main file
   path to `app.py`.
4. Streamlit Cloud reads `requirements.txt` automatically and installs the
   pinned dependencies.
5. Click **Deploy**. The first build takes a few minutes; after that you get
   a public URL you can share.
6. If you retrain locally later, just commit the new `.pkl` files and push —
   Community Cloud redeploys automatically.

## Notes for discussing this project

- Encoders and models are trained once, offline, and only ever *loaded* by
  the app — no retraining or refitting happens on app startup or on each
  prediction.
- The dropdown options for categorical fields come directly from each saved
  encoder's `classes_` attribute, so the app can never offer a category value
  the model wasn't trained on.
- Feature order is hardcoded identically in both `train_models.py` and
  `app.py` to guarantee the model always sees columns in the order it was
  trained on.
