# Credit Default XGBoost Model

This folder contains a complete, reproducible default-prediction project built from `../credit.csv`.

## What This Builds

- An XGBoost binary classifier for `credit_card_default`.
- Bayesian hyperparameter optimization that maximizes validation ROC-AUC.
- A stratified 70% training / 30% validation split.
- A compact feature set with fewer than 20 variables.
- Feature selection from explainable credit-behavior and financial-capacity variables only.
- A correlation screen that rejects variables correlated above the configured limit.
- Validation metrics, threshold analysis, feature-selection reports, and partial dependence plots.
- A Streamlit app for manual scoring, default probability, final prediction, and top 3 reason codes.
- Docker and shell entrypoints that can be hosted on AWS.

## Files

- `train_model.py`: trains the XGBoost model, selects features, writes metrics, and generates PDP plots.
- `app.py`: Streamlit scoring interface.
- `requirements.txt`: Python dependencies.
- `run_app.sh`: executable launcher for local or server use.
- `Dockerfile`: container definition for AWS App Runner, ECS, or EC2.
- `.streamlit/config.toml`: Streamlit server settings.
- `artifacts/default_xgboost_artifacts.joblib`: generated model package after training.
- `artifacts/metrics.json`: generated validation metrics.
- `artifacts/selected_features.json`: generated selected variables and labels.
- `reports/model_report.md`: generated model report.
- `reports/model_leaderboard.csv`: generated XGBoost parameter search results.
- `reports/threshold_analysis.csv`: generated threshold performance table.
- `reports/validation_predictions.csv`: generated validation predictions.
- `reports/validation_score_distribution.png`: generated histogram of validation predicted probabilities.
- `reports/validation_score_quantiles.csv`: generated validation score quantiles by actual outcome.
- `reports/validation_score_deciles.csv`: generated validation score decile/lift table.
- `reports/validation_decile_lift.png`: generated validation decile lift plot.
- `reports/validation_decile_capture.png`: generated cumulative default-capture chart by risk decile.
- `reports/partial_dependence_summary.csv`: generated PDP summary.
- `reports/pdp/*.png`: generated partial dependence plots.

## Setup

From the workspace root:

```bash
cd "/Users/nalinaggarwal/Desktop/Visual Studio"
pip install -r credit_default_xgboost/requirements.txt
```

## Train The Model

```bash
cd "/Users/nalinaggarwal/Desktop/Visual Studio/credit_default_xgboost"
python train_model.py --data ../credit.csv
```

The training script prints the search method, number of hyperparameter evaluations, selected features, ROC-AUC, Gini, threshold, artifact path, and report path.

By default, training uses Bayesian optimization:

```bash
python train_model.py --data ../credit.csv --search-method bayesian
```

You can run a longer overnight search by increasing the Bayesian search budget:

```bash
python train_model.py --data ../credit.csv --bayes-init 30 --bayes-iterations 250 --bayes-candidates 1500
```

For comparison with the older fixed search:

```bash
python train_model.py --data ../credit.csv --search-method grid
```

## Run The Streamlit App

```bash
cd "/Users/nalinaggarwal/Desktop/Visual Studio/credit_default_xgboost"
./run_app.sh
```

Then open:

```text
http://localhost:8501
```

## Model Measurement Method

Primary measurement and optimization objective: **ROC-AUC**, translated to **Gini = 2 x AUC - 1**.

Why this is the primary method:

- Credit default scoring is a ranking problem: the model should assign higher risk scores to borrowers who default.
- ROC-AUC is threshold-independent and stable for this dataset's default rate.
- Gini is the standard credit-risk interpretation of ROC-AUC.

Supporting measures are also reported:

- Precision-recall AUC.
- KS statistic.
- Log loss.
- F1, precision, and recall at the selected threshold.
- Confusion matrix.

## Feature Policy

The model deliberately avoids demographic and hard-to-explain fields such as `sex`, `education`, `marriage`, `region`, `state_code`, and `zip_income_percentile`.

Feature candidates are limited to intuitive credit-risk themes:

- Payment behavior.
- Credit quality.
- Balance pressure.
- Income and repayment capacity.
- Credit seeking.
- Liquidity pressure.
- Account behavior.

The selector ranks explainable candidates and keeps at most 18 variables, rejecting variables with pairwise absolute correlation above the configured default limit of `0.60`.

## Reason Codes

The app uses XGBoost's native prediction-contribution output to produce the top 3 reason codes for each single prediction.

For a default-risk prediction, the reason codes show the strongest positive model contributors. For a non-default prediction, they show the strongest protective contributors.

These reason codes explain model behavior. They are not a legal adverse-action notice and should be reviewed before production use.

## Partial Dependence Plots

Training generates partial dependence plots for the most important model variables:

```text
reports/pdp/
```

The plots show how the average predicted default probability changes as one feature changes while the rest of the validation population is held constant.

## AWS Hosting

Train the model first so the `artifacts/` and `reports/` folders exist, then build the image:

```bash
cd "/Users/nalinaggarwal/Desktop/Visual Studio/credit_default_xgboost"
docker build -t credit-default-xgboost .
docker run -p 8501:8501 credit-default-xgboost
```

AWS options:

- **AWS App Runner**: push this Docker image to Amazon ECR, then create an App Runner service from the ECR image.
- **Amazon ECS/Fargate**: push the image to ECR and create an ECS service exposing container port `8501`.
- **EC2**: copy this folder to the instance, install dependencies, and run `./run_app.sh` behind a reverse proxy or security group rule.

## Important Production Notes

- Validate fairness, bias, and regulatory requirements before any production credit decisioning use.
- Confirm that each field is available at decision time and is legally permissible.
- Monitor drift, approval rates, default rates, and reason-code stability over time.
- Re-train and revalidate before using new data or changed definitions.
