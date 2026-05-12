# Credit Default XGBoost Model Report

## Hyperparameter Search

- Search method: bayesian
- Search objective: maximize validation ROC-AUC
- Evaluated configurations: 90

## Measurement Method

Primary model measurement: **ROC-AUC**, translated to **Gini = 2 x AUC - 1**.

Why: credit default scoring is a ranking/discrimination problem. ROC-AUC is threshold-independent, is robust for this 30% default-rate dataset, and maps to the Gini measure commonly used in credit-risk model monitoring. Precision-recall AUC, KS, F1, recall, precision, and confusion matrix are reported as supporting measures.

## Validation Metrics

- target: credit_card_default
- train_rows: 7000
- validation_rows: 3000
- validation_size: 0.3
- positive_default_rate_train: 0.29928571428571427
- positive_default_rate_validation: 0.29933333333333334
- primary_measurement_method: ROC-AUC / Gini
- hyperparameter_search_method: bayesian
- hyperparameter_search_objective: maximize validation ROC-AUC
- hyperparameter_search_evaluations: 90
- best_hyperparameters: {'n_estimators': 226, 'max_depth': 2, 'learning_rate': 0.03569289, 'min_child_weight': 6.45174431, 'subsample': 0.97155331, 'colsample_bytree': 0.96386298, 'reg_lambda': 16.98982705, 'reg_alpha': 1.84235857, 'gamma': 1.58167425}
- roc_auc: 0.774645
- gini: 0.54929
- average_precision_pr_auc: 0.672665
- ks_statistic: 0.424232
- log_loss: 0.549963
- classification_threshold: 0.515
- precision_at_threshold: 0.583512
- recall_at_threshold: 0.606904
- f1_at_threshold: 0.594978

Confusion matrix order: actual rows [non-default, default] x predicted columns [non-default, default].

```json
[
  [
    1713,
    389
  ],
  [
    353,
    545
  ]
]
```

## Selected Features

- **Payment-to-income ratio** (`payment_to_income_ratio`): Compares repayment behavior with income capacity.
- **Annual income** (`annual_income`): Represents borrower repayment capacity.
- **Worst recent payment delay** (`max_pay_delay`): Captures the most severe recent repayment delay.
- **Number of late payments** (`num_late_payments`): Counts how often recent payments were late.
- **Recent credit inquiries** (`num_credit_inquiries_6m`): Frequent recent credit searches may indicate liquidity pressure.
- **Autopay enrolled** (`autopay_enrolled`): Autopay can reduce missed-payment risk.
- **Customer service calls in 6 months** (`customer_service_calls_6m`): Frequent service contact can indicate account stress or disputes.
- **Minimum payment flag** (`min_pay_flag`): Indicates reliance on minimum payments.
- **Public records** (`num_public_records`): Captures public credit events that may indicate elevated risk.
- **Credit utilization** (`credit_utilization`): Shows how much of the available credit line is currently being used.
- **Delinquencies in last 2 years** (`num_delinquencies_2yr`): Counts recent serious repayment issues.
- **Payment-to-bill ratio** (`pay_to_bill_ratio`): Compares recent payments with billed amounts.
- **Open accounts** (`num_open_accounts`): Shows breadth of current credit obligations.
- **Spend-to-limit ratio** (`spend_to_limit_ratio`): Compares spending level with the credit limit.
- **Bankruptcy flag** (`bankruptcy_flag`): Indicates whether a bankruptcy signal is present.
- **Returned payment count** (`returned_payment_count`): Counts failed or returned payments.
- **Income stability index** (`income_stability_index`): Captures the steadiness of income over time.
- **Cash advance usage** (`cash_advance_usage_pct`): Cash advance reliance can indicate short-term liquidity stress.

## Features Rejected By Correlation Rule

- `credit_limit_to_income` rejected because it correlated with `payment_to_income_ratio` at 0.945.
- `monthly_obligation_ratio` rejected because it correlated with `payment_to_income_ratio` at 0.984.
- `spend_to_income_ratio` rejected because it correlated with `payment_to_income_ratio` at 0.963.
- `debt_to_income_ratio` rejected because it correlated with `payment_to_income_ratio` at 0.783.
- `avg_pay_amt` rejected because it correlated with `payment_to_income_ratio` at 0.782.
- `limit_bal` rejected because it correlated with `payment_to_income_ratio` at 0.736.
- `current_balance` rejected because it correlated with `payment_to_income_ratio` at 0.606.
- `repeat_late_payer_flag` rejected because it correlated with `num_late_payments` at 0.650.
- `payment_consistency_score` rejected because it correlated with `num_late_payments` at 1.000.
- `balance_to_limit_6m_avg` rejected because it correlated with `credit_utilization` at 0.970.

## Best XGBoost Parameter Sets By Validation ROC-AUC

```text
search_method  search_step                  search_phase  acquisition_expected_improvement  n_estimators  max_depth  learning_rate  min_child_weight  subsample  colsample_bytree  reg_lambda  reg_alpha    gamma  roc_auc     gini  average_precision  log_loss
     bayesian           21 bayesian_expected_improvement                          0.003804           226          2       0.035693          6.451744   0.971553          0.963863   16.989827   1.842359 1.581674 0.774645 0.549290           0.672665  0.549963
     bayesian           80 bayesian_expected_improvement                          0.000006           349          2       0.024544          1.052400   0.760352          0.923151   23.015902   0.124115 1.068748 0.774481 0.548962           0.673139  0.548852
     bayesian           25 bayesian_expected_improvement                          0.000435           443          2       0.019556          1.202194   0.726990          0.955895   23.406385   0.687900 2.189547 0.774441 0.548881           0.673435  0.549195
     bayesian           68 bayesian_expected_improvement                          0.000062           182          2       0.030905          1.677652   0.881465          0.983297    7.207829   0.012968 1.056270 0.774300 0.548600           0.671025  0.551656
     bayesian           45 bayesian_expected_improvement                          0.000039           344          2       0.015844          7.202806   0.650667          0.983271    0.485052   0.654922 2.584908 0.774180 0.548360           0.672271  0.551448
     bayesian           83 bayesian_expected_improvement                          0.000010           353          2       0.020225          5.016702   0.943975          0.999042    7.982185   0.000817 1.832594 0.774171 0.548342           0.671971  0.549941
     bayesian           35 bayesian_expected_improvement                          0.000810           423          2       0.015436          2.518075   0.965388          0.855139   12.863490   4.986721 0.535850 0.774115 0.548229           0.671815  0.551783
     bayesian           47 bayesian_expected_improvement                          0.000315           202          2       0.024993          3.204893   0.962742          0.994265    0.720041   0.171814 2.563162 0.774090 0.548179           0.670808  0.552577
     bayesian           26 bayesian_expected_improvement                          0.000365           183          2       0.029151          0.914940   0.838739          0.809502    0.959481   0.825644 1.514841 0.774083 0.548167           0.671803  0.552132
     bayesian           64 bayesian_expected_improvement                          0.000060           286          2       0.030179          7.209902   0.897451          0.993024   21.285607   0.359816 2.734447 0.774058 0.548117           0.672670  0.549644
```
