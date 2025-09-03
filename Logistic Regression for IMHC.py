import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from fairlearn.metrics import MetricFrame, false_positive_rate, false_negative_rate
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer

from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from lime.lime_tabular import LimeTabularExplainer

# ========== Data preprocessing ==========
data_openml = fetch_openml(data_id=45040)
data = data_openml.data

# Labels and sensitive attributes
data["Diagnosis"] = data_openml.target
data['Race'] = data['Race'].str.strip()
data['Sex'] = data['Sex'].str.strip()
data['race_sex'] = data['Race'] + "_" + data['Sex']

# Binary classification label coding
le_label = LabelEncoder()
y = le_label.fit_transform(data['Diagnosis'])

exclude_cols = ['Diagnosis', 'Race', 'Sex', 'race_sex', 'dataset']
feature_cols = [col for col in data.columns if col not in exclude_cols]
X = pd.get_dummies(data[feature_cols])

race = data['Race']
sex = data['Sex']
race_sex = data['race_sex']

# Dataset division
X_train, X_test, y_train, y_test, race_train, race_test, sex_train, sex_test, race_sex_train, race_sex_test = train_test_split(
    X, y, race, sex, race_sex, test_size=0.3, random_state=42, stratify=y
)

sensitive_train = race_sex_train
sensitive_test = race_sex_test

# ========== evaluation function ==========
metrics = {
    'accuracy': accuracy_score,
    'FPR': false_positive_rate,
    'FNR': false_negative_rate
}

def evaluate_model(name, y_pred, y_true=None, sensitive_features=None):
    if y_true is None:
        y_true = y_test
    if sensitive_features is None:
        sensitive_features = sensitive_test

    metric_frame = MetricFrame( metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)

    print(f"\n{name} - Overall Metrics:")
    print(metric_frame.overall)
    print(f"\n{name} - Metrics by Race_Sex:")
    print(metric_frame.by_group)

    metric_frame.by_group.plot(kind='bar', subplots=True, layout=(1, 3), figsize=(15, 5), legend=False)
    plt.suptitle(f"Fairness Metrics by Race_Sex - {name}")
    plt.tight_layout()
    plt.show()


# ========== Baseline ==========
print("\n=== Baseline: Logistic Regression ===")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
evaluate_model("Logistic Regression", y_pred)


# ========== Exponentiated Gradient ==========
print("\n=== Mitigation: Exponentiated Gradient ===")
expgrad_lr = ExponentiatedGradient(
    estimator=LogisticRegression(max_iter=1000, random_state=3),
    constraints=EqualizedOdds()
)
expgrad_lr.fit(X_train, y_train, sensitive_features=sensitive_train)
y_pred_expgrad_lr = expgrad_lr.predict(X_test)
evaluate_model("Exponentiated Gradient", y_pred_expgrad_lr)

# LIME Interpreter
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=list(X_train.columns),
    class_names=list(le_label.classes_),
    mode='classification',
    discretize_continuous=True
)

def predict_proba_baseline(X_input):
    X_df = pd.DataFrame(X_input, columns=X_train.columns)
    return lr.predict_proba(X_df)

def predict_proba_expgrad(X_input):
    X_df = pd.DataFrame(X_input, columns=X_train.columns)
    preds = expgrad_lr.predict(X_df)
    return np.vstack([1 - preds, preds]).T

# Randomly select a few samples for explanation
for i in np.random.choice(len(X_test), 3, replace=False):
    print(f"\n--- LIME Explanation for test instance {i} ---")

    exp_before = explainer.explain_instance(X_test.values[i], predict_proba_baseline)
    exp_after = explainer.explain_instance(X_test.values[i], predict_proba_expgrad)

    print("Baseline Logistic Regression:", exp_before.as_list())
    print("After Mitigation (ExpGrad):", exp_after.as_list())

# ========== Threshold Optimizer ==========
print("\n=== Mitigation: ThresholdOptimizer (Equalized Odds Postprocessing) ===")

X_train_lime = X_train.copy()
X_test_lime  = X_test.copy()

explainer = LimeTabularExplainer(
    training_data=X_train_lime.values,
    feature_names=list(X_train_lime.columns),
    class_names=['negative', 'positive'],
    mode='classification',
    discretize_continuous=True
)

def predict_proba_baseline(X_input):
    X_df = pd.DataFrame(X_input, columns=X_train_lime.columns)
    return lr_prob.predict_proba(X_df)

lr_prob = LogisticRegression(max_iter=1000, random_state=2)
lr_prob.fit(X_train, y_train)

# Define different sensitive attributes
sensitive_options = {
    "Race": (race_train, race_test),
    "Sex": (sex_train, sex_test),
    "Race+Sex": (race_sex_train, race_sex_test)
}

for name, (sens_train, sens_test) in sensitive_options.items():
    print(f"\n--- ThresholdOptimizer using sensitive attribute: {name} ---")

    postprocessor_lr = ThresholdOptimizer(
        estimator=lr_prob,
        constraints="equalized_odds",
        prefit=True,
        predict_method="predict_proba"
    )
    postprocessor_lr.fit(X_test, y_test, sensitive_features=sens_test)

    # prediction
    y_pred_post = postprocessor_lr.predict(X_test, sensitive_features=sens_test)

    # evaluation
    evaluate_model(f"ThresholdOptimizer ({name})", y_pred_post, y_true=y_test, sensitive_features=sens_test)

    # Define the post-mitigation prediction function (return probability)
    def predict_proba_post(X_input):
        X_df = pd.DataFrame(X_input, columns=X_train_lime.columns)
        preds = postprocessor_lr.predict(X_df, sensitive_features=[sens_test.iloc[0]] * len(X_input))
        return np.vstack([1 - preds, preds]).T

    # LIME Comparative Explanation
    for i in np.random.choice(len(X_test_lime), 3, replace=False):
        print(f"\n--- LIME explanation for test instance {i}, sensitive={name} ---")

        exp_base = explainer.explain_instance(X_test_lime.values[i], predict_proba_baseline)
        exp_post = explainer.explain_instance(X_test_lime.values[i], predict_proba_post)

        print("\nBaseline Logistic Regression:", exp_base.as_list())
        print("\nAfter Mitigation (ExpGrad):", exp_post.as_list())


# ========== Reweighing (AIF360) ==========
print("\n=== Mitigation: Reweighing (Preprocessing) ===")

explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=list(X_train.columns),
    class_names=['negative', 'positive'],
    mode='classification',
    discretize_continuous=True
)

def predict_proba_baseline(X_input):
    X_df = pd.DataFrame(X_input, columns=X_train.columns)
    return lr.predict_proba(X_df)

def iterative_reweighting(X_train, y_train, sensitive_train,
                                        X_test, y_test, sensitive_test,
                                        base_estimator, max_iter=10, lr=0.5):

    sample_weights = np.ones(len(X_train))

    for it in range(max_iter):
        model = base_estimator
        model.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred_train = model.predict(X_train)
        mf_train = MetricFrame(metrics={'FPR': false_positive_rate, 'FNR': false_negative_rate},
                               y_true=y_train, y_pred=y_pred_train,
                               sensitive_features=sensitive_train)

        # Update the weights by group
        fpr_by_group = mf_train.by_group['FPR']
        fnr_by_group = mf_train.by_group['FNR']

        overall_fpr = fpr_by_group.mean()
        overall_fnr = fnr_by_group.mean()

        for group in fpr_by_group.index:
            mask = (sensitive_train == group)
            # Calculate the residuals: the differences between FPR, FNR and the whole
            fpr_diff = fpr_by_group[group] - overall_fpr
            fnr_diff = fnr_by_group[group] - overall_fnr
            fairness_residual = abs(fpr_diff) + abs(fnr_diff)

            # Weight update
            sample_weights[mask] *= np.exp(lr * fairness_residual)

        sample_weights /= np.mean(sample_weights)

        print(f"Round {it+1} - Train metrics by group:")
        print(mf_train.by_group)

    # final mould
    final_model = base_estimator
    final_model.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred_final = final_model.predict(X_test)

    return final_model, y_pred_final


# reweighting
base_lr = LogisticRegression(max_iter=1000, random_state=42)
final_model, y_pred_reweight_lr = iterative_reweighting(
    X_train, y_train, sensitive_train,
    X_test, y_test, sensitive_test,
    base_lr,
    max_iter=5, lr=0.8
)

evaluate_model("Iterative Reweighting + Logistic Regression", y_pred_reweight_lr)

# LIME partial explanation comparison
def predict_proba_reweight(X_input):
    X_df = pd.DataFrame(X_input, columns=X_train.columns)
    preds = final_model.predict(X_df)
    return np.vstack([1 - preds, preds]).T

for i in np.random.choice(len(X_test), 3, replace=False):
    print(f"\n--- LIME explanation for test instance {i} ---")

    exp_base = explainer.explain_instance(X_test.values[i], predict_proba_baseline)
    exp_reweight = explainer.explain_instance(X_test.values[i], predict_proba_reweight)

    print("\nBaseline Logistic Regression:", exp_base.as_list())
    print("\nAfter Mitigation (ExpGrad):", exp_reweight.as_list())