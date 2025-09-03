import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from fairlearn.metrics import MetricFrame, false_positive_rate, false_negative_rate
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from lime.lime_tabular import LimeTabularExplainer

# ========== Data preprocessing ==========
df = pd.read_csv('./Dataset/compas-scores-two-years.csv')
df = df[df['race'].isin(['African-American', 'Caucasian'])]

features = ['age', 'priors_count', 'sex', 'c_charge_degree']
df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
df['c_charge_degree'] = df['c_charge_degree'].map({'F': 1, 'M': 0})
df = df.dropna(subset=features + ['two_year_recid', 'race'])

X = df[features]
y = df['two_year_recid']

# ===== mapping =====
race_map = {'Caucasian': 0, 'African-American': 1}
race_num = df['race'].map(race_map)
sex_num = df['sex']


# ===== splitting function =====
def stratified_split(X, y, sensitive_features, test_size=0.3, random_state=42):
    if isinstance(sensitive_features, pd.DataFrame):
        stratify_col = sensitive_features.astype(str).agg('_'.join, axis=1)
    else:
        stratify_col = sensitive_features
    X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
        X, y, sensitive_features, test_size=test_size,
        random_state=random_state, stratify=stratify_col
    )
    return X_train, X_test, y_train, y_test, sf_train, sf_test


# ===== Divide the training set and the test set =====
X_train, X_test, y_train, y_test, race_train, race_test = stratified_split(X, y, race_num)
_, _, _, _, sex_train, sex_test = stratified_split(X, y, sex_num)
race_sex = pd.DataFrame({'race': race_num, 'sex': sex_num})
_, _, _, _, race_sex_train, race_sex_test = stratified_split(X, y, race_sex)

# ========== Evaluation function ==========
metrics = {
    'accuracy': lambda y_true, y_pred: (y_true == y_pred).mean(),
    'FPR': false_positive_rate,
    'FNR': false_negative_rate
}


def evaluate_model(name, y_pred, sensitive_features):
    metric_frame = MetricFrame(metrics=metrics,
                               y_true=y_test,
                               y_pred=y_pred,
                               sensitive_features=sensitive_features)
    print(f"\n{name} - Overall Metrics:")
    print(metric_frame.overall)
    print(f"\n{name} - Metrics by Group:")
    print(metric_frame.by_group)
    metric_frame.by_group.plot(kind='bar', subplots=True, layout=(1, 3), figsize=(15, 5), legend=False)
    plt.suptitle(f"Fairness Metrics by Group - {name}")
    plt.tight_layout()
    plt.show()


# ========== Baseline Decision Tree ==========
print("\n=== Baseline: Decision Tree ===")
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
evaluate_model("Decision Tree", y_pred, race_test)

# ========== Exponentiated Gradient ==========
print("\n=== Mitigation: Exponentiated Gradient (Decision Tree) ===")
mitigator = ExponentiatedGradient(
    estimator=DecisionTreeClassifier(max_depth=5, random_state=42),
    constraints=EqualizedOdds()
)
mitigator.fit(X_train, y_train, sensitive_features=race_train)
y_pred_expgrad = mitigator.predict(X_test)
evaluate_model("Exponentiated Gradient (Decision Tree)", y_pred_expgrad, race_test)

explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=list(X_train.columns),
    class_names=['no_recid', 'recid'],
    mode='classification',
    discretize_continuous=True
)

def predict_proba_baseline(X_input):
    X_df = pd.DataFrame(X_input, columns=X_train.columns)
    return dt.predict_proba(X_df)

# LIME
def predict_proba_expgrad(X_input):
    X_df = pd.DataFrame(X_input, columns=X_train.columns)
    # ExponentiatedGradient 对象没有 predict_proba，直接使用 predict 再转换
    preds = mitigator.predict(X_df)
    return np.vstack([1 - preds, preds]).T

# Select three test samples for partial explanation
for i in np.random.choice(len(X_test), 3, replace=False):
    print(f"\n--- LIME explanation for test instance {i} ---")
    exp_base = explainer.explain_instance(X_test.values[i], predict_proba_baseline)
    exp_expgrad = explainer.explain_instance(X_test.values[i], predict_proba_expgrad)

    print("\n[Baseline Decision Tree]")
    print(exp_base.as_list())

    print("\n[Exponentiated Gradient + Decision Tree]")
    print(exp_expgrad.as_list())

# ========== ThresholdOptimizer ==========
print("\n=== Mitigation: ThresholdOptimizer (Decision Tree) ===")
dt_prob = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_prob.fit(X_train, y_train)
postprocessor = ThresholdOptimizer(
    estimator=dt_prob,
    constraints="equalized_odds",
    prefit=True,
    predict_method="predict_proba"
)
postprocessor.fit(X_test, y_test, sensitive_features=race_test)
y_pred_post = postprocessor.predict(X_test, sensitive_features=race_test)
evaluate_model("ThresholdOptimizer (Decision Tree)", y_pred_post, race_test)

# preparation LIME
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=list(X_train.columns),
    class_names=['no_recid', 'recid'],
    mode='classification',
    discretize_continuous=True
)

def predict_proba_baseline(X_input):
    X_df = pd.DataFrame(X_input, columns=X_train.columns)
    return dt_prob.predict_proba(X_df)

def predict_proba_post(X_input):
    X_df = pd.DataFrame(X_input, columns=X_train.columns)
    # ThresholdOptimizer 没有 predict_proba，使用 predict 并转换成概率形式
    preds = postprocessor.predict(X_df, sensitive_features=[race_test.iloc[0]] * len(X_df))
    return np.vstack([1 - preds, preds]).T

for i in np.random.choice(len(X_test), 3, replace=False):
    print(f"\n--- LIME explanation for test instance {i} ---")
    exp_base = explainer.explain_instance(X_test.values[i], predict_proba_baseline)
    exp_post = explainer.explain_instance(X_test.values[i], predict_proba_post)

    print("\n[Baseline Decision Tree]")
    print(exp_base.as_list())

    print("\n[ThresholdOptimizer + Decision Tree]")
    print(exp_post.as_list())

# ========== Reweighing ==========
print("\n=== Mitigation: Reweighing (Decision Tree) ===")
sensitive_options = [
    {"name": "race", "protected": ["race"], "train_sf": race_train, "test_sf": race_test, "privileged": [[0]],
     "unprivileged": [{'race': 1}]},
    {"name": "sex", "protected": ["sex"], "train_sf": sex_train, "test_sf": sex_test, "privileged": [[0]],
     "unprivileged": [{'sex': 1}]},
    {"name": "race+sex", "protected": ["race", "sex"], "train_sf": race_sex_train, "test_sf": race_sex_test,
     "privileged": [[0, 0]], "unprivileged": [{'race': 1, 'sex': 1}]}
]

for opt in sensitive_options:
    print(f"\n=== Mitigation: Reweighing on {opt['name']} ===")

    # Construct the AIF360 dataset
    train_df = X_train.copy()
    train_df['two_year_recid'] = y_train.values
    for p in opt["protected"]:
        if p not in train_df.columns:
            if isinstance(opt["train_sf"], pd.DataFrame):
                train_df[p] = opt["train_sf"][p].values
            else:  # Series
                train_df[p] = opt["train_sf"].values

    test_df = X_test.copy()
    test_df['two_year_recid'] = y_test.values
    for p in opt["protected"]:
        if p not in test_df.columns:
            if isinstance(opt["test_sf"], pd.DataFrame):
                test_df[p] = opt["test_sf"][p].values
            else:
                test_df[p] = opt["test_sf"].values

    baseline_cols = X_train.columns.tolist()
    all_features = baseline_cols + [p for p in opt["protected"] if p not in baseline_cols]

    train_aif = StandardDataset(
        train_df,
        label_name='two_year_recid',
        favorable_classes=[0],
        protected_attribute_names=opt["protected"],
        privileged_classes=opt["privileged"]
    )
    test_aif = StandardDataset(
        test_df,
        label_name='two_year_recid',
        favorable_classes=[0],
        protected_attribute_names=opt["protected"],
        privileged_classes=opt["privileged"]
    )

    # Reweighing
    rw = Reweighing(
        unprivileged_groups=opt["unprivileged"],
        privileged_groups=[dict(zip(opt["protected"], p)) for p in opt["privileged"]]
    )
    rw.fit(train_aif)
    train_reweighed = rw.transform(train_aif)

    # training decision tree
    dt_rw = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_rw.fit(train_reweighed.features,
              train_reweighed.labels.ravel(),
              sample_weight=train_reweighed.instance_weights)

    # 基线决策树（不含 Reweighing）
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(train_aif.features[:, :len(baseline_cols)],
           train_aif.labels.ravel())

    y_pred_rw = dt_rw.predict(test_aif.features)

    if len(opt["protected"]) == 1:
        sensitive_features = test_df[opt["protected"][0]]
    else:
        sensitive_features = test_df[opt["protected"]].astype(str).agg('_'.join, axis=1)

    evaluate_model(f"Reweighing on {opt['name']} (Decision Tree)", y_pred_rw, sensitive_features)

    # LIME explainer
    explainer = LimeTabularExplainer(
        training_data=train_df[all_features].values,
        feature_names=all_features,
        class_names=['no_recid', 'recid'],
        mode='classification',
        discretize_continuous=True
    )

    def predict_proba_baseline(X_input):
        X_df = pd.DataFrame(X_input, columns=all_features)
        X_base = X_df[baseline_cols].to_numpy()  # 保证列顺序和训练时一致
        return dt.predict_proba(X_base)

    def predict_proba_rw(X_input, protected_values=None):
        X_df = pd.DataFrame(X_input, columns=all_features)
        if protected_values is not None:
            for p in opt["protected"]:
                X_df[p] = protected_values[p]
        X_rw = X_df[all_features].to_numpy()  # 保证列顺序一致
        return dt_rw.predict_proba(X_rw)

    for i in np.random.choice(len(test_df), 3, replace=False):
        protected_values = test_df.iloc[i][opt["protected"]]
        print(f"\n--- LIME explanation for test instance {i} ({opt['name']}) ---")

        exp_base = explainer.explain_instance(
            test_df[all_features].iloc[i].values,
            lambda x: predict_proba_baseline(x)
        )
        exp_rw = explainer.explain_instance(
            test_df[all_features].iloc[i].values,
            lambda x: predict_proba_rw(x, protected_values=protected_values)
        )

        print("\n[Baseline Decision Tree]")
        print(exp_base.as_list())

        print("\n[Reweighing + Decision Tree]")
        print(exp_rw.as_list())