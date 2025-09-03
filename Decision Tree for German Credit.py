import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, false_positive_rate, false_negative_rate
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer
from aif360.datasets import StandardDataset
from aif360.algorithms.preprocessing import Reweighing
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from lime.lime_tabular import LimeTabularExplainer

TEST_SIZE = 0.3
RANDOM_STATE = 42

# ----------------- Read & Basic mapping -----------------
statlog_german_credit_data = fetch_ucirepo(id=144)
X = statlog_german_credit_data.data.features.copy()
y = statlog_german_credit_data.data.targets.copy()
y = y.replace({1: 0, 2: 1}).values.ravel()

mapping = {
    'Attribute1': {'A11':'< 0 DM', 'A12':'0 <= ... < 200 DM', 'A13':'>=200 DM / salary >=1yr', 'A14':'no checking account'},
    'Attribute3': {'A30':'no credits taken / all credits paid back duly', 'A31':'all credits at this bank paid back duly',
                   'A32':'existing credits paid back duly till now', 'A33':'delay in paying off in the past', 'A34':'critical account / other credits existing'},
    'Attribute4': {'A40':'car (new)','A41':'car (used)','A42':'furniture/equipment','A43':'radio/television','A44':'domestic appliances',
                   'A45':'repairs','A46':'education','A47':'vacation','A48':'retraining','A49':'business','A410':'others'},
    'Attribute6': {'A61':'<100 DM','A62':'100<=...<500 DM','A63':'500<=...<1000 DM','A64':'>=1000 DM','A65':'unknown / no savings'},
    'Attribute7': {'A71':'unemployed','A72':'<1 year','A73':'1<=...<4 years','A74':'4<=...<7 years','A75':'>=7 years'},
    'Attribute9': {'A91':'male: divorced/separated','A92':'female: divorced/separated/married','A93':'male: single',
                   'A94':'male: married/widowed','A95':'female: single'},
    'Attribute10': {'A101':'none','A102':'co-applicant','A103':'guarantor'},
    'Attribute12': {'A121':'real estate','A122':'building society/life insurance','A123':'car/other','A124':'unknown / no property'},
    'Attribute14': {'A141':'bank','A142':'stores','A143':'none'},
    'Attribute15': {'A151':'rent','A152':'own','A153':'for free'},
    'Attribute17': {'A171':'unemployed/unskilled non-resident','A172':'unskilled resident','A173':'skilled employee / official','A174':'management/self-employed/highly qualified'},
    'Attribute19': {'A191':'none','A192':'yes, registered'},
    'Attribute20': {'A201':'yes','A202':'no'}
}
for col, map_dict in mapping.items():
    X.loc[:, col] = X[col].replace(map_dict)

column_rename = {
    'Attribute1': 'checking_status',
    'Attribute2': 'duration_month',
    'Attribute3': 'credit_history',
    'Attribute4': 'purpose',
    'Attribute5': 'credit_amount',
    'Attribute6': 'savings_status',
    'Attribute7': 'employment_since',
    'Attribute8': 'installment_rate',
    'Attribute9': 'personal_status_sex',
    'Attribute10': 'other_debtors',
    'Attribute11': 'present_residence_since',
    'Attribute12': 'property',
    'Attribute13': 'age',
    'Attribute14': 'other_installment_plans',
    'Attribute15': 'housing',
    'Attribute16': 'existing_credits',
    'Attribute17': 'job',
    'Attribute18': 'people_liable',
    'Attribute19': 'telephone',
    'Attribute20': 'foreign_worker'
}
X.rename(columns=column_rename, inplace=True)

# ----------------- sensitive attributes -----------------
def make_age_group(a):
    try:
        a = float(a)
    except:
        return "unknown"
    if a < 25:
        return "<25"
    if a <= 50:
        return "25-50"
    return ">50"

sensitive_df = pd.DataFrame(index=X.index)
sensitive_df['personal_status_sex'] = X['personal_status_sex'].astype(str)
sensitive_df['age_group'] = X['age'].apply(make_age_group)
sensitive_df['residence_since'] = X['present_residence_since'].astype(str)
sensitive_df['housing'] = X['housing'].astype(str)
sensitive_df['savings_status'] = X['savings_status'].astype(str)
sensitive_df['checking_status'] = X['checking_status'].astype(str)
sensitive_df['property'] = X['property'].astype(str)

X_model = X.drop(columns=['personal_status_sex'])

# ----------------- Divide set -----------------
stratify_col = pd.Series([f"{int(lbl)}__{s}" for lbl, s in zip(y, sensitive_df['personal_status_sex'])])
X_train, X_test, y_train, y_test, sens_train_df, sens_test_df = train_test_split(
    X_model, y, sensitive_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_col
)

# ----------------- Preprocessing -----------------
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

numeric_transformer = Pipeline([('scaler', StandardScaler())])
try:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

categorical_transformer = Pipeline([('onehot', ohe)])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
], sparse_threshold=0)

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# ----------------- evaluation function -----------------
metrics = {'accuracy': accuracy_score, 'FPR': false_positive_rate, 'FNR': false_negative_rate}

def evaluate_and_print(title, y_true, y_pred, sensitive_series):
    mf = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_series)
    print(f"\n=== {title} ===")
    print("Overall:\n", mf.overall)
    print("By group:\n", mf.by_group)
    mf.by_group.plot(kind='bar', subplots=True, layout=(1,3), figsize=(15,4), legend=False)
    plt.suptitle(f"{title} - metrics by sensitive group")
    plt.tight_layout()
    plt.show()

# ----------------- Baseline Decision Tree -----------------
clf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5))
])
clf_pipeline.fit(X_train, y_train)
y_pred_baseline = clf_pipeline.predict(X_test)
evaluate_and_print("Baseline DecisionTree", y_test, y_pred_baseline, sens_test_df['personal_status_sex'])

# ----------------- ExponentiatedGradient -----------------
base_est = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5)
sensitive_to_run = [
    'personal_status_sex',
    'age_group',
    'residence_since',
    'housing',
    'savings_status',
    'checking_status',
    'property'
]

# Training data for LIME interpretation
X_train_lime = X_train.copy()
X_test_lime = X_test.copy()

oe = OrdinalEncoder()
X_train_lime[categorical_features] = oe.fit_transform(X_train_lime[categorical_features])
X_test_lime[categorical_features] = oe.transform(X_test_lime[categorical_features])

# Classification column index
categorical_indices = [X_train_lime.columns.get_loc(c) for c in categorical_features]

for attr in sensitive_to_run:
    print(f"\n>>> Running ExponentiatedGradient with sensitive attribute = {attr}")

    s_train = sens_train_df[attr].values
    s_test = sens_test_df[attr].values

    # train ExponentiatedGradient
    expgrad = ExponentiatedGradient(base_est, constraints=EqualizedOdds())
    expgrad.fit(X_train_proc, y_train, sensitive_features=s_train)
    y_pred_eg = expgrad.predict(X_test_proc)

    evaluate_and_print(f"ExpGrad (sensitive={attr})", y_test, y_pred_eg, s_test)


    # Define the LIME prediction function
    def predict_proba_expgrad(X_input):
        X_df = pd.DataFrame(X_input, columns=X_train_lime.columns)

        for c in categorical_features:
            X_df[c] = np.round(X_df[c]).astype(X_train[c].dtype)

        X_proc = preprocessor.transform(X_df)

        pred = expgrad.predict(X_proc)

        return np.vstack([1 - pred, pred]).T

    explainer_eg = LimeTabularExplainer(
        training_data=X_train_lime.values,
        feature_names=list(X_train_lime.columns),
        class_names=['bad', 'good'],
        mode='classification',
        categorical_features=categorical_indices,
        discretize_continuous=True
    )

    # Explain several test samples
    for i in np.random.choice(len(X_test_lime), 5, replace=False):
        exp = explainer_eg.explain_instance(X_test_lime.values[i], predict_proba_expgrad)
        print(f"\n--- LIME ExpGrad Explanation for test instance {i}, sensitive={attr} ---")
        print(exp.as_list())

# ----------------- ThresholdOptimizer -----------------
print("\n>>> Running ThresholdOptimizer (Equalized Odds) using personal_status_sex")
prob_pipeline = Pipeline([('preprocessor', preprocessor),
                          ('classifier', DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5))])
prob_pipeline.fit(X_train, y_train)

postproc = ThresholdOptimizer(estimator=prob_pipeline, constraints="equalized_odds", prefit=True, predict_method="predict_proba")
postproc.fit(X_train, y_train, sensitive_features=sens_train_df['personal_status_sex'])
y_pred_to = postproc.predict(X_test, sensitive_features=sens_test_df['personal_status_sex'])
evaluate_and_print("ThresholdOptimizer (EqualizedOdds) - personal_status_sex", y_test, y_pred_to, sens_test_df['personal_status_sex'])

X_train_lime = X_train.copy()
X_test_lime = X_test.copy()

oe = OrdinalEncoder()
X_train_lime[categorical_features] = oe.fit_transform(X_train_lime[categorical_features])
X_test_lime[categorical_features] = oe.transform(X_test_lime[categorical_features])

categorical_indices = [X_train_lime.columns.get_loc(c) for c in categorical_features]

def predict_proba_thresholdoptimizer(X_input):
    X_df = pd.DataFrame(X_input, columns=X_train_lime.columns)

    X_df[categorical_features] = oe.inverse_transform(X_df[categorical_features])

    sens_array = np.repeat(sens_test_df['personal_status_sex'].values[0], X_df.shape[0])

    pred = postproc.predict(X_df, sensitive_features=sens_array)

    return np.vstack([1 - pred, pred]).T

explainer_to = LimeTabularExplainer(
    training_data=X_train_lime.values,
    feature_names=list(X_train_lime.columns),
    class_names=['bad', 'good'],
    mode='classification',
    categorical_features=categorical_indices,
    discretize_continuous=True
)

for i in np.random.choice(len(X_test_lime), 5, replace=False):
    exp = explainer_to.explain_instance(X_test_lime.values[i], predict_proba_thresholdoptimizer)
    print(f"\n--- LIME ThresholdOptimizer Explanation for test instance {i} ---")
    print(exp.as_list())

# ----------------- Reweighing  -----------------
print("\n>>> Running AIF360 Reweighing (binary sex)")
train_df = X_train.copy()
train_df['credit_class'] = y_train
train_df['sex'] = sens_train_df['personal_status_sex'].apply(lambda v: 1 if 'male' in str(v) else 0).values

test_df = X_test.copy()
test_df['credit_class'] = y_test
test_df['sex'] = sens_test_df['personal_status_sex'].apply(lambda v: 1 if 'male' in str(v) else 0).values

aif_categorical = categorical_features.copy()
le_dict = {}
for col in aif_categorical:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    le_dict[col] = le

train_aif = StandardDataset(train_df, label_name='credit_class', favorable_classes=[1],
                            protected_attribute_names=['sex'], privileged_classes=[[1]],
                            features_to_keep=None, categorical_features=aif_categorical)
test_aif = StandardDataset(test_df, label_name='credit_class', favorable_classes=[1],
                           protected_attribute_names=['sex'], privileged_classes=[[1]],
                           features_to_keep=None, categorical_features=aif_categorical)

rw = Reweighing(unprivileged_groups=[{'sex':0}], privileged_groups=[{'sex':1}])
rw.fit(train_aif)
train_rw = rw.transform(train_aif)

dt_rw = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=5)
dt_rw.fit(train_rw.features, train_rw.labels.ravel(), sample_weight=train_rw.instance_weights)
y_pred_rw = dt_rw.predict(test_aif.features)
evaluate_and_print("AIF360 Reweighing (binary sex)", y_test, y_pred_rw, sens_test_df['personal_status_sex'])

# Raw training data for LIME
X_train_lime = X_train.copy()
X_test_lime = X_test.copy()

# Convert the classification column to the value of LabelEncoder
for col in aif_categorical:
    X_train_lime[col] = le_dict[col].transform(X_train_lime[col].astype(str))
    X_test_lime[col] = le_dict[col].transform(X_test_lime[col].astype(str))

categorical_indices = [X_train_lime.columns.get_loc(c) for c in categorical_features]

# LIME prediction function
def predict_proba_rw(X_input, sex_value=None):
    X_df = pd.DataFrame(X_input, columns=X_train.columns)

    X_df['sex'] = sex_value

    X_df['credit_class'] = 0

    dataset = StandardDataset(
        X_df,
        label_name='credit_class',
        favorable_classes=[1],
        protected_attribute_names=['sex'],
        privileged_classes=[[1]],
        features_to_keep=None,
        categorical_features=aif_categorical
    )

    X_rw = rw.transform(dataset).features

    pred = dt_rw.predict(X_rw)

    return np.vstack([1 - pred, pred]).T


explainer_rw = LimeTabularExplainer(
    training_data=X_train_lime.values,
    feature_names=list(X_train_lime.columns),
    class_names=['bad', 'good'],
    mode='classification',
    categorical_features=categorical_indices,
    discretize_continuous=True
)

for i in np.random.choice(len(X_test_lime), 5, replace=False):
    # Get the true sex value of the current sample
    sex_value = test_df.iloc[i]['sex']

    # Wrap predict_proba_rw with lambda and pass sex_value into it
    exp = explainer_rw.explain_instance(
        X_test_lime.values[i],
        lambda x: predict_proba_rw(x, sex_value=sex_value)
    )

    print(f"\n--- LIME Reweighing Explanation for test instance {i} ---")
    print(exp.as_list())