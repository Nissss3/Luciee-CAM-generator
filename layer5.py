import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import json
from datetime import datetime

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (roc_auc_score, classification_report, confusion_matrix,
                              roc_curve, precision_recall_curve, average_precision_score)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb
import lightgbm as lgb
import shap

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

print("✅ All libraries loaded successfully")
print(f"📅 Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================
# SECTION 1 — DATA LOADING
# ============================================================
print("\n" + "="*60)
print("SECTION 1: LOADING DATA")
print("="*60)

def load_data(data_path="./"):
    """
    Load all Home Credit files.
    data_path: folder where your CSV files are stored
    In Colab, upload files and set data_path = "/content/"
    """
    print("Loading application_train.csv ...")
    app_train = pd.read_csv(r"C:\Users\anisk\Downloads\home-credit-default-risk\application_train.csv")

    print("Loading bureau.csv ...")
    bureau = pd.read_csv(r"C:\Users\anisk\Downloads\home-credit-default-risk\bureau.csv")

    print("Loading bureau_balance.csv ...")
    bureau_bal = pd.read_csv(r"C:\Users\anisk\Downloads\home-credit-default-risk\bureau_balance.csv")

    print("Loading previous_application.csv ...")
    prev_app = pd.read_csv(r"C:\Users\anisk\Downloads\home-credit-default-risk\previous_application.csv")

    print("Loading installments_payments.csv ...")
    installments = pd.read_csv(r"C:\Users\anisk\Downloads\home-credit-default-risk\installments_payments.csv")

    print("Loading credit_card_balance.csv ...")
    cc_bal = pd.read_csv(r"C:\Users\anisk\Downloads\home-credit-default-risk\credit_card_balance.csv")

    print("Loading POS_CASH_balance.csv ...")
    pos_cash = pd.read_csv(r"C:\Users\anisk\Downloads\home-credit-default-risk\POS_CASH_balance.csv")

    print(f"\n✅ application_train shape: {app_train.shape}")
    print(f"   Target distribution:\n{app_train['TARGET'].value_counts()}")
    print(f"   Default rate: {app_train['TARGET'].mean()*100:.2f}%")

    return app_train, bureau, bureau_bal, prev_app, installments, cc_bal, pos_cash


# ============================================================
# SECTION 2 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "="*60)
print("SECTION 2: EDA")
print("="*60)

def run_eda(app_train):
    """Quick EDA — understand data quality and distributions."""

    print(f"Total applications: {len(app_train):,}")
    print(f"Total features: {app_train.shape[1]}")
    print(f"\nDefault rate: {app_train['TARGET'].mean()*100:.2f}% "
          f"({app_train['TARGET'].sum():,} defaults out of {len(app_train):,})")

    # Missing value analysis
    missing = app_train.isnull().sum()
    missing_pct = (missing / len(app_train) * 100).sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 40]
    print(f"\nFeatures with >40% missing values: {len(high_missing)}")
    print("These will be dropped automatically.")

    # Plot target distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Target imbalance
    app_train['TARGET'].value_counts().plot(kind='bar', ax=axes[0],
                                             color=['steelblue', 'crimson'])
    axes[0].set_title('Target Distribution\n(0=No Default, 1=Default)')
    axes[0].set_xlabel('')

    # Income distribution by target
    app_train[app_train['TARGET']==0]['AMT_INCOME_TOTAL'].clip(0, 500000).plot(
        kind='hist', ax=axes[1], alpha=0.6, label='No Default', bins=50, color='steelblue')
    app_train[app_train['TARGET']==1]['AMT_INCOME_TOTAL'].clip(0, 500000).plot(
        kind='hist', ax=axes[1], alpha=0.6, label='Default', bins=50, color='crimson')
    axes[1].set_title('Income Distribution by Default')
    axes[1].legend()

    # Credit amount vs income
    axes[2].scatter(app_train['AMT_INCOME_TOTAL'].clip(0, 500000),
                    app_train['AMT_CREDIT'].clip(0, 2000000),
                    c=app_train['TARGET'], cmap='coolwarm', alpha=0.1, s=1)
    axes[2].set_title('Credit Amount vs Income\n(Red=Default)')
    axes[2].set_xlabel('Income')
    axes[2].set_ylabel('Credit Amount')

    plt.tight_layout()
    plt.savefig('eda_overview.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ EDA plot saved as eda_overview.png")

    return missing_pct


# ============================================================
# SECTION 3 — FEATURE ENGINEERING
# ============================================================
print("\n" + "="*60)
print("SECTION 3: FEATURE ENGINEERING")
print("="*60)

def engineer_bureau_features(bureau, bureau_bal):
    """Aggregate bureau (credit history) data per applicant."""
    print("  Engineering bureau features...")

    # Bureau balance aggregations
    bb_agg = bureau_bal.groupby('SK_ID_BUREAU').agg(
        MONTHS_BALANCE_MEAN=('MONTHS_BALANCE', 'mean'),
        MONTHS_BALANCE_MAX=('MONTHS_BALANCE', 'max'),
        STATUS_C_COUNT=('STATUS', lambda x: (x == 'C').sum()),  # Closed
        STATUS_X_COUNT=('STATUS', lambda x: (x == 'X').sum()),  # No info
        DPD_COUNT=('STATUS', lambda x: x.isin(['1','2','3','4','5']).sum()),  # Days past due
    ).reset_index()

    bureau_with_bal = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')

    # Bureau aggregations per applicant
    bureau_agg = bureau_with_bal.groupby('SK_ID_CURR').agg(
        BUREAU_LOAN_COUNT=('SK_ID_BUREAU', 'count'),
        BUREAU_ACTIVE_LOANS=('CREDIT_ACTIVE', lambda x: (x == 'Active').sum()),
        BUREAU_CLOSED_LOANS=('CREDIT_ACTIVE', lambda x: (x == 'Closed').sum()),
        BUREAU_CREDIT_SUM_MEAN=('AMT_CREDIT_SUM', 'mean'),
        BUREAU_CREDIT_SUM_MAX=('AMT_CREDIT_SUM', 'max'),
        BUREAU_CREDIT_DEBT_SUM=('AMT_CREDIT_SUM_DEBT', 'sum'),
        BUREAU_OVERDUE_SUM=('AMT_CREDIT_SUM_OVERDUE', 'sum'),
        BUREAU_DAYS_CREDIT_MEAN=('DAYS_CREDIT', 'mean'),
        BUREAU_DPD_MAX=('CREDIT_DAY_OVERDUE', 'max'),
        BUREAU_DPD_MEAN=('CREDIT_DAY_OVERDUE', 'mean'),
        BUREAU_PROLONGED_COUNT=('CNT_CREDIT_PROLONG', 'sum'),
        BUREAU_DPD_HISTORY=('DPD_COUNT', 'sum'),
    ).reset_index()

    # Derived bureau ratios
    bureau_agg['BUREAU_DEBT_TO_CREDIT_RATIO'] = (
        bureau_agg['BUREAU_CREDIT_DEBT_SUM'] /
        (bureau_agg['BUREAU_CREDIT_SUM_MAX'] + 1)
    )
    bureau_agg['BUREAU_ACTIVE_RATIO'] = (
        bureau_agg['BUREAU_ACTIVE_LOANS'] /
        (bureau_agg['BUREAU_LOAN_COUNT'] + 1)
    )

    print(f"  ✅ Bureau features: {bureau_agg.shape[1]-1} new features for "
          f"{bureau_agg.shape[0]:,} applicants")
    return bureau_agg


def engineer_prev_app_features(prev_app):
    """Aggregate previous application data per applicant."""
    print("  Engineering previous application features...")

    prev_agg = prev_app.groupby('SK_ID_CURR').agg(
        PREV_APP_COUNT=('SK_ID_PREV', 'count'),
        PREV_APPROVED_COUNT=('NAME_CONTRACT_STATUS', lambda x: (x == 'Approved').sum()),
        PREV_REFUSED_COUNT=('NAME_CONTRACT_STATUS', lambda x: (x == 'Refused').sum()),
        PREV_CREDIT_MEAN=('AMT_CREDIT', 'mean'),
        PREV_CREDIT_MAX=('AMT_CREDIT', 'max'),
        PREV_ANNUITY_MEAN=('AMT_ANNUITY', 'mean'),
        PREV_DAYS_DECISION_MEAN=('DAYS_DECISION', 'mean'),
        PREV_DOWN_PAYMENT_MEAN=('AMT_DOWN_PAYMENT', 'mean'),
        PREV_INTEREST_RATE_MEAN=('RATE_INTEREST_PRIMARY', 'mean'),
    ).reset_index()

    prev_agg['PREV_APPROVAL_RATE'] = (
        prev_agg['PREV_APPROVED_COUNT'] /
        (prev_agg['PREV_APP_COUNT'] + 1)
    )

    print(f"  ✅ Previous app features: {prev_agg.shape[1]-1} new features")
    return prev_agg


def engineer_installment_features(installments):
    """Capture payment behavior from installment history."""
    print("  Engineering installment payment features...")

    installments['PAYMENT_DIFF'] = (
        installments['AMT_INSTALMENT'] - installments['AMT_PAYMENT']
    )
    installments['DAYS_LATE'] = (
        installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
    ).clip(lower=0)
    installments['PAID_LATE_FLAG'] = (installments['DAYS_LATE'] > 0).astype(int)

    inst_agg = installments.groupby('SK_ID_CURR').agg(
        INST_COUNT=('SK_ID_PREV', 'count'),
        INST_PAYMENT_DIFF_MEAN=('PAYMENT_DIFF', 'mean'),
        INST_PAYMENT_DIFF_MAX=('PAYMENT_DIFF', 'max'),
        INST_DAYS_LATE_MEAN=('DAYS_LATE', 'mean'),
        INST_DAYS_LATE_MAX=('DAYS_LATE', 'max'),
        INST_LATE_RATE=('PAID_LATE_FLAG', 'mean'),
        INST_LATE_COUNT=('PAID_LATE_FLAG', 'sum'),
    ).reset_index()

    print(f"  ✅ Installment features: {inst_agg.shape[1]-1} new features")
    return inst_agg


def engineer_pos_cash_features(pos_cash):
    """
    POS_CASH_balance.csv — tracks monthly status of previous POS/cash loans.
    Key signals: how often was borrower late? How many loans completed vs active?
    This is a direct behavioral default signal — very predictive.
    """
    print("  Engineering POS CASH features...")

    # Flag months where borrower was overdue
    pos_cash['DPD_FLAG'] = (pos_cash['SK_DPD'] > 0).astype(int)
    pos_cash['DPD_DEF_FLAG'] = (pos_cash['SK_DPD_DEF'] > 0).astype(int)

    # Remaining installments ratio — how far along is the loan?
    pos_cash['INSTALMENT_COMPLETION_RATIO'] = (
        1 - pos_cash['CNT_INSTALMENT_FUTURE'] / (pos_cash['CNT_INSTALMENT'] + 1)
    ).clip(0, 1)

    pos_agg = pos_cash.groupby('SK_ID_CURR').agg(
        POS_COUNT=('SK_ID_PREV', 'count'),
        POS_MONTHS_BALANCE_MEAN=('MONTHS_BALANCE', 'mean'),
        POS_CNT_INSTALMENT_MEAN=('CNT_INSTALMENT', 'mean'),
        POS_CNT_INSTALMENT_FUTURE_MEAN=('CNT_INSTALMENT_FUTURE', 'mean'),
        POS_DPD_MEAN=('SK_DPD', 'mean'),
        POS_DPD_MAX=('SK_DPD', 'max'),
        POS_DPD_DEF_MEAN=('SK_DPD_DEF', 'mean'),
        POS_DPD_FLAG_SUM=('DPD_FLAG', 'sum'),       # Total months borrower was late
        POS_DPD_FLAG_RATE=('DPD_FLAG', 'mean'),     # % of months borrower was late
        POS_DPD_DEF_SUM=('DPD_DEF_FLAG', 'sum'),
        POS_COMPLETION_RATIO=('INSTALMENT_COMPLETION_RATIO', 'mean'),
    ).reset_index()

    # How many unique previous loans had ANY overdue months?
    dpd_by_loan = pos_cash.groupby('SK_ID_PREV')['DPD_FLAG'].max().reset_index()
    dpd_by_loan.columns = ['SK_ID_PREV', 'HAD_DPD']
    pos_with_dpd = pos_cash[['SK_ID_CURR', 'SK_ID_PREV']].drop_duplicates()
    pos_with_dpd = pos_with_dpd.merge(dpd_by_loan, on='SK_ID_PREV', how='left')
    loans_with_dpd = pos_with_dpd.groupby('SK_ID_CURR')['HAD_DPD'].agg(
        POS_LOANS_WITH_DPD='sum',
        POS_TOTAL_LOANS='count'
    ).reset_index()
    loans_with_dpd['POS_DPD_LOAN_RATE'] = (
        loans_with_dpd['POS_LOANS_WITH_DPD'] / (loans_with_dpd['POS_TOTAL_LOANS'] + 1)
    )
    pos_agg = pos_agg.merge(loans_with_dpd[['SK_ID_CURR', 'POS_LOANS_WITH_DPD',
                                              'POS_DPD_LOAN_RATE']],
                             on='SK_ID_CURR', how='left')

    print(f"  ✅ POS CASH features: {pos_agg.shape[1]-1} new features for "
          f"{pos_agg.shape[0]:,} applicants")
    return pos_agg


def engineer_credit_card_features(cc_bal):
    """Aggregate credit card behavior."""
    print("  Engineering credit card features...")

    cc_bal['CC_UTILIZATION'] = (
        cc_bal['AMT_BALANCE'] / (cc_bal['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
    ).clip(0, 1)

    cc_agg = cc_bal.groupby('SK_ID_CURR').agg(
        CC_COUNT=('SK_ID_PREV', 'count'),
        CC_BALANCE_MEAN=('AMT_BALANCE', 'mean'),
        CC_BALANCE_MAX=('AMT_BALANCE', 'max'),
        CC_UTILIZATION_MEAN=('CC_UTILIZATION', 'mean'),
        CC_UTILIZATION_MAX=('CC_UTILIZATION', 'max'),
        CC_DRAWINGS_MEAN=('AMT_DRAWINGS_CURRENT', 'mean'),
        CC_PAYMENT_RATE_MEAN=('AMT_PAYMENT_CURRENT', 'mean'),
        CC_DPD_MEAN=('SK_DPD', 'mean'),
        CC_DPD_DEF_MEAN=('SK_DPD_DEF', 'mean'),
    ).reset_index()

    print(f"  ✅ Credit card features: {cc_agg.shape[1]-1} new features")
    return cc_agg


def engineer_application_features(app):
    """Create derived features from the main application table."""
    print("  Engineering application-level features...")
    df = app.copy()

    # ── Core Financial Ratios ──────────────────────────────
    # Credit to Income ratio (like LTV for personal loans)
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)

    # Annuity (EMI) burden as % of income
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)

    # Loan to goods value ratio
    df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1)

    # Income per family member
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)

    # ── Time-Based Features ────────────────────────────────
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
    df['YEARS_EMPLOYED'] = -df['DAYS_EMPLOYED'].clip(upper=0) / 365
    df['EMPLOYMENT_TO_AGE_RATIO'] = df['YEARS_EMPLOYED'] / (df['AGE_YEARS'] + 1)
    df['YEARS_ID_PUBLISH'] = -df['DAYS_ID_PUBLISH'] / 365
    df['YEARS_LAST_PHONE_CHANGE'] = -df['DAYS_LAST_PHONE_CHANGE'] / 365

    # ── External Score Composite ───────────────────────────
    # EXT_SOURCE_1/2/3 are like credit score proxies (CIBIL equivalent)
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
    df['EXT_SOURCE_MIN'] = df[ext_cols].min(axis=1)
    df['EXT_SOURCE_MAX'] = df[ext_cols].max(axis=1)
    df['EXT_SOURCE_STD'] = df[ext_cols].std(axis=1)
    df['EXT_SOURCE_PRODUCT'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']

    # High EXT_SOURCE mean → lower default risk
    df['EXT_CREDIT_COMBINED'] = (
        df['EXT_SOURCE_MEAN'] * df['CREDIT_INCOME_RATIO']
    )

    # ── Document & Contact Completeness ───────────────────
    doc_cols = [c for c in df.columns if 'FLAG_DOCUMENT' in c]
    df['DOCUMENT_COUNT'] = df[doc_cols].sum(axis=1)

    contact_cols = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE',
                    'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']
    contact_cols_present = [c for c in contact_cols if c in df.columns]
    df['CONTACT_COUNT'] = df[contact_cols_present].sum(axis=1)

    # ── Risk Signal Features ───────────────────────────────
    # DAYS_EMPLOYED = 365243 means unemployed in this dataset
    df['IS_UNEMPLOYED'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['YEARS_EMPLOYED'] = np.where(
        df['IS_UNEMPLOYED'] == 1, 0, df['YEARS_EMPLOYED']
    )

    # High credit with low income → risk
    df['HIGH_CREDIT_LOW_INCOME'] = (
        (df['CREDIT_INCOME_RATIO'] > 5) &
        (df['AMT_INCOME_TOTAL'] < 100000)
    ).astype(int)

    # Young borrowers with high debt → higher risk historically
    df['YOUNG_HIGH_DEBT'] = (
        (df['AGE_YEARS'] < 30) &
        (df['ANNUITY_INCOME_RATIO'] > 0.3)
    ).astype(int)

    print(f"  ✅ Application features: added 20+ engineered features")
    return df


def build_master_dataset(app_train, bureau, bureau_bal, prev_app,
                          installments, cc_bal, pos_cash):
    """Merge all feature tables into one master dataset."""
    print("\nBuilding master dataset...")

    df = engineer_application_features(app_train)
    bureau_feats = engineer_bureau_features(bureau, bureau_bal)
    prev_feats = engineer_prev_app_features(prev_app)
    inst_feats = engineer_installment_features(installments)
    cc_feats = engineer_credit_card_features(cc_bal)
    pos_feats = engineer_pos_cash_features(pos_cash)

    df = df.merge(bureau_feats, on='SK_ID_CURR', how='left')
    df = df.merge(prev_feats, on='SK_ID_CURR', how='left')
    df = df.merge(inst_feats, on='SK_ID_CURR', how='left')
    df = df.merge(cc_feats, on='SK_ID_CURR', how='left')
    df = df.merge(pos_feats, on='SK_ID_CURR', how='left')

    print(f"\n✅ Master dataset shape: {df.shape}")
    print(f"   Total features: {df.shape[1]-2}")  # -2 for ID and TARGET
    return df


# ============================================================
# SECTION 4 — DATA CLEANING & PREPROCESSING
# ============================================================
print("\n" + "="*60)
print("SECTION 4: PREPROCESSING")
print("="*60)

def preprocess(df, target_col='TARGET', drop_missing_threshold=0.6):
    """Clean data and prepare for ML."""
    print("Preprocessing master dataset...")

    # Drop identifier
    drop_cols = ['SK_ID_CURR']

    # Drop high-missing columns
    missing_pct = df.isnull().mean()
    high_missing_cols = missing_pct[missing_pct > drop_missing_threshold].index.tolist()
    drop_cols += high_missing_cols
    print(f"  Dropping {len(high_missing_cols)} columns with >{drop_missing_threshold*100}% missing")

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode categoricals
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    print(f"  Encoding {len(cat_cols)} categorical columns...")
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].fillna('MISSING')
        df[col] = le.fit_transform(df[col].astype(str))

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"  Final feature matrix: {X.shape}")
    print(f"  Target: {y.value_counts().to_dict()}")

    return X, y, cat_cols


# ============================================================
# SECTION 5 — ENSEMBLE ML MODEL TRAINING
# ============================================================
print("\n" + "="*60)
print("SECTION 5: MODEL TRAINING (ENSEMBLE)")
print("="*60)

def train_ensemble_models(X_train, X_test, y_train, y_test):
    """
    Train 4 base models + stacking meta-model.
    Returns all models and their predictions.
    """
    results = {}

    # Impute missing values
    print("\nImputing missing values...")
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Scale for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    # ── Base Model 1: Logistic Regression ─────────────────
    print("\n[1/4] Training Logistic Regression (interpretable baseline)...")
    lr = LogisticRegression(
        C=0.1,
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict_proba(X_test_scaled)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_pred)
    results['LogisticRegression'] = {'model': lr, 'pred': lr_pred, 'auc': lr_auc}
    print(f"   ✅ LR AUC: {lr_auc:.4f}")

    # ── Base Model 2: Random Forest ────────────────────────
    print("\n[2/4] Training Random Forest (robustness)...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=50,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_imp, y_train)
    rf_pred = rf.predict_proba(X_test_imp)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_pred)
    results['RandomForest'] = {'model': rf, 'pred': rf_pred, 'auc': rf_auc}
    print(f"   ✅ RF AUC: {rf_auc:.4f}")

    # ── Base Model 3: XGBoost ──────────────────────────────
    print("\n[3/4] Training XGBoost (non-linear patterns)...")
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_model.fit(
        X_train_imp, y_train,
        eval_set=[(X_test_imp, y_test)],
        #early_stopping_rounds=50,
        verbose=False
    )
    xgb_pred = xgb_model.predict_proba(X_test_imp)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_pred)
    results['XGBoost'] = {'model': xgb_model, 'pred': xgb_pred, 'auc': xgb_auc}
    print(f"   ✅ XGB AUC: {xgb_auc:.4f}")

    # ── Base Model 4: LightGBM ─────────────────────────────
    print("\n[4/4] Training LightGBM (speed + performance)...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=7,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_train_imp, y_train,
        eval_set=[(X_test_imp, y_test)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    lgb_pred = lgb_model.predict_proba(X_test_imp)[:, 1]
    lgb_auc = roc_auc_score(y_test, lgb_pred)
    results['LightGBM'] = {'model': lgb_model, 'pred': lgb_pred, 'auc': lgb_auc}
    print(f"   ✅ LGB AUC: {lgb_auc:.4f}")

    # ── Stacking Meta-Model ────────────────────────────────
    print("\n[META] Training Stacking Ensemble...")
    # Stack base model predictions as features for meta-model
    stack_features_test = np.column_stack([
        lr_pred, rf_pred, xgb_pred, lgb_pred
    ])

    # For training meta-model — use cross-val OOF predictions
    print("  Generating Out-of-Fold predictions for stacking...")
    oof_preds = np.zeros((len(X_train_imp), 4))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_imp, y_train)):
        X_tr, X_val = X_train_imp[tr_idx], X_train_imp[val_idx]
        X_tr_s = X_train_scaled[tr_idx]
        X_val_s = X_train_scaled[val_idx]
        y_tr = y_train.iloc[tr_idx]

        # LR OOF
        lr_tmp = LogisticRegression(C=0.1, class_weight='balanced',
                                     max_iter=1000, random_state=42)
        lr_tmp.fit(X_tr_s, y_tr)
        oof_preds[val_idx, 0] = lr_tmp.predict_proba(X_val_s)[:, 1]

        # RF OOF
        rf_tmp = RandomForestClassifier(n_estimators=100, max_depth=8,
                                         class_weight='balanced', random_state=42)
        rf_tmp.fit(X_tr, y_tr)
        oof_preds[val_idx, 1] = rf_tmp.predict_proba(X_val)[:, 1]

        # XGB OOF
        xgb_tmp = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05,
                                      scale_pos_weight=scale_pos,
                                      use_label_encoder=False,
                                      eval_metric='auc', random_state=42, verbosity=0)
        xgb_tmp.fit(X_tr, y_tr)
        oof_preds[val_idx, 2] = xgb_tmp.predict_proba(X_val)[:, 1]

        # LGB OOF
        lgb_tmp = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                      class_weight='balanced', random_state=42, verbose=-1)
        lgb_tmp.fit(X_tr, y_tr)
        oof_preds[val_idx, 3] = lgb_tmp.predict_proba(X_val)[:, 1]

        print(f"    Fold {fold+1}/5 done")

    # Train meta-model on OOF predictions
    meta_model = LogisticRegression(C=1.0, random_state=42)
    meta_model.fit(oof_preds, y_train)

    # Final ensemble prediction
    ensemble_pred = meta_model.predict_proba(stack_features_test)[:, 1]
    ensemble_auc = roc_auc_score(y_test, ensemble_pred)
    results['Ensemble_Stack'] = {
        'model': meta_model,
        'pred': ensemble_pred,
        'auc': ensemble_auc
    }
    print(f"\n   🏆 ENSEMBLE STACKING AUC: {ensemble_auc:.4f}")

    # ── Weighted Average Ensemble (simpler alternative) ────
    # Give more weight to better-performing models
    aucs = [lr_auc, rf_auc, xgb_auc, lgb_auc]
    weights = np.array(aucs) / sum(aucs)
    weighted_pred = (
        weights[0] * lr_pred +
        weights[1] * rf_pred +
        weights[2] * xgb_pred +
        weights[3] * lgb_pred
    )
    weighted_auc = roc_auc_score(y_test, weighted_pred)
    results['Weighted_Average'] = {
        'pred': weighted_pred,
        'auc': weighted_auc,
        'weights': weights
    }
    print(f"   📊 Weighted Average AUC: {weighted_auc:.4f}")

    return results, imputer, scaler


def plot_model_comparison(results):
    """Bar chart comparing all model AUCs."""
    model_names = list(results.keys())
    aucs = [results[m]['auc'] for m in model_names]

    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#FFD700', '#E67E22']

    plt.figure(figsize=(10, 5))
    bars = plt.bar(model_names, aucs, color=colors[:len(model_names)],
                   edgecolor='black', linewidth=0.8)
    plt.ylim(0.5, 0.85)
    plt.axhline(0.7, color='red', linestyle='--', label='Good threshold (0.70)')
    plt.axhline(0.75, color='green', linestyle='--', label='Strong threshold (0.75)')

    for bar, auc in zip(bars, aucs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{auc:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.title('Model AUC-ROC Comparison\n(Higher = Better Credit Decisioning)',
              fontsize=13)
    plt.ylabel('AUC-ROC Score')
    plt.xticks(rotation=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Model comparison plot saved")


# ============================================================
# SECTION 6 — SHAP EXPLAINABILITY
# ============================================================
print("\n" + "="*60)
print("SECTION 6: SHAP EXPLAINABILITY")
print("="*60)

def run_shap_analysis(lgb_model, X_test_imp, feature_names, n_samples=500):
    """
    SHAP values tell us WHY the model made each decision.
    Essential for credit committee review and regulatory compliance.
    """
    print("Computing SHAP values (this may take 1-2 minutes)...")

    # Use a sample for speed
    X_sample = X_test_imp[:n_samples]

    explainer = shap.TreeExplainer(lgb_model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification LightGBM returns list [neg_class, pos_class]
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # Default class (1)
    else:
        shap_vals = shap_values

    # ── Plot 1: Global Feature Importance ─────────────────
    plt.figure(figsize=(5, 4))
    shap.summary_plot(shap_vals, X_sample,
                      feature_names=feature_names,
                      plot_type="bar",
                      show=False,
                      max_display=20)
    plt.title("Top 20 Features — Global Credit Risk Drivers", fontsize=13)
    plt.tight_layout()
    plt.savefig('shap_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ── Plot 2: SHAP Beeswarm (impact direction) ───────────
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_vals, X_sample,
                      feature_names=feature_names,
                      show=False,
                      max_display=15)
    plt.title("SHAP Feature Impact — Direction & Magnitude", fontsize=13)
    plt.tight_layout()
    plt.savefig('shap_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("✅ SHAP plots saved")
    return explainer, shap_vals


def explain_single_prediction(explainer, X_single, feature_names, borrower_name="Borrower"):
    """
    Explain ONE borrower's credit decision in plain English.
    This is what goes into the CAM's explainability section.
    """
    shap_vals = explainer.shap_values(X_single.reshape(1, -1))
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]
    else:
        sv = shap_vals[0]

    # Get top risk drivers
    feature_shap = pd.DataFrame({
        'feature': feature_names,
        'shap_value': sv,
        'abs_shap': np.abs(sv)
    }).sort_values('abs_shap', ascending=False).head(10)

    print(f"\n{'='*55}")
    print(f"  CREDIT DECISION EXPLANATION — {borrower_name}")
    print(f"{'='*55}")
    print("\nTop 10 factors that influenced this credit decision:\n")

    explanations = []
    for _, row in feature_shap.iterrows():
        direction = "⬆️ INCREASES" if row['shap_value'] > 0 else "⬇️ DECREASES"
        impact = "HIGH" if row['abs_shap'] > 0.1 else "MEDIUM" if row['abs_shap'] > 0.05 else "LOW"
        explanation = f"  {direction} default risk [{impact}]: {row['feature']}"
        print(explanation)
        explanations.append({
            'feature': row['feature'],
            'direction': 'risk_increase' if row['shap_value'] > 0 else 'risk_decrease',
            'impact': impact,
            'shap_value': round(float(row['shap_value']), 4)
        })

    return explanations


# ============================================================
# SECTION 7 — PD / LGD / EAD + RISK RATING
# ============================================================
print("\n" + "="*60)
print("SECTION 7: PD / LGD / EAD COMPUTATION")
print("="*60)

def compute_expected_loss(pd_score, loan_amount, collateral_value=0):
    """
    Compute the 3 core credit risk metrics used by all banks:

    PD  = Probability of Default (from ML model, 0-1)
    LGD = Loss Given Default (% of loan lost if default happens)
    EAD = Exposure at Default (total amount at risk)
    EL  = Expected Loss = PD × LGD × EAD
    """
    # ── PD: Direct from ML model output ───────────────────
    pd_value = pd_score  # Already a probability (0 to 1)

    # ── LGD: Estimated from collateral coverage ────────────
    # If collateral covers 80%+ of loan → LGD is low (20%)
    # If no collateral → LGD is high (75-90%)
    if collateral_value > 0:
        collateral_coverage = collateral_value / (loan_amount + 1)
        lgd = max(0.10, 1 - (collateral_coverage * 0.8))  # Floor at 10%
        lgd = min(lgd, 0.90)  # Cap at 90%
    else:
        lgd = 0.75  # Unsecured loan → 75% LGD (industry standard)

    # ── EAD: Full loan exposure ────────────────────────────
    ead = loan_amount  # For term loans, EAD = full amount
    # For revolving credit: EAD = drawn amount + 75% of undrawn

    # ── Expected Loss ─────────────────────────────────────
    el = pd_value * lgd * ead

    return {
        'PD': round(pd_value, 4),
        'LGD': round(lgd, 4),
        'EAD': round(ead, 2),
        'Expected_Loss': round(el, 2),
        'Expected_Loss_Pct': round((el / ead) * 100, 2)
    }


def assign_risk_rating(pd_score):
    """
    Map PD score to internal risk rating (1=Best, 8=Worst).
    Based on standard bank risk rating scales.
    """
    rating_map = [
        (0.005, 1, "AAA", "Exceptional"),
        (0.01,  2, "AA",  "Excellent"),
        (0.02,  3, "A",   "Good"),
        (0.05,  4, "BBB", "Satisfactory"),
        (0.10,  5, "BB",  "Acceptable"),
        (0.20,  6, "B",   "Watch"),
        (0.35,  7, "CCC", "Substandard"),
        (1.00,  8, "D",   "Default/Loss")
    ]
    for threshold, rating, grade, label in rating_map:
        if pd_score <= threshold:
            return {'rating': rating, 'grade': grade, 'label': label}
    return {'rating': 8, 'grade': 'D', 'label': 'Default/Loss'}


def compute_risk_premium(pd_score, lgd, risk_free_rate=0.065):
    """
    Risk-Based Pricing: what interest rate should this borrower pay?

    Formula: Rate = Risk-Free Rate + Credit Spread
    Credit Spread compensates the lender for expected loss
    """
    expected_loss_rate = pd_score * lgd          # Annual EL as % of loan
    capital_charge = pd_score * lgd * 8          # Basel II simplified capital
    operating_cost = 0.015                        # 1.5% operational cost
    profit_margin = 0.01                          # 1% target profit

    total_spread = expected_loss_rate + capital_charge * 0.12 + operating_cost + profit_margin
    final_rate = risk_free_rate + total_spread

    return {
        'risk_free_rate': f"{risk_free_rate*100:.2f}%",
        'credit_spread': f"{total_spread*100:.2f}%",
        'recommended_rate': f"{final_rate*100:.2f}%",
        'rate_band': f"{(final_rate-0.005)*100:.2f}% - {(final_rate+0.01)*100:.2f}%"
    }


def make_lending_decision(pd_score, loan_amount, annual_income,
                           collateral_value=0, borrower_name="Applicant"):
    """
    Final credit decision combining all metrics.
    This is the output of Layer 5 that feeds into the CAM.
    """
    el_metrics = compute_expected_loss(pd_score, loan_amount, collateral_value)
    rating = assign_risk_rating(pd_score)
    pricing = compute_risk_premium(pd_score, el_metrics['LGD'])

    # Decision logic
    dscr_proxy = annual_income / (loan_amount * 0.12 + 1)  # Simplified DSCR

    if pd_score < 0.05 and dscr_proxy > 1.5:
        decision = "APPROVE"
        decision_color = "🟢"
        max_limit = min(loan_amount, annual_income * 6)
    elif pd_score < 0.15 and dscr_proxy > 1.2:
        decision = "CONDITIONAL APPROVE"
        decision_color = "🟡"
        max_limit = min(loan_amount * 0.75, annual_income * 4)
    elif pd_score < 0.25:
        decision = "REFER TO SENIOR CREDIT"
        decision_color = "🟠"
        max_limit = min(loan_amount * 0.5, annual_income * 3)
    else:
        decision = "REJECT"
        decision_color = "🔴"
        max_limit = 0

    print(f"\n{'='*60}")
    print(f"  CREDIT DECISION MEMO — {borrower_name}")
    print(f"{'='*60}")
    print(f"  Decision:          {decision_color} {decision}")
    print(f"  Risk Rating:       {rating['rating']} ({rating['grade']}) — {rating['label']}")
    print(f"  PD Score:          {pd_score*100:.2f}%")
    print(f"  LGD:               {el_metrics['LGD']*100:.1f}%")
    print(f"  Expected Loss:     ₹{el_metrics['Expected_Loss']:,.0f} "
          f"({el_metrics['Expected_Loss_Pct']:.2f}% of loan)")
    print(f"  Recommended Limit: ₹{max_limit:,.0f}")
    print(f"  Pricing:           {pricing['recommended_rate']} "
          f"(Band: {pricing['rate_band']})")
    print(f"{'='*60}")

    return {
        'borrower': borrower_name,
        'decision': decision,
        'risk_rating': rating,
        'pd_score': pd_score,
        'metrics': el_metrics,
        'pricing': pricing,
        'credit_limit': max_limit,
        'timestamp': datetime.now().isoformat()
    }


# ============================================================
# SECTION 8 — STRESS TESTING ENGINE
# ============================================================
print("\n" + "="*60)
print("SECTION 8: STRESS TESTING")
print("="*60)

def run_stress_tests(base_pd, base_income, loan_amount,
                     base_ebitda=None, interest_rate=0.12):
    """
    Test how the borrower's creditworthiness changes under stress.
    4 standard scenarios used by RBI-regulated banks.
    """
    scenarios = {
        "Base Case": {
            "income_shock": 1.0,
            "rate_shock": 0.0,
            "cost_shock": 1.0
        },
        "Revenue Drop 20%": {
            "income_shock": 0.80,
            "rate_shock": 0.0,
            "cost_shock": 1.0
        },
        "Interest Rate +200bps": {
            "income_shock": 1.0,
            "rate_shock": 0.02,
            "cost_shock": 1.0
        },
        "Cost Spike 15%": {
            "income_shock": 1.0,
            "rate_shock": 0.0,
            "cost_shock": 1.15
        },
        "Combined Stress": {
            "income_shock": 0.80,
            "rate_shock": 0.02,
            "cost_shock": 1.10
        },
    }

    results = []
    print(f"\n{'Scenario':<25} {'Income':>10} {'EMI':>10} {'DSCR':>8} {'PD':>8} {'Decision':>12}")
    print("-" * 80)

    for name, params in scenarios.items():
        stressed_income = base_income * params['income_shock']
        stressed_rate = interest_rate + params['rate_shock']
        stressed_emi = loan_amount * stressed_rate / 12

        # Simplified DSCR under stress
        annual_debt_service = stressed_emi * 12
        dscr = stressed_income / (annual_debt_service + 1)

        # Increase PD under stress (simplified — in production use retrained model)
        income_pct_drop = 1 - params['income_shock']
        rate_increase_impact = params['rate_shock'] * 5
        pd_stressed = min(base_pd * (1 + income_pct_drop * 2 + rate_increase_impact), 0.99)

        if dscr >= 1.5 and pd_stressed < 0.10:
            decision = "✅ PASS"
        elif dscr >= 1.2 and pd_stressed < 0.20:
            decision = "⚠️ WATCH"
        else:
            decision = "❌ FAIL"

        results.append({
            'Scenario': name,
            'Income': stressed_income,
            'Annual_EMI': annual_debt_service,
            'DSCR': round(dscr, 2),
            'PD_Stressed': round(pd_stressed, 4),
            'Decision': decision
        })

        print(f"{name:<25} {stressed_income:>10,.0f} {annual_debt_service:>10,.0f} "
              f"{dscr:>8.2f} {pd_stressed*100:>7.2f}% {decision:>12}")

    # Visualize
    df_stress = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ['green' if d == '✅ PASS' else 'orange' if d == '⚠️ WATCH' else 'red'
              for d in df_stress['Decision']]

    df_stress.plot(x='Scenario', y='DSCR', kind='bar', ax=axes[0],
                   color=colors, legend=False)
    axes[0].axhline(1.5, color='green', linestyle='--', label='Strong (1.5x)')
    axes[0].axhline(1.2, color='orange', linestyle='--', label='Minimum (1.2x)')
    axes[0].set_title('DSCR Under Stress Scenarios')
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis='x', rotation=25)

    df_stress.plot(x='Scenario', y='PD_Stressed', kind='bar', ax=axes[1],
                   color=colors, legend=False)
    axes[1].axhline(0.10, color='orange', linestyle='--', label='10% PD threshold')
    axes[1].set_title('Probability of Default Under Stress')
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis='x', rotation=25)

    df_stress.plot(x='Scenario', y='Income', kind='bar', ax=axes[2],
                   color=colors, legend=False)
    axes[2].set_title('Income Under Stress Scenarios')
    axes[2].tick_params(axis='x', rotation=25)

    plt.suptitle('Stress Testing Results — Credit Resilience Analysis',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('stress_test_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Stress test visualization saved")

    return df_stress


# ============================================================
# SECTION 9 — SAVE MODELS + GENERATE REPORT
# ============================================================

def save_models(results, imputer, scaler, feature_names):
    """Save trained models for use in the production pipeline."""
    print("\nSaving models...")

    joblib.dump(results['LightGBM']['model'], 'model_lgb.pkl')
    joblib.dump(results['XGBoost']['model'], 'model_xgb.pkl')
    joblib.dump(results['RandomForest']['model'], 'model_rf.pkl')
    joblib.dump(results['LogisticRegression']['model'], 'model_lr.pkl')
    joblib.dump(imputer, 'imputer.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    with open('feature_names.json', 'w') as f:
        json.dump(feature_names, f)

    # Save performance summary
    summary = {
        'run_timestamp': datetime.now().isoformat(),
        'model_aucs': {k: round(v['auc'], 4) for k, v in results.items()},
        'best_model': max(results, key=lambda k: results[k]['auc']),
        'feature_count': len(feature_names)
    }
    with open('model_performance_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("✅ Models saved: model_lgb.pkl, model_xgb.pkl, model_rf.pkl, model_lr.pkl")
    print("✅ Performance summary saved: model_performance_summary.json")
    return summary


# ============================================================
# SECTION 10 — FULL PIPELINE RUNNER
# ============================================================

def run_full_pipeline(data_path="./"):
    """
    Master function — runs the entire ML pipeline end to end.
    Call this with the folder path to your CSV files.
    """
    print("\n" + "🚀 "*20)
    print("CREDIT DECISIONING ENGINE — FULL ML PIPELINE")
    print("🚀 "*20 + "\n")

    # ── Step 1: Load ──────────────────────────────────────
    app_train, bureau, bureau_bal, prev_app, installments, cc_bal, pos_cash = \
        load_data(data_path)

    # ── Step 2: EDA ───────────────────────────────────────
    missing_pct = run_eda(app_train)

    # ── Step 3: Feature Engineering ───────────────────────
    master_df = build_master_dataset(
        app_train, bureau, bureau_bal, prev_app, installments, cc_bal, pos_cash
    )

    # ── Step 4: Preprocessing ─────────────────────────────
    X, y, cat_cols = preprocess(master_df)
    feature_names = X.columns.tolist()

    # Train/test split — stratified to maintain class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train set: {X_train.shape[0]:,} | Test set: {X_test.shape[0]:,}")

    # ── Step 5: Train Ensemble ────────────────────────────
    results, imputer, scaler = train_ensemble_models(
        X_train.values, X_test.values, y_train, y_test
    )
    plot_model_comparison(results)

    # ── Step 6: SHAP Explainability ───────────────────────
    X_test_imp = imputer.transform(X_test.values)
    explainer, shap_vals = run_shap_analysis(
        results['LightGBM']['model'], X_test_imp, feature_names
    )

    # Explain one sample decision
    explain_single_prediction(
        explainer, X_test_imp[0], feature_names, "Sample Borrower #1"
    )

    # ── Step 7: Demo Credit Decision ──────────────────────
    sample_pd = float(results['LightGBM']['pred'][0])
    demo_decision = make_lending_decision(
        pd_score=sample_pd,
        loan_amount=500000,
        annual_income=800000,
        collateral_value=600000,
        borrower_name="Demo Corp Pvt Ltd"
    )

    # ── Step 8: Stress Testing ────────────────────────────
    stress_df = run_stress_tests(
        base_pd=sample_pd,
        base_income=800000,
        loan_amount=500000
    )

    # ── Step 9: Save ──────────────────────────────────────
    summary = save_models(results, imputer, scaler, feature_names)

    print("\n" + "✅ "*20)
    print("PIPELINE COMPLETE!")
    print(f"Best Model: {summary['best_model']} | AUC: {summary['model_aucs'][summary['best_model']]}")
    print("✅ "*20)

    return results, explainer, demo_decision, stress_df


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # ── FOR GOOGLE COLAB ──────────────────────────────────
    # 1. Upload all 10 CSV files to Colab
    # 2. Set DATA_PATH to where you uploaded them:
    #    DATA_PATH = "/content/"   ← if uploaded directly
    #    DATA_PATH = "/content/drive/MyDrive/homecredit/"  ← if using Google Drive

    DATA_PATH = "/content/drive/MyDrive/credit"   # Change this to your path

    results, explainer, decision, stress_df = run_full_pipeline(DATA_PATH)