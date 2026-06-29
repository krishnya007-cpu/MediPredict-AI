# ============================================================
# train_models.py — MEDIPREDICT AI  |  Improved Training
# Usage: python train_models.py
#
# Split: 60% train / 20% validation / 20% test  ← FIXED
# All models saved with joblib (version-safe)
# ============================================================

import numpy as np
import pandas as pd
import joblib, os
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    classification_report, mean_absolute_error,
    r2_score, roc_auc_score, mean_squared_error
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT_DIR = "./models"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
    'age', 'gender', 'region', 'marital_status', 'num_dependents',
    'bmi', 'smoker', 'chronic_disease', 'prev_hospitalizations',
    'num_medications', 'exercise_frequency', 'coverage_type',
    'policy_duration', 'deductible', 'num_prev_claims',
    'annual_income', 'credit_score'
]
CAT_FEATURES = ['gender', 'region', 'marital_status', 'coverage_type',
                'smoker', 'exercise_frequency']
NUM_FEATURES = [f for f in FEATURES if f not in CAT_FEATURES]


# ── 60 / 20 / 20 split ────────────────────────────────────
def split_data(X, y, stratify=None):
    """
    WHY 60/20/20:
      60% train  - enough signal for the model to learn
      20% val    - used for Optuna tuning & early stopping
      20% test   - completely held-out; gives unbiased final metrics

    A single 80/20 split leaks val info into hyperparameter choices,
    making reported accuracy optimistically inflated.
    """
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.20,
        stratify=y if stratify else None, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25,          # 25% of 80% = 20% overall
        stratify=y_tmp if stratify else None, random_state=42)
    print(f"  Split → Train {len(X_train):,} (60%)  "
          f"Val {len(X_val):,} (20%)  Test {len(X_test):,} (20%)")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Synthetic data (replace with pd.read_csv) ─────────────
def generate_data(n=10_000, seed=42):
    rng = np.random.RandomState(seed)
    age=rng.randint(18,80,n); gender=rng.choice(['Male','Female'],n)
    region=rng.choice(['North','South','East','West'],n)
    marital=rng.choice(['Single','Married','Divorced','Widowed'],n)
    ndep=rng.randint(0,6,n); bmi=rng.uniform(16,45,n).round(1)
    smoker=rng.choice(['Yes','No'],n,p=[0.2,0.8])
    # Chronic disease increases with age and smoking
    chronic = (
        (age > 60).astype(int)
        + (bmi > 30).astype(int)
        + (smoker == "Yes").astype(int)
        + rng.binomial(1, 0.15, n)
    )
    chronic = np.clip(chronic, 0, 3)

    # Hospitalisations depend on chronic disease
    prev_hosp = np.clip(
        rng.poisson(0.5 + chronic * 0.8),
        0,
        6
    )

    # Medication count depends on chronic disease
    nmeds = np.clip(
        rng.poisson(1 + chronic * 2),
        0,
        10
    )
    exfreq=rng.choice(['Never','Rarely','Sometimes','Often','Daily'],n)
    ctype=rng.choice(['Basic','Standard','Premium'],n)
    pdur=rng.randint(1,20,n); ded=rng.choice([500,1000,2000,5000],n)
    npc=rng.randint(0,10,n); inc=(25000 + age * 1200 + rng.normal(0, 25000, n)).clip(20000, 300000).astype(int)
    cscore= (700 - npc * 20 - prev_hosp * 10 + rng.normal(0, 40, n)).clip(300, 850).astype(int)

    df=pd.DataFrame({'age':age,'gender':gender,'region':region,
        'marital_status':marital,'num_dependents':ndep,'bmi':bmi,
        'smoker':smoker,'chronic_disease':chronic,
        'prev_hospitalizations':prev_hosp,'num_medications':nmeds,
        'exercise_frequency':exfreq,'coverage_type':ctype,
        'policy_duration':pdur,'deductible':ded,
        'num_prev_claims':npc,'annual_income':inc,'credit_score':cscore})

    rs=(((age>55)*2)+((bmi>30)*1.5)+((smoker=='Yes')*3)+
        chronic*1.2+prev_hosp*0.8+((cscore<500)*1)+rng.normal(0,.5,n))
    df['risk_label']=pd.cut(rs,bins=[-np.inf,2,5,np.inf],labels=[0,1,2]).astype(int)
    df['premium']=(500+age*30+(smoker=='Yes').astype(int)*3000+bmi*50+
        chronic*400+npc*200-(cscore-300)*1.5+
        (ctype=='Premium').astype(int)*2000+rng.normal(0,300,n)).clip(500,30000).round(2)
    # -------- Improved Fraud Generation --------
    fraud_score = (
        (npc >= 4) * 2.5 +
        (pdur <= 2) * 2.0 +
        (prev_hosp >= 4) * 1.5 +
        (chronic == 0) * 1.0 +
        (cscore < 500) * 2.0 +
        (inc < 50000) * 1.5 +
        (age < 25) * 1.2 +
        rng.normal(0, 0.5, n)
    )

    fraud_prob = 1 / (1 + np.exp(-(fraud_score - 4)))

    df["fraud"] = (rng.rand(n) < fraud_prob).astype(int)
    base = (
        500
        + prev_hosp * 2500
        + chronic * 1800
        + nmeds * 400
        + (smoker == "Yes").astype(int) * 2500
        + (bmi > 30).astype(int) * 1200
    )

    noise = rng.normal(0, 1200, n)

    df["claim_amount"] = (
        base + noise
    ).clip(0, 100000).round(2)
    return df


from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder

def build_encoders(df):

    # Categorical columns
    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    ordinal_encoder.fit(df[CAT_FEATURES])

    # Risk label encoder
    risk_label_encoder = LabelEncoder()
    risk_label_encoder.fit(df["risk_label"])

    # Scale numerical columns
    scaler = StandardScaler()
    scaler.fit(df[NUM_FEATURES])

    encoders = {
        "ordinal_encoder": ordinal_encoder,
        "risk_label_encoder": risk_label_encoder,
        "scaler": scaler,
        "cat_feature_cols": CAT_FEATURES,
        "scale_cols": NUM_FEATURES,
    }

    return encoders

def encode(df, encoders):
    df = df.copy()

    # Encode categorical features
    df[CAT_FEATURES] = encoders["ordinal_encoder"].transform(df[CAT_FEATURES])

    # Scale numerical features
    df[NUM_FEATURES] = encoders["scaler"].transform(df[NUM_FEATURES])

    return df


# ── 1. RISK ────────────────────────────────────────────────
def train_risk_model(Xtr,Xv,Xte,ytr,yv,yte):
    print("\n[1/4] RISK model (RandomForestClassifier) ...")
    Xall=np.vstack([Xtr,Xv]); yall=np.concatenate([ytr,yv])
    def obj(trial):
        p=dict(n_estimators=trial.suggest_int('n_estimators',100,500),
               max_depth=trial.suggest_int('max_depth',4,20),
               min_samples_split=trial.suggest_int('min_samples_split',2,20),
               min_samples_leaf=trial.suggest_int('min_samples_leaf',1,10),
               max_features=trial.suggest_categorical('max_features',['sqrt','log2']),
               class_weight='balanced',random_state=42,n_jobs=-1)
        return cross_val_score(RandomForestClassifier(**p),Xall,yall,
            cv=StratifiedKFold(5),scoring='f1_weighted',n_jobs=-1).mean()
    s=optuna.create_study(direction='maximize'); s.optimize(obj,n_trials=30,show_progress_bar=False)
    m=RandomForestClassifier(**{**s.best_params,'class_weight':'balanced','random_state':42,'n_jobs':-1})
    m.fit(Xall,yall)
    print(f"  Best: {s.best_params}")
    print(f"  TEST:\n{classification_report(yte,m.predict(Xte),target_names=['Low','Medium','High'])}")
    return m


# ── 2. PREMIUM ─────────────────────────────────────────────
def train_premium_model(Xtr,Xv,Xte,ytr,yv,yte):
    print("\n[2/4] PREMIUM model (XGBRegressor + log transform) ...")
    ytr_l,yv_l=np.log1p(ytr),np.log1p(yv)
    def obj(trial):
        p=dict(n_estimators=trial.suggest_int('n_estimators',100,600),
               max_depth=trial.suggest_int('max_depth',3,10),
               learning_rate=trial.suggest_float('learning_rate',.01,.3,log=True),
               subsample=trial.suggest_float('subsample',.6,1.),
               colsample_bytree=trial.suggest_float('colsample_bytree',.6,1.),
               reg_alpha=trial.suggest_float('reg_alpha',1e-4,10,log=True),
               reg_lambda=trial.suggest_float('reg_lambda',1e-4,10,log=True),
               min_child_weight=trial.suggest_int('min_child_weight',1,10),
               gamma=trial.suggest_float('gamma',0,5),
               objective='reg:squarederror',tree_method='hist',random_state=42,n_jobs=-1)
        m=xgb.XGBRegressor(**p)
        m.fit(
            Xtr,
            ytr_l,
            eval_set=[(Xv, yv_l)],
            verbose=False
        )
        return mean_absolute_error(yv_l,m.predict(Xv))
    s=optuna.create_study(direction='minimize'); s.optimize(obj,n_trials=40,show_progress_bar=False)
    m=xgb.XGBRegressor(**{**s.best_params,'objective':'reg:squarederror',
        'tree_method':'hist','random_state':42,'n_jobs':-1})
    m.fit(
        Xtr,
        ytr_l,
        eval_set=[(Xv, yv_l)],
        verbose=False
    )
    pred=np.expm1(m.predict(Xte))
    print(f"  Best: {s.best_params}")
    print(f"  TEST MAE:${mean_absolute_error(yte,pred):,.0f}  R²:{r2_score(yte,pred):.4f}")
    return m


# ── 3. FRAUD ───────────────────────────────────────────────
def train_fraud_model(Xtr,Xv,Xte,ytr,yv,yte):
    print("\n[3/4] FRAUD model (XGBClassifier + SMOTE) ...")
    print(f"  Before SMOTE: {np.bincount(ytr)}")
    Xr,yr=SMOTE(random_state=42).fit_resample(Xtr,ytr)  # SMOTE on train only
    print(f"  After  SMOTE: {np.bincount(yr)}")
    ratio=(ytr==0).sum()/max((ytr==1).sum(),1)
    def obj(trial):
        p=dict(n_estimators=trial.suggest_int('n_estimators',100,500),
               max_depth=trial.suggest_int('max_depth',3,8),
               learning_rate=trial.suggest_float('learning_rate',.01,.3,log=True),
               subsample=trial.suggest_float('subsample',.6,1.),
               colsample_bytree=trial.suggest_float('colsample_bytree',.6,1.),
               reg_alpha=trial.suggest_float('reg_alpha',1e-4,10,log=True),
               reg_lambda=trial.suggest_float('reg_lambda',1e-4,10,log=True),
               min_child_weight=trial.suggest_int('min_child_weight',1,10),
               gamma=trial.suggest_float('gamma',0,5),
               scale_pos_weight=ratio,objective='binary:logistic',
               eval_metric='auc',tree_method='hist',random_state=42,n_jobs=-1)
        m=xgb.XGBClassifier(**p)
        m.fit(
            Xr,
            yr,
            eval_set=[(Xv, yv)],
            verbose=False
        )
        return roc_auc_score(yv,m.predict_proba(Xv)[:,1])
    s=optuna.create_study(direction='maximize'); s.optimize(obj,n_trials=40,show_progress_bar=False)
    m=xgb.XGBClassifier(**{**s.best_params,'scale_pos_weight':ratio,
        'objective':'binary:logistic','eval_metric':'auc',
        'tree_method':'hist','random_state':42,'n_jobs':-1})
    m.fit(
        Xr,
        yr,
        eval_set=[(Xv, yv)],
        verbose=False
    )
    prob=m.predict_proba(Xte)[:,1]
    print(f"  Best: {s.best_params}")
    print(f"  TEST AUC:{roc_auc_score(yte,prob):.4f}")
    print(f"  TEST:\n{classification_report(yte,m.predict(Xte),target_names=['Legit','Fraud'])}")
    return m


# ── 4. CLAIM ───────────────────────────────────────────────
def train_claim_model(Xtr,Xv,Xte,ytr,yv,yte):
    print("\n[4/4] CLAIM model (XGBRegressor + log transform) ...")
    ytr_l,yv_l=np.log1p(ytr),np.log1p(yv)
    def obj(trial):
        p=dict(n_estimators=trial.suggest_int('n_estimators',100,600),
               max_depth=trial.suggest_int('max_depth',3,10),
               learning_rate=trial.suggest_float('learning_rate',.01,.3,log=True),
               subsample=trial.suggest_float('subsample',.6,1.),
               colsample_bytree=trial.suggest_float('colsample_bytree',.6,1.),
               reg_alpha=trial.suggest_float('reg_alpha',1e-4,10,log=True),
               reg_lambda=trial.suggest_float('reg_lambda',1e-4,10,log=True),
               min_child_weight=trial.suggest_int('min_child_weight',1,10),
               gamma=trial.suggest_float('gamma',0,5),
               objective='reg:squarederror',tree_method='hist',random_state=42,n_jobs=-1)
        m=xgb.XGBRegressor(**p)
        m.fit(
            Xtr,
            ytr_l,
            eval_set=[(Xv, yv_l)],
            verbose=False
        )
        return mean_absolute_error(yv_l,m.predict(Xv))
    s=optuna.create_study(direction='minimize'); s.optimize(obj,n_trials=40,show_progress_bar=False)
    m=xgb.XGBRegressor(**{**s.best_params,'objective':'reg:squarederror',
        'tree_method':'hist','random_state':42,'n_jobs':-1})
    m.fit(
        Xtr,
        ytr_l,
        eval_set=[(Xv, yv_l)],
        verbose=False
    )
    pred=np.expm1(m.predict(Xte))
    print(f"  Best: {s.best_params}")
    print(f"  TEST MAE:${mean_absolute_error(yte,pred):,.0f}  R²:{r2_score(yte,pred):.4f}")
    return m


# ── Main ───────────────────────────────────────────────────
def main():
    print("="*60)
    print("  MEDIPREDICT AI — Training Pipeline  (60 / 20 / 20)")
    print("="*60)

    df=generate_data(n=10_000)     # ← swap to pd.read_csv("data.csv")
    print(f"\nData shape: {df.shape}")
    encoders=build_encoders(df)
    df_enc=encode(df,encoders)
    X=df_enc[FEATURES].values

    models={}

    y=df_enc['risk_label'].values
    args=split_data(X,y,stratify=True)
    models['risk']=train_risk_model(*args)

    y=df_enc['premium'].values
    args=split_data(X,y)
    models['premium']=train_premium_model(*args)

    y=df_enc['fraud'].values
    args=split_data(X,y,stratify=True)
    models['fraud']=train_fraud_model(*args)

    y=df_enc['claim_amount'].values
    args=split_data(X,y)
    models['claim']=train_claim_model(*args)

    print("\nSaving ...")
    saves={
        'risk_model.joblib':    models['risk'],
        'premium_model.joblib': models['premium'],
        'fraud_model.joblib':   models['fraud'],
        'claim_model.joblib':   models['claim'],
        'encoders.joblib':      encoders,
        'feature_meta.joblib':  {'features':FEATURES,'cat':CAT_FEATURES,'num':NUM_FEATURES},
    }
    for fname,obj in saves.items():
        p=os.path.join(OUT_DIR,fname)
        joblib.dump(obj,p,compress=3)
        print(f"  ✓ {fname:<28} ({os.path.getsize(p)/1024:,.0f} KB)")

    print(f"\n✅ All models saved to {os.path.abspath(OUT_DIR)}/")
    print("""
HOW TO LOAD IN YOUR APP:
  import joblib, numpy as np
  risk_model    = joblib.load('models/risk_model.joblib')
  premium_model = joblib.load('models/premium_model.joblib')
  fraud_model   = joblib.load('models/fraud_model.joblib')
  claim_model   = joblib.load('models/claim_model.joblib')
  encoders      = joblib.load('models/encoders.joblib')

  # Decode regressors:
  premium = float(np.expm1(premium_model.predict(X)))
  claim   = float(np.expm1(claim_model.predict(X)))

  # Risk label:
  label = ['Low','Medium','High'][risk_model.predict(X)[0]]

  # Fraud probability:
  fraud_prob = fraud_model.predict_proba(X)[0][1]
""")

if __name__=="__main__":
    main()
