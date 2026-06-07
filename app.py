import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Loan Risk Predictor", layout="wide")

# load model
model = joblib.load("xgb_pipeline.pkl")

try:
    cols = model.named_steps["model"].get_booster().feature_names
except Exception:
    try:
        cols = model.named_steps["model"].feature_names_in_
    except Exception:
        try:
            cols = model.feature_names_in_
        except Exception:
            cols = None

if not cols:
    cols = [
        'loan_amnt','term','int_rate','installment','emp_length','dti',
        'earliest_cr_line','open_acc','pub_rec','revol_util','total_acc','mort_acc',
        'pub_rec_bankruptcies','log_annual_inc','fico_score','log_revol_bal',
        *[f"sub_grade_{a}{b}" for a in "ABCDEFG" for b in range(1,6)],
        *[f"home_ownership_{x}" for x in ["OWN","RENT"]],
        *[f"verification_status_{x}" for x in ["Source Verified","Verified"]],
        *[f"purpose_{x}" for x in [
            "credit_card","debt_consolidation","educational","home_improvement",
            "house","major_purchase","medical","moving","other",
            "renewable_energy","small_business","vacation","wedding"]],
        *[f"addr_state_{s}" for s in [
            "AL","AR","AZ","CA","CO","CT","DC","DE","FL","GA","HI","IA","ID","IL","IN",
            "KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH",
            "NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VA",
            "VT","WA","WI","WV","WY"]],
        'initial_list_status_w','application_type_Joint App'
    ]

# ui inputs
st.title("Loan Risk Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    loan_amnt = st.number_input(
        "Loan Amount", 0.0, 1e7, 10000.0,
        help="Definition: The total amount of money requested by the borrower."
    )
    term = st.selectbox(
        "Term", [36, 60],
        help="Definition: The number of months over which the loan will be repaid."
    )
    int_rate = st.number_input(
        "Interest Rate (%)", 0.0, 100.0, 7.96,
        help="Definition: The interest rate charged on the loan."
    )
    installment = st.number_input(
        "Installment", 0.0, 10000.0, 313.18,
        help="Definition: The monthly payment owed by the borrower if the loan is funded."
    )
    emp_length = st.number_input(
        "Employment Length (years)", 0.0, 60.0, 5.0,
        help="Definition: The number of years the borrower has been employed."
    )
    dti = st.number_input(
        "Debt-to-Income Ratio", 0.0, 100.0, 29.57,
        help="Definition: The borrower’s total monthly debt payments divided by monthly income."
    )

with col2:
    earliest_cr_line = st.number_input(
        "Earliest Credit Line (year)", 1950, 2025, 2006,
        help="Definition: The year the borrower’s first credit line was opened."
    )
    open_acc = st.number_input(
        "Open Accounts", 0.0, 100.0, 15.0,
        help="Definition: The number of open credit lines currently in the borrower’s credit file."
    )
    pub_rec = st.number_input(
        "Public Records", 0.0, 10.0, 1.0,
        help="Definition: The number of derogatory public records (e.g., bankruptcy filings, tax liens)."
    )
    revol_util = st.number_input(
        "Revolving Utilization (%)", 0.0, 100.0, 23.5,
        help="Definition: The percentage of available revolving credit currently being used."
    )
    total_acc = st.number_input(
        "Total Accounts", 0.0, 200.0, 26.0,
        help="Definition: The total number of credit accounts in the borrower’s credit file."
    )
    mort_acc = st.number_input(
        "Mortgage Accounts", 0.0, 50.0, 2.0,
        help="Definition: The number of mortgage accounts currently held by the borrower."
    )

with col3:
    pub_rec_bankruptcies = st.number_input(
        "Public Record Bankruptcies", 0.0, 10.0, 1.0,
        help="Definition: The number of bankruptcies in the borrower’s credit history."
    )
    annual_inc = st.number_input(
        "Annual Income", 0.0, 1e8, 125000.0,
        help="Definition: The borrower’s self-reported annual income."
    )
    fico_score = st.number_input(
        "FICO Score", 0.0, 900.0, 712.0,
        help="Definition: The borrower’s credit score at loan origination."
    )
    revol_bal = st.number_input(
        "Revolving Balance", 0.0, 1e7, 5000.0,
        help="Definition: The borrower’s total credit revolving balance."
    )
    initial_list_status_w = st.checkbox(
        "Initial List Status = w", value=True,
        help="Definition: Indicates if the loan was listed as whole loan (‘w’) or fractional (‘f’)."
    )
    application_type_joint = st.checkbox(
        "Joint Application", value=False,
        help="Definition: Whether the loan application was made jointly with another borrower."
    )

st.markdown("---")

# categorical variabes
subgrade_letter = st.selectbox(
    "Subgrade Letter", ["A","B","C","D","E","F","G"],
    help="Definition: Loan grade assigned by the lender, where 'A' is the best and 'G' the worst."
)
subgrade_number = st.selectbox(
    "Subgrade Number", ["1","2","3","4","5"],
    help="Definition: The finer classification within a loan grade (1 = best, 5 = worst)."
)
home_ownership = st.selectbox(
    "Home Ownership", ["OWN","RENT"],
    help="Definition: Indicates if the borrower owns, rents, or has a mortgage on their home."
)
verification_status = st.selectbox(
    "Verification Status", ["Source Verified","Verified"],
    help="Definition: Whether the borrower’s income was verified by the lender."
)
purpose = st.selectbox(
    "Purpose", [
        "credit_card","debt_consolidation","educational","home_improvement",
        "house","major_purchase","medical","moving","other","renewable_energy",
        "small_business","vacation","wedding"
    ],
    help="Definition: The stated reason for the loan request."
)
addr_state = st.selectbox(
    "Address State", [
        "AL","AR","AZ","CA","CO","CT","DC","DE","FL","GA","HI","IA","ID","IL","IN",
        "KS","KY","LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH",
        "NJ","NM","NV","NY","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VA",
        "VT","WA","WI","WV","WY"
    ],
    help="Definition: The U.S. state where the borrower resides."
)

# build feature vector (conv user input into the model input)
sample = {c: 0 for c in cols}
sample.update({
    "loan_amnt": loan_amnt,
    "term": term,
    "int_rate": int_rate,
    "installment": installment,
    "emp_length": emp_length,
    "dti": dti,
    "earliest_cr_line": earliest_cr_line,
    "open_acc": open_acc,
    "pub_rec": pub_rec,
    "revol_util": revol_util,
    "total_acc": total_acc,
    "mort_acc": mort_acc,
    "pub_rec_bankruptcies": pub_rec_bankruptcies,
    "log_annual_inc": np.log(annual_inc + 1),
    "fico_score": fico_score,
    "log_revol_bal": np.log(revol_bal + 1),
})

for key in [
    f"sub_grade_{subgrade_letter}{subgrade_number}",
    f"home_ownership_{home_ownership}",
    f"verification_status_{verification_status}",
    f"purpose_{purpose}",
    f"addr_state_{addr_state}",
]:
    if key in sample:
        sample[key] = 1

if "initial_list_status_w" in sample:
    sample["initial_list_status_w"] = int(initial_list_status_w)
if "application_type_Joint App" in sample:
    sample["application_type_Joint App"] = int(application_type_joint)

# verification of feature vecotr shape
df = pd.DataFrame([sample], columns=cols)
st.write(f"Feature vector shape: {df.shape[1]} columns (expected 120)")
st.dataframe(df.T, use_container_width=True)

# predict
if st.button("Predict"):
    try:
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0, 1]
        st.success(f"Prediction: {'Low Risk' if pred == 0 else 'High Risk'} ({proba:.2%})")

        # --- LOCAL EXPLAINABILITY ---
        st.subheader("Local Feature Impact (Sensitivity Analysis)")

        active_cols = [c for c in df.columns if df[c].iloc[0] != 0 or df[c].dtype in [np.float64, np.int64]]
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c in active_cols]

        base_pred = proba
        impacts = {}
        for col in numeric_cols:
            temp = df.copy()
            temp[col] = temp[col] * 1.1
            new_pred = model.predict_proba(temp)[0, 1]
            impacts[col] = new_pred - base_pred

        impact_df = pd.DataFrame({
            "Feature": impacts.keys(),
            "Change_in_Risk": impacts.values()
        }).sort_values("Change_in_Risk", ascending=False)

        st.bar_chart(impact_df.set_index("Feature"))
        st.caption("Features with positive values increase predicted risk when raised by 10%.")

        # pos neg tablel splti
        pos_df = impact_df[impact_df["Change_in_Risk"] > 0].copy()
        neg_df = impact_df[impact_df["Change_in_Risk"] < 0].copy()

        if not pos_df.empty:
            st.markdown("### 🔺 Features Increasing Risk")
            st.dataframe(pos_df.style.format({"Change_in_Risk": "{:.3f}"}))

        if not neg_df.empty:
            st.markdown("### 🟩 Features Decreasing Risk")
            st.dataframe(neg_df.style.format({"Change_in_Risk": "{:.3f}"}))

        # for ppl to read
        top_pos = pos_df.head(3)["Feature"].tolist()
        top_neg = neg_df.tail(3)["Feature"].tolist()

        reasons = []
        if top_pos:
            reasons.append("Risk increases mainly due to " + ", ".join(top_pos))
        if top_neg:
            reasons.append("Risk decreases with higher " + ", ".join(top_neg))
        st.markdown("### Explanation Summary")
        st.write(" | ".join(reasons) if reasons else "No significant feature influence detected.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
