
import streamlit as st
import pandas as pd
import json
from PSI_02_19_Gemini_v2_Fixed import PSICalculator

# Page setup
st.set_page_config(page_title="PSI Analyzer", layout="wide")

PSI_CODES = [f"PSI_{i:02}" for i in range(2, 20)]

# Validate required headers
def validate_file_headers(df):
    required_columns = [
        "EncounterID", "AGE", "SEX", "ATYPE", "BWGT", "MS-DRG", "MDC", "Pdx",
    ] + [f"POA{i}" for i in range(1, 26)] + [f"DX{i}" for i in range(1, 26)] +         [f"Proc{i}" for i in range(1, 11)] + [f"Proc{i}_Date" for i in range(1, 11)] + [f"Proc{i}_Time" for i in range(1, 11)] +         ["Admission_Date", "Discharge_Date", "Length_of_stay", "DQTR", "YEAR", "Discharge_Disposition"]
    missing = [col for col in required_columns if col not in df.columns]
    return missing

def run_psi_analysis(df, calculator, appendix):
    results = []
    errors = []
    for idx, row in df.iterrows():
        enc_id = row.get("EncounterID", f"Row{idx+1}")
        for psi_code in PSI_CODES:
            psi_func_name = f"evaluate_psi{psi_code[-2:]}"
            psi_func = getattr(calculator, psi_func_name, None)
            if callable(psi_func):
                try:
                    status, rationale = psi_func(row, appendix)
                    results.append({
                        "EncounterID": enc_id,
                        "PSI": psi_code,
                        "Status": status,
                        "Rationale": rationale
                    })
                except Exception as e:
                    errors.append({
                        "EncounterID": enc_id,
                        "PSI": psi_code,
                        "Error": str(e)
                    })
            else:
                errors.append({
                    "EncounterID": enc_id,
                    "PSI": psi_code,
                    "Error": f"{psi_func_name} not implemented"
                })
    return pd.DataFrame(results), pd.DataFrame(errors)

def display_dashboard(df):
    if "Status" not in df.columns:
        st.warning("No PSI results to display.")
        return
    total = len(df)
    inclusions = (df["Status"] == "Inclusion").sum()
    exclusions = (df["Status"] == "Exclusion").sum()
    errors = (df["Status"] == "Error").sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Evaluated", total)
    col2.metric("Inclusions", inclusions)
    col3.metric("Exclusions", exclusions)
    col4.metric("Errors", errors)

def display_results_table(results_df):
    psi_filter = st.multiselect("Filter by PSI", sorted(results_df["PSI"].unique()))
    status_filter = st.multiselect("Filter by Status", ["Inclusion", "Exclusion", "Error"])

    filtered_df = results_df.copy()
    if psi_filter:
        filtered_df = filtered_df[filtered_df["PSI"].isin(psi_filter)]
    if status_filter:
        filtered_df = filtered_df[filtered_df["Status"].isin(status_filter)]

    st.dataframe(filtered_df, use_container_width=True)
    st.download_button("⬇ Download Results", data=filtered_df.to_csv(index=False, encoding="utf-8-sig"), file_name="PSI_Results.csv")

def main():
    st.title("🧬 Patient Safety Indicator (PSI) Analyzer")
    st.markdown("Upload claim-level data and evaluate inclusion/exclusion across **PSI 02 to 19**.")

    uploaded_file = st.file_uploader("📂 Upload Excel or CSV File", type=["xlsx", "xls", "csv"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            df.columns = df.columns.str.strip()  # Normalize headers

            st.success(f"File uploaded: {uploaded_file.name}")
            st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

            missing = validate_file_headers(df)
            if missing:
                st.error(f"❌ Missing required columns: {missing}")
                return

            try:
                with open("PSI_02_19_Compiled_Stitched_Final.json", "r") as f:
                    appendix = json.load(f)
            except Exception as e:
                st.error(f"Failed to load Appendix JSON: {e}")
                return

            calculator = PSICalculator()

            with st.spinner("🧪 Running PSI analysis..."):
                results_df, error_df = run_psi_analysis(df, calculator, appendix)

            st.subheader("📊 Dashboard")
            display_dashboard(results_df)

            st.subheader("📋 PSI Results")
            st.write(f"{len(results_df)} rows evaluated.")
            display_results_table(results_df)

            if not error_df.empty:
                st.subheader("⚠️ Error Log")
                st.dataframe(error_df)
                st.download_button("⬇ Download Errors", data=error_df.to_csv(index=False, encoding="utf-8-sig"), file_name="PSI_Errors.csv")

        except Exception as e:
            st.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
