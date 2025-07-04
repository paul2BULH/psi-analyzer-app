
import streamlit as st
import pandas as pd
import json
from PSI_02_19_Gemini_v2_Fixed import PSICalculator
from typing import Dict, Any
import google.generativeai as genai

# ✅ Secure API Key handling
GEMINI_API_KEY = st.secrets["GEMINI"]["api_key"]

@st.cache_resource
def load_gemini_model():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-1.5-pro")

def get_gemini_explanation(prompt: str) -> str:
    try:
        model = load_gemini_model()
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini error: {str(e)}"

st.set_page_config(page_title="Enhanced PSI Analyzer", layout="wide")

@st.cache_data
def load_appendix():
    with open("PSI_02_19_Compiled_Stitched_Final.json", "r") as f:
        return json.load(f)

def validate_headers(df: pd.DataFrame) -> list:
    required = ["EncounterID", "AGE", "MS-DRG", "Pdx", "Admission_Date", "Discharge_Date", "Length_of_stay"]
    return [col for col in required if col not in df.columns]

def process_psi(encounter_df: pd.DataFrame, appendix: Dict[str, Any]):
    calculator = PSICalculator()
    results = []
    for idx, row in encounter_df.iterrows():
        enc_id = row.get("EncounterID", f"Row{idx}")
        encounter_summary = {"EncounterID": enc_id, "Triggered": [], "Details": {}}
        for psi_code in [f"PSI_{i:02}" for i in range(2, 20)]:
            fn_name = f"evaluate_psi{psi_code[-2:]}"
            psi_func = getattr(calculator, fn_name, None)
            if psi_func:
                try:
                    status, rationale, matches = psi_func(row, appendix, debug=True)
                    if status == "Inclusion":
                        encounter_summary["Triggered"].append(psi_code)
                    encounter_summary["Details"][psi_code] = {
                        "Status": status,
                        "Rationale": rationale,
                        "Checklist": matches
                    }
                except Exception as e:
                    encounter_summary["Details"][psi_code] = {
                        "Status": "Error",
                        "Rationale": str(e),
                        "Checklist": {}
                    }
        results.append(encounter_summary)
    return results

def render_checklist(details: dict, psi: str):
    data = details["Details"][psi]
    st.markdown(f"### PSI: {psi} – {data['Status']}")
    st.markdown(f"**Rationale**: {data['Rationale']}")
    checklist = data.get("Checklist", {})
    if checklist:
        st.write("**Checklist Validation:**")
        df = pd.DataFrame.from_dict(checklist, orient="index", columns=["Status"])
        df.index.name = "Rule"
        st.dataframe(df)

def render_flat_table(results):
    flat_rows = []
    for entry in results:
        enc_id = entry["EncounterID"]
        for psi, detail in entry["Details"].items():
            flat_rows.append({
                "EncounterID": enc_id,
                "PSI": psi,
                "Status": detail["Status"],
                "Rationale": detail["Rationale"],
                "Checklist": detail["Checklist"]
            })
    df = pd.DataFrame(flat_rows)
    st.dataframe(df[["EncounterID", "PSI", "Status", "Rationale"]], use_container_width=True)

    if st.checkbox("🔍 Enable Gemini Explanation"):
        row_index = st.number_input("Enter row index to explain", min_value=0, max_value=len(df)-1, step=1)
        selected = df.iloc[row_index]
        prompt = f"Explain why PSI code {selected['PSI']} was marked as {selected['Status']} for the following rationale: {selected['Rationale']}"
        st.markdown("🧠 **Gemini Explanation Prompt:**")
        st.code(prompt)
        explanation = get_gemini_explanation(prompt)
        st.markdown("🧠 **Gemini Explanation Output:**")
        st.info(explanation)

def main():
    st.title("🧬 Enhanced Patient Safety Indicator (PSI) Analyzer")
    appendix = load_appendix()
    uploaded_file = st.file_uploader("📂 Upload input Excel or CSV", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df.columns = df.columns.str.strip()
        missing = validate_headers(df)
        if missing:
            st.error(f"Missing required columns: {missing}")
            return

        st.success("✅ File uploaded and validated.")
        results = process_psi(df, appendix)

        view_mode = st.radio("Select View Mode", ["Encounter Summary", "Flat PSI Table"], horizontal=True)

        if view_mode == "Encounter Summary":
            for entry in results:
                with st.expander(f"📌 Encounter: {entry['EncounterID']} | Triggered: {', '.join(entry['Triggered']) or 'None'}"):
                    for psi in entry["Details"]:
                        with st.expander(f"📍 {psi} – {entry['Details'][psi]['Status']}"):
                            render_checklist(entry, psi)
        else:
            render_flat_table(results)

if __name__ == "__main__":
    main()
