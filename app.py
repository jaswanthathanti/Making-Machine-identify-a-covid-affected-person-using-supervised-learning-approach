import streamlit as st
import pickle
import numpy as np
import time

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="COVID-19 Risk Prediction",
    page_icon="🦠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# ADVANCED UI CSS
# --------------------------------------------------
st.markdown("""
<style>
.main { padding: 1rem; }
.card {
    background: #ffffff;
    padding: 1.2rem;
    border-radius: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 1rem;
}
h1, h3 { text-align: center; }
.stButton>button {
    width: 100%;
    font-size: 18px;
    padding: 0.7rem;
    border-radius: 10px;
}
.result-box {
    padding: 1rem;
    border-radius: 12px;
    margin-top: 1rem;
}
.low { background: #e7fff1; color: #006b3c; }
.medium { background: #fff5e6; color: #b36b00; }
.high { background: #ffe6e6; color: #a80000; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# MULTI-LANGUAGE SUPPORT
# --------------------------------------------------
LANG = {
    "English": {
        "title": "COVID-19 Risk Prediction System",
        "subtitle": "Please answer the following questions",
        "symptoms": "Health Symptoms",
        "exposure": "Exposure History",
        "predict": "Predict COVID-19 Risk",
        "note": "This system is for educational purposes only."
    },
    "Hindi": {
        "title": "कोविड-19 जोखिम पूर्वानुमान प्रणाली",
        "subtitle": "कृपया निम्नलिखित प्रश्नों का उत्तर दें",
        "symptoms": "स्वास्थ्य लक्षण",
        "exposure": "संपर्क विवरण",
        "predict": "जोखिम जांचें",
        "note": "यह केवल शैक्षणिक उपयोग के लिए है।"
    },
    "Telugu": {
        "title": "కోవిడ్-19 ప్రమాద అంచనా వ్యవస్థ",
        "subtitle": "క్రింది ప్రశ్నలకు సమాధానం ఇవ్వండి",
        "symptoms": "ఆరోగ్య లక్షణాలు",
        "exposure": "సంపర్క వివరాలు",
        "predict": "ప్రమాదాన్ని అంచనా వేయండి",
        "note": "ఇది విద్యాపరమైన ఉపయోగం కోసం మాత్రమే."
    }
}

language = st.selectbox("🌐 Language", LANG.keys())
T = LANG[language]

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open("covid_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(f"""
<div class="card">
    <h1>🦠 {T['title']}</h1>
    <p style="text-align:center;">{T['subtitle']}</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# FORM (CHECKBOXES UNCHANGED)
# --------------------------------------------------
with st.form("covid_form"):

    st.markdown(f"<div class='card'><h3>{T['symptoms']}</h3>", unsafe_allow_html=True)
    breathing = st.checkbox("Breathing Problem")
    fever = st.checkbox("Fever")
    dry_cough = st.checkbox("Dry Cough")
    sore_throat = st.checkbox("Sore Throat")
    hypertension = st.checkbox("Hypertension")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='card'><h3>{T['exposure']}</h3>", unsafe_allow_html=True)
    abroad = st.checkbox("Recent Abroad Travel")
    contact = st.checkbox("Contact with COVID Patient")
    gathering = st.checkbox("Attended Large Gathering")
    public_place = st.checkbox("Visited Public Exposed Places")
    family_public = st.checkbox("Family Working in Public Exposed Places")
    st.markdown("</div>", unsafe_allow_html=True)

    submit = st.form_submit_button(T["predict"])

# --------------------------------------------------
# PREDICTION + RISK PERCENTAGE + ANIMATED BAR
# --------------------------------------------------
if submit:
    X = np.array([
        int(breathing),
        int(fever),
        int(dry_cough),
        int(sore_throat),
        int(hypertension),
        int(abroad),
        int(contact),
        int(gathering),
        int(public_place),
        int(family_public)
    ]).reshape(1, -1)

    prediction = model.predict(X)[0]
    risk_percent = round(model.predict_proba(X)[0][1] * 100, 2)

    st.divider()
    st.subheader("📊 Risk Assessment")

    # Animated progress bar
    progress = st.progress(0)
    for i in range(int(risk_percent) + 1):
        progress.progress(i)
        time.sleep(0.01)

    # Risk interpretation
    if risk_percent < 30:
        level = "LOW RISK"
        css = "low"
    elif risk_percent < 70:
        level = "MODERATE RISK"
        css = "medium"
    else:
        level = "HIGH RISK"
        css = "high"

    st.markdown(
        f"""
        <div class="result-box {css}">
            <h3>{level}</h3>
            <p><b>COVID-19 Risk Probability:</b> {risk_percent}%</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if prediction == 1:
        st.error("⚠️ High Risk: COVID-19 Positive")
        st.markdown("**Recommendation:** Please consult a medical professional immediately.")
    else:
        st.success("✅ Low Risk: COVID-19 Negative")
        st.markdown("**Recommendation:** Continue safety precautions.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(f"""
<p style="text-align:center; font-size:13px; color:gray;">
⚠️ {T['note']}<br>
Developed by <b>Athanti Jaswanth</b>
</p>
""", unsafe_allow_html=True)
