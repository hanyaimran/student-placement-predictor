import streamlit as st
import pickle
import numpy as np
import pandas as pd

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #f8f7ff 0%,
        #eef7ff 50%,
        #f8faff 100%
    );
}

/* Main title */
.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    color: #6C63FF;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #666666;
    font-size: 16px;
    margin-bottom: 30px;
}

/* Section headings */
.section-title {
    font-size: 25px;
    font-weight: 700;
    color: #29245c;
    margin-top: 20px;
    margin-bottom: 15px;
}

/* Prediction box */
.prediction-container {
    background: linear-gradient(
        135deg,
        #6C63FF,
        #8E5CF7,
        #00B4D8
    );
    padding: 30px;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin: 20px 0;
}

.prediction-label {
    font-size: 16px;
    color: white;
    opacity: 0.9;
}

.prediction-result {
    font-size: 32px;
    font-weight: 800;
    color: white;
    margin: 10px 0;
}

.prediction-probability {
    font-size: 20px;
    color: white;
}

/* Streamlit button */
.stButton > button {
    background: linear-gradient(
        90deg,
        #6C63FF,
        #00B4D8
    );
    color: white;
    border: none;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: 700;
}

.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #594fe8,
        #0099bb
    );
    color: white;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #e5e5f0;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
}

/* Footer */
.footer {
    text-align: center;
    color: #777777;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

try:

    with open("model.pkl", "rb") as file:
        model_data = pickle.load(file)

    clf = model_data["model"]
    scaler = model_data["scaler"]

except FileNotFoundError:

    st.error(
        "❌ model.pkl nahi mila. "
        "model.pkl ko app.py ke same folder mein rakhein."
    )

    st.stop()

except Exception as e:

    st.error(f"❌ Model load karte waqt error: {e}")

    st.stop()

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🎓 Student Placement Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Powered Placement Analysis'
    '</div>',
    unsafe_allow_html=True
)

# =========================================================
# STUDENT INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">👩‍🎓 Student Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

# =========================================================
# CGPA
# =========================================================

with col1:

    cgpa = st.slider(
        "📚 CGPA",
        min_value=0.0,
        max_value=10.0,
        value=7.0,
        step=0.1
    )

# =========================================================
# IQ
# =========================================================

with col2:

    iq = st.slider(
        "🧠 IQ Score",
        min_value=50,
        max_value=180,
        value=100,
        step=1
    )

st.write("")

# =========================================================
# PREDICT BUTTON
# =========================================================

predict = st.button(
    "🔮 Predict Placement",
    type="primary",
    use_container_width=True
)

# =========================================================
# PREDICTION
# =========================================================

if predict:

    # -----------------------------------------------------
    # INPUT DATA
    # -----------------------------------------------------

    input_data = np.array([
        [cgpa, iq]
    ])

    # -----------------------------------------------------
    # SCALE INPUT
    # -----------------------------------------------------

    input_scaled = scaler.transform(input_data)

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    prediction = clf.predict(input_scaled)[0]

    # -----------------------------------------------------
    # PROBABILITY
    # -----------------------------------------------------

    probabilities = clf.predict_proba(input_scaled)[0]

    # Safely map probabilities to classes
    class_probabilities = dict(
        zip(clf.classes_, probabilities)
    )

    placed_probability = (
        class_probabilities.get(1, 0) * 100
    )

    not_placed_probability = (
        class_probabilities.get(0, 0) * 100
    )

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    if prediction == 1:

        result = "LIKELY TO BE PLACED"

        st.success(
            "🎉 The model predicts that the student is likely to be placed."
        )

    else:

        result = "LIKELY NOT TO BE PLACED"

        st.warning(
            "⚠️ The model predicts that the student is unlikely to be placed."
        )

    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🎯 Prediction Result</div>',
        unsafe_allow_html=True
    )

    # IMPORTANT:
    # No custom HTML div here.
    # Native Streamlit components are used.

    if prediction == 1:

        st.success(
            f"🎉 {result}"
        )

    else:

        st.error(
            f"⚠️ {result}"
        )

    st.metric(
        "🎯 Placement Probability",
        f"{placed_probability:.1f}%"
    )

    # =====================================================
    # STUDENT METRICS
    # =====================================================

    st.markdown(
        '<div class="section-title">📋 Student Metrics</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📚 CGPA",
            f"{cgpa:.1f} / 10"
        )

    with col2:

        st.metric(
            "🧠 IQ Score",
            f"{iq}"
        )

    with col3:

        st.metric(
            "🎯 Placement Probability",
            f"{placed_probability:.1f}%"
        )

    # =====================================================
    # WHY THIS PREDICTION
    # =====================================================

    st.markdown(
        '<div class="section-title">🔎 Why This Prediction?</div>',
        unsafe_allow_html=True
    )

    st.info(
        f"""
        **Student Information**

        📚 **CGPA:** {cgpa:.1f} / 10

        🧠 **IQ Score:** {iq}

        The model evaluates these two features after applying
        the same scaling process that was used during model training.
        """
    )

    # =====================================================
    # PROBABILITY PROGRESS
    # =====================================================

    st.progress(
        min(int(round(placed_probability)), 100),
        text=f"Placement Probability: {placed_probability:.1f}%"
    )

    # =====================================================
    # PROBABILITY GRAPH
    # =====================================================

    st.markdown(
        '<div class="section-title">📊 Prediction Probability</div>',
        unsafe_allow_html=True
    )

    probability_df = pd.DataFrame(
        {
            "Outcome": [
                "Not Placed",
                "Placed"
            ],
            "Probability": [
                not_placed_probability,
                placed_probability
            ]
        }
    )

    st.bar_chart(
        probability_df.set_index("Outcome"),
        color="#6C63FF"
    )

    # =====================================================
    # INTERPRETATION
    # =====================================================

    st.markdown(
        '<div class="section-title">💡 Interpretation</div>',
        unsafe_allow_html=True
    )

    if placed_probability >= 75:

        st.success(
            "🌟 Strong placement probability. "
            "The student's current profile is favorable according to the model."
        )

    elif placed_probability >= 50:

        st.info(
            "👍 Moderate placement probability. "
            "Improving academic performance and skills may increase the chances."
        )

    else:

        st.warning(
            "📈 Lower placement probability. "
            "The student may benefit from improving CGPA, "
            "technical skills, aptitude, and interview preparation."
        )

# =========================================================
# MACHINE LEARNING PIPELINE
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">⚙️ Machine Learning Pipeline</div>',
    unsafe_allow_html=True
)

st.caption(
    "From student data to final placement prediction"
)

p1, p2, p3, p4, p5 = st.columns(5)

with p1:

    st.info(
        """
        📂 **Dataset**

        Student placement data
        """
    )

with p2:

    st.info(
        """
        ✂️ **Train / Test**

        Data splitting
        """
    )

with p3:

    st.info(
        """
        📐 **Scaling**

        StandardScaler
        """
    )

with p4:

    st.info(
        """
        🤖 **Training**

        Logistic Regression
        """
    )

with p5:

    st.info(
        """
        🎯 **Prediction**

        Placed / Not Placed
        """

    )

# =========================================================
# MODEL DETAILS
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🧠 Model Details</div>',
    unsafe_allow_html=True
)

m1, m2, m3 = st.columns(3)

with m1:

    st.metric(
        "Algorithm",
        "Logistic Regression"
    )

with m2:

    st.metric(
        "Task",
        "Binary Classification"
    )

with m3:

    st.metric(
        "Features",
        "CGPA + IQ"
    )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="footer">
        Python • NumPy • Pandas • Scikit-learn • Streamlit
        <br>
        Student Placement Prediction | Machine Learning Portfolio Project
    </div>
    """,
    unsafe_allow_html=True
)