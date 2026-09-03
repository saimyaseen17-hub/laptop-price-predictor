import streamlit as st
import pandas as pd
import joblib


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("laptop_price_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")


# =========================================================
# LOAD CSS
# =========================================================

def load_css():
    with open("style.css", "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


load_css()


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("💻 Laptop Price Predictor")

st.markdown(
    """
    <p style="text-align:center; font-size:17px;">
        AI-powered laptop price estimation using Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# SPECIFICATIONS
# =========================================================

st.subheader("⚙️ Laptop Specifications")

st.write(
    "Select the laptop specifications below and get an estimated price."
)


# =========================================================
# ROW 1
# =========================================================

col1, col2, col3 = st.columns(3)


with col1:

    brand = st.selectbox(
        "🏷️ Laptop Brand",
        [
            "HP",
            "Acer",
            "Lenovo",
            "Apple",
            "Dell",
            "Asus",
            "Samsung",
            "Ultimus",
            "Primebook",
            "MSI",
            "Infinix",
            "Wings",
            "Honor",
            "Zebronics",
            "Xiaomi",
            "iBall",
            "Chuwi",
            "Realme",
            "Avita",
            "Walker",
            "Huawei",
            "Tecno",
            "Gigabyte",
            "Vaio",
            "Microsoft",
            "Fujitsu",
            "LG",
            "Ninkear",
            "Razer",
            "AXL"
        ]
    )


with col2:

    generation = st.selectbox(
        "⚙️ Processor Generation",
        [
            "3rd Gen",
            "4th Gen",
            "5th Gen",
            "6th Gen",
            "7th Gen",
            "8th Gen",
            "9th Gen",
            "10th Gen",
            "11th Gen",
            "12th Gen",
            "13th Gen",
            "Apple M1",
            "Apple M2",
            "Other"
        ]
    )


with col3:

    processor_type = st.selectbox(
        "🧠 Processor Type",
        [
            "Intel Core i3",
            "Intel Core i5",
            "Intel Core i7",
            "Intel Core i9",
            "AMD Ryzen 3",
            "AMD Ryzen 5",
            "AMD Ryzen 7",
            "AMD Ryzen 9",
            "Intel Celeron",
            "Intel Pentium",
            "AMD Athlon",
            "Apple M1",
            "Apple M2",
            "Other"
        ]
    )


# =========================================================
# ROW 2
# =========================================================

col4, col5, col6 = st.columns(3)


with col4:

    ram = st.selectbox(
        "💾 RAM",
        [2, 4, 8, 12, 16, 32, 64],
        index=2,
        format_func=lambda x: f"{x} GB"
    )


with col5:

    ram_type = st.selectbox(
        "🔧 RAM Type",
        [
            "DDR3",
            "DDR4",
            "DDR5",
            "LPDDR4",
            "LPDDR4X",
            "LPDDR5",
            "LPDDR5X",
            "Unified"
        ]
    )


with col6:

    rom = st.selectbox(
        "💿 Storage",
        [32, 64, 128, 256, 512, 1024, 2048],
        index=4,
        format_func=lambda x:
        f"{x // 1024} TB" if x >= 1024 else f"{x} GB"
    )


# =========================================================
# ROW 3
# =========================================================

col7, col8, col9 = st.columns(3)


with col7:

    rom_type = st.selectbox(
        "📀 Storage Type",
        [
            "SSD",
            "Hard-Disk"
        ]
    )


with col8:

    gpu = st.selectbox(
        "🎮 GPU",
        [
            "NVIDIA",
            "AMD",
            "Intel",
            "Apple",
            "ARM",
            "Other"
        ]
    )


with col9:

    os = st.selectbox(
        "🖥️ Operating System",
        [
            "Windows 11",
            "Windows 10",
            "Windows",
            "macOS",
            "Chrome OS",
            "Ubuntu",
            "Android",
            "DOS",
            "Other"
        ]
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.write("")

predict_button = st.button(
    "🔮 Predict Laptop Price",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # Create input dataframe
    input_data = pd.DataFrame({

        "brand": [brand],

        "Generation": [generation],

        "Processor_Type": [processor_type],

        "Ram": [ram],

        "Ram_type": [ram_type],

        "ROM": [rom],

        "ROM_type": [rom_type],

        "GPU": [gpu],

        "OS": [os]
    })


    # One-hot encoding
    input_data = pd.get_dummies(
        input_data,
        dtype=int
    )


    # Match training features
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # Predict
    prediction = model.predict(input_data)[0]


    # Result
    st.success(
        f"💰 Estimated Laptop Price: Rs. {prediction:,.0f}"
    )


    st.info(
        "This is an estimated price generated by the trained "
        "Random Forest Regression model."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748b; padding:15px;">
        💻 Laptop Price Prediction System
        <br>
        <span style="color:#93c5fd;">
        Powered by Random Forest Machine Learning
        </span>
    </div>
    """,
    unsafe_allow_html=True
)