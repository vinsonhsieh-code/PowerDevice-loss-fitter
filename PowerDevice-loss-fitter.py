import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Loss Fitter - Method SW3", layout="wide")

st.title("⚡ Power Device Loss Evaluator (Method SW3)")
st.markdown("""
本工具採用論文中的 **Method SW3**：使用二階多項式 $E = A + B \cdot i + C \cdot i^2$ 進行曲線擬合。
""")

# --- 1. Session State 初始化 ---
if "curves" not in st.session_state:
    st.session_state.curves = {} 
if "current_curve_name" not in st.session_state:
    st.session_state.current_curve_name = "Curve 1"

# --- 2. Sidebar 控制區 ---
st.sidebar.header("🛠️ 曲線管理")

new_name = st.sidebar.text_input("新曲線名稱", value=f"Curve {len(st.session_state.curves) + 1}")
if st.sidebar.button("➕ 新增一條曲線檔案"):
    if new_name not in st.session_state.curves:
        st.session_state.curves[new_name] = {"points": [], "model": "Method SW3 (Polynomial)", "params": None}
        st.session_state.current_curve_name = new_name
        st.rerun()

all_curve_names = list(st.session_state.curves.keys())
if all_curve_names:
    selected_name = st.sidebar.selectbox("切換編輯對象", all_curve_names, index=all_curve_names.index(st.session_state.current_curve_name))
    st.session_state.current_curve_name = selected_name
    
    # 模型選擇，預設為 Method SW3
    model_type = st.sidebar.selectbox("擬合模型", [
        "Method SW3 (2nd Order Poly: E = A + B*i + C*i^2)",
        "Linear: V = V0 + r*I",
        "Power: E = a*I^b"
    ])
    st.session_state.curves[selected_name]["model"] = model_type
else:
    st.sidebar.warning("請先新增一條曲線")
    st.stop()

# --- 3. 檔案上傳 ---
uploaded_file = st.file_uploader("上傳 Datasheet 圖檔", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    width, height = img.size
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"📍 正在編輯: {st.session_state.current_curve_name}")
        value = streamlit_image_coordinates(img, key="pill")
        if value:
            point = (value["x"], value["y"])
            if not st.session_state.curves[st.session_state.current_curve_name]["points"] or \
               point != st.session_state.curves[st.session_state.current_curve_name]["points"][-1]:
                st.session_state.curves[st.session_state.current_curve_name]["points"].append(point)
        
        if st.button("🗑️ 清除目前曲線數據點"):
            st.session_state.curves[st.session_state.current_curve_name]["points"] = []
            st.rerun()

    with col2:
        st.subheader("📏 座標標定")
        x_min = st.number_input("X 軸最小值 (I_min)", value=0.0)
        x_max = st.number_input("X 軸最大值 (I_max)", value=100.0)
        y_min = st.number_input("Y 軸最小值 (E_min)", value=0.0)
        y_max = st.number_input("Y 軸最大值 (E_max)", value=10.0)
        
        points = st.session_state.curves[st.session_state.current_curve_name]["points"]
        if points:
            data_pts = []
            for px, py in points:
                real_x = x_min + (px / width) * (x_max - x_min)
                real_y = y_max - (py / height) * (y_max - y_min)
                data_pts.append((real_x, real_y))
            df = pd.DataFrame(data_pts, columns=["X", "Y"])
            st.dataframe(df)

    # --- 4. 擬合與驗證 ---
    st.divider()
    if st.button("🚀 執行 Method SW3 擬合並繪圖驗證", type="primary"):
        if len(points) < 4:
            st.error("二階多項式建議至少選取 4 個點以確保精確度")
        else:
            x_data, y_data = df["X"].values, df["Y"].values
            try:
                if "Method SW3" in model_type:
                    # E = A + B*i + C*i^2
                    def func(i, A, B, C): return A + B * i + C * i**2
                    popt, _ = curve_fit(func, x_data, y_data)
                    st.session_state.curves[selected_name]["params"] = popt
                    st.success(f"✅ 擬合成功! A={popt[0]:.6f}, B={popt[1]:.6f}, C={popt[2]:.6f}")
                elif "Linear" in model_type:
                    def func(i, v0, r): return v0 + r * i
                    popt, _ = curve_fit(func, x_data, y_data)
                    st.session_state.curves[selected_name]["params"] = popt
                else:
                    def func(i, a, b): return a * (i**b)
                    popt, _ = curve_fit(func, x_data, y_data)
                    st.session_state.curves[selected_name]["params"] = popt
            except Exception as e:
                st.error(f"擬合失敗: {e}")

    # --- 5. 繪製驗證圖 ---
    if st.session_state.curves[selected_name]["params"] is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, data in st.session_state.curves.items():
            if data["params"] is not None:
                p = data["params"]
                x_test = np.linspace(x_min, x_max, 100)
                if "Method SW3" in data["model"]:
                    y_test = p[0] + p[1] * x_test + p[2] * x_test**2
                    label = f"{name} (A={p[0]:.2f}, B={p[1]:.2f}, C={p[2]:.2f})"
                elif "Linear" in data["model"]:
                    y_test = p[0] + p[1] * x_test
                    label = f"{name} (Linear)"
                else:
                    y_test = p[0] * (x_test**p[1])
                    label = f"{name} (Power)"
                ax.plot(x_test, y_test, label=label)
                if name == selected_name:
                    ax.scatter(df["X"], df["Y"], color='red', label="Data Points")
        ax.set_xlabel("Current (A)")
        ax.set_ylabel("Loss / Voltage")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # --- 6. 係數表彙整 ---
    if st.session_state.curves:
        st.subheader("📋 Method SW3 係數總結 (A, B, C)")
        summary = []
        for name, data in st.session_state.curves.items():
            if data["params"] is not None:
                p = data["params"]
                if "Method SW3" in data["model"]:
                    summary.append({"Curve": name, "A (sw)": p[0], "B (sw)": p[1], "C (sw)": p[2]})
                else:
                    summary.append({"Curve": name, "Param 1": p[0], "Param 2": p[1], "Param 3": "N/A"})
        st.table(pd.DataFrame(summary))