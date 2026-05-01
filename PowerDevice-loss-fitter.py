
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Loss Evaluator", layout="wide")

st.title("⚡ Power Device Loss Evaluator (Curve Fitting Tool)")
st.markdown("""
基於文獻 **"A New Approach for Power Losses Evaluation of IGBT_Diode Module"** 的概念，
本工具協助你從 Datasheet 圖檔中提取係數，並建立損耗模型。
""")

# 1. Sidebar - Model Selection
st.sidebar.header("模型設定")
model_type = st.sidebar.selectbox("選擇擬合模型", ["導通損耗 (Linear: V = V0 + r*I)", "切換損耗 (Power: E = a*I^b)"])

# 2. File Upload
uploaded_file = st.file_bar = st.file_uploader("上傳 Datasheet 圖檔 (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    width, height = img.size

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("1. 點擊圖表提取數據點")
        st.info("請依序點擊曲線上的點。建議至少選取 5 個點。")
        
        # Get coordinates from clicks
        value = streamlit_image_coordinates(img, key="pill")
        
        if "points" not in st.session_state:
            st.session_state.points = []
        
        if value:
            point = (value["x"], value["y"])
            if not st.session_state.points or point != st.session_state.points[-1]:
                st.session_state.points.append(point)
        
        if st.button("重置數據點"):
            st.session_state.points = []
            st.rerun()

    with col2:
        st.subheader("2. 座標軸標定 (Calibration)")
        st.write("請輸入圖表的座標軸邊界值以進行轉換：")
        x_min = st.number_input("X 軸最小值 (e.g., 0 A)", value=0.0)
        x_max = st.number_input("X 軸最大值 (e.g., 100 A)", value=100.0)
        y_min = st.number_input("Y 軸最小值 (e.g., 0 V or 0 mJ)", value=0.0)
        y_max = st.number_input("Y 軸最大值 (e.g., 3 V or 50 mJ)", value=10.0)
        
        if st.session_state.points:
            st.write(f"已選取點數: {len(st.session_state.points)}")
            
            # Convert pixel coords to data coords
            # Note: y-axis in image is top-down
            data_pts = []
            for px, py in st.session_state.points:
                real_x = x_min + (px / width) * (x_max - x_min)
                real_y = y_max - (py / height) * (y_max - y_min) # Inverse Y
                data_pts.append((real_x, real_y))
            
            df = pd.DataFrame(data_pts, columns=["X", "Y"])
            st.dataframe(df)

    # 3. Curve Fitting
    if len(st.session_state.points) >= 3:
        st.divider()
        st.subheader("3. 曲線擬合結果 (Curve Fitting)")
        
        x_data = df["X"].values
        y_data = df["Y"].values
        
        try:
            if "Linear" in model_type:
                # V = V0 + r * I
                def func(x, v0, r): return v0 + r * x
                popt, _ = curve_fit(func, x_data, y_data)
                v0, r = popt
                st.success(f"擬合成功！ 係數結果： **V0 = {v0:.4f}**, **r = {r:.4f}**")
                st.latex(f"V(i) = {v0:.4f} + {r:.4f} \cdot i")
            else:
                # E = a * I^b
                def func(x, a, b): return a * (x**b)
                popt, _ = curve_fit(func, x_data, y_data)
                a, b = popt
                st.success(f"擬合成功！ 係數結果： **a = {a:.4f}**, **b = {b:.4f}**")
                st.latex(f"E(i) = {a:.4f} \cdot i^{{{b:.4f}}}")

            # Plot Comparison
            fig, ax = plt.subplots()
            ax.scatter(x_data, y_data, color='red', label='Raw Points')
            x_range = np.linspace(x_min, x_max, 100)
            ax.plot(x_range, func(x_range, *popt), label='Fitted Curve')
            ax.set_xlabel("Current (A)")
            ax.set_ylabel("Value (V or mJ)")
            ax.legend()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"擬合出錯: {e}")

else:
    st.info("請上傳圖片以開始。")
