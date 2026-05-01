import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Loss Fitter Pro", layout="wide")

st.title("⚡ Power Device Loss Evaluator (Multi-Curve Version)")
st.markdown("""
本工具協助你從 Datasheet 讀取數據、進行多曲線擬合，並即時驗證係數準確性。
""")

# --- 1. Session State 初始化 ---
if "curves" not in st.session_state:
    st.session_state.curves = {}  # 格式: {"Curve 1": {"points": [], "model": "Linear", "params": None}}
if "current_curve_name" not in st.session_state:
    st.session_state.current_curve_name = "New Curve 1"

# --- 2. Sidebar 控制區 ---
st.sidebar.header("🛠️ 曲線管理")

# 新增曲線按鈕
new_name = st.sidebar.text_input("新曲線名稱", value=f"Curve {len(st.session_state.curves) + 1}")
if st.sidebar.button("➕ 新增一條曲線檔案"):
    if new_name not in st.session_state.curves:
        st.session_state.curves[new_name] = {"points": [], "model": "Linear", "params": None}
        st.session_state.current_curve_name = new_name
        st.rerun()

# 選擇目前要編輯的曲線
all_curve_names = list(st.session_state.curves.keys())
if all_curve_names:
    selected_name = st.sidebar.selectbox("切換編輯對象", all_curve_names, index=all_curve_names.index(st.session_state.current_curve_name))
    st.session_state.current_curve_name = selected_name
    
    # 選擇模型類型
    model_type = st.sidebar.selectbox("擬合模型", ["導通損耗 (Linear: V = V0 + r*I)", "切換損耗 (Power: E = a*I^b)"])
    st.session_state.curves[selected_name]["model"] = model_type
else:
    st.sidebar.warning("請先新增一條曲線")
    st.stop()

# --- 3. 檔案上傳 ---
uploaded_file = st.file_uploader("上傳 Datasheet 圖檔 (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    width, height = img.size
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"📍 正在編輯: {st.session_state.current_curve_name}")
        st.info("請在圖中點擊曲線路徑。")
        
        # 獲取點擊座標
        value = streamlit_image_coordinates(img, key="pill")
        
        if value:
            point = (value["x"], value["y"])
            # 避免重複記錄相同點
            if not st.session_state.curves[st.session_state.current_curve_name]["points"] or \
               point != st.session_state.curves[st.session_state.current_curve_name]["points"][-1]:
                st.session_state.curves[st.session_state.current_curve_name]["points"].append(point)
        
        if st.button("🗑️ 清除目前曲線數據點"):
            st.session_state.curves[st.session_state.current_curve_name]["points"] = []
            st.rerun()

    with col2:
        st.subheader("📏 座標標定 (Calibration)")
        x_min = st.number_input("X 軸最小值", value=0.0)
        x_max = st.number_input("X 軸最大值", value=100.0)
        y_min = st.number_input("Y 軸最小值", value=0.0)
        y_max = st.number_input("Y 軸最大值", value=10.0)
        
        points = st.session_state.curves[st.session_state.current_curve_name]["points"]
        if points:
            # 座標轉換邏輯
            data_pts = []
            for px, py in points:
                real_x = x_min + (px / width) * (x_max - x_min)
                real_y = y_max - (py / height) * (y_max - y_min)
                data_pts.append((real_x, real_y))
            
            df = pd.DataFrame(data_pts, columns=["X", "Y"])
            st.write("已選點數據 (Real Units):")
            st.dataframe(df)

    # --- 4. 擬合與驗證按鈕 ---
    st.divider()
    c1, c2 = st.columns(2)
    
    with c1:
        if st.button("🚀 執行擬合並繪圖驗證", type="primary"):
            if len(points) < 3:
                st.error("請至少點選 3 個點再進行擬合")
            else:
                x_data = df["X"].values
                y_data = df["Y"].values
                try:
                    if "Linear" in model_type:
                        def func(x, v0, r): return v0 + r * x
                        popt, _ = curve_fit(func, x_data, y_data)
                        st.session_state.curves[selected_name]["params"] = popt
                        st.success(f"✅ 擬合成功! V0={popt[0]:.4f}, r={popt[1]:.4f}")
                    else:
                        def func(x, a, b): return a * (x**b)
                        popt, _ = curve_fit(func, x_data, y_data)
                        st.session_state.curves[selected_name]["params"] = popt
                        st.success(f"✅ 擬合成功! a={popt[0]:.4f}, b={popt[1]:.4f}")
                except Exception as e:
                    st.error(f"擬合失敗: {e}")

    # --- 5. 繪製驗證圖表 ---
    if st.session_state.curves[selected_name]["params"] is not None:
        popt = st.session_state.curves[selected_name]["params"]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        # 繪製所有已完成擬合的曲線進行對比
        for name, data in st.session_state.curves.items():
            if data["params"] is not None:
                # 重新定義函數以繪圖
                if "Linear" in data["model"]:
                    f = lambda x, p0, p1: p0 + p1 * x
                    label_str = f"{name} (Linear)"
                else:
                    f = lambda x, p0, p1: p0 * (x**p1)
                    label_str = f"{name} (Power)"
                
                x_test = np.linspace(x_min, x_max, 100)
                ax.plot(x_test, f(x_test, *data["params"]), label=label_str)
                
                # 只有當前編輯的曲線顯示原始點，方便驗證準確度
                if name == selected_name:
                    pts = np.array(data_pts)
                    ax.scatter(pts[:,0], pts[:,1], color='red', label="Current Points")

        ax.set_xlabel("Current (A)")
        ax.set_ylabel("Value (V or mJ)")
        ax.set_title("Curve Accuracy Verification")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# --- 6. 最終係數匯總表 ---
if st.session_state.curves:
    st.divider()
    st.subheader("📋 係數彙整表")
    summary = []
    for name, data in st.session_state.curves.items():
        if data["params"] is not None:
            p = data["params"]
            summary.append({
                "Curve Name": name,
                "Model": data["model"],
                "Param A (V0/a)": f"{p[0]:.6f}",
                "Param B (r/b)": f"{p[1]:.6f}"
            })
    if summary:
        st.table(pd.DataFrame(summary))
    else:
        st.write("尚無已存檔的擬合結果。")

else:
    st.info("請上傳圖片以開始。")