import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Loss Fitter - Method SW3 精修版", layout="wide")

st.title("⚡ Power Device Loss Evaluator (精準標定版)")
st.markdown("""
### 修正說明：
為了避免落差，請務必先執行 **[步驟 2：標定]**，點選圖片中座標軸的起點與終點，系統會自動校正像素偏移。
""")

# --- 1. Session State 初始化 ---
if "curves" not in st.session_state:
    st.session_state.curves = {} 
if "calib_pts" not in st.session_state:
    st.session_state.calib_pts = [] # 儲存 (0,0) 和 (Max, Max) 的像素座標

# --- 2. Sidebar 控制區 ---
st.sidebar.header("🛠️ 曲線與標定管理")
if st.sidebar.button("🔄 重置所有標定與數據"):
    st.session_state.curves = {}
    st.session_state.calib_pts = []
    st.rerun()

# --- 3. 檔案上傳 ---
uploaded_file = st.file_uploader("1. 上傳 Datasheet 圖檔", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    width, height = img.size
    
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📍 圖片互動區")
        # 標定提示
        if len(st.session_state.calib_pts) < 2:
            steps = ["請點擊圖表的【左下角原點 (0,0)】", "請點擊圖表的【右上方最大刻度交點】"]
            st.warning(f"標定步驟：{steps[len(st.session_state.calib_pts)]}")
        else:
            st.success("✅ 標定完成！現在請開始點選曲線上的數據點。")
        
        value = streamlit_image_coordinates(img, key="calib_click")
        
        if value:
            curr_pt = (value["x"], value["y"])
            if len(st.session_state.calib_pts) < 2:
                if not st.session_state.calib_pts or curr_pt != st.session_state.calib_pts[-1]:
                    st.session_state.calib_pts.append(curr_pt)
                    st.rerun()
            else:
                # 這裡開始記錄數據點
                curve_name = st.session_state.get("current_curve", "Eon_Curve")
                if curve_name not in st.session_state.curves:
                    st.session_state.curves[curve_name] = {"points": [], "params": None}
                
                pts = st.session_state.curves[curve_name]["points"]
                if not pts or curr_pt != pts[-1]:
                    st.session_state.curves[curve_name]["points"].append(curr_pt)

    with col2:
        st.subheader("📏 2. 設定真實數值範圍")
        real_x_max = st.number_input("X 軸最大刻度 (例如 1000 A)", value=1000.0)
        real_y_max = st.number_input("Y 軸最大刻度 (例如 125 mJ)", value=125.0)
        
        if len(st.session_state.calib_pts) == 2:
            p0 = st.session_state.calib_pts[0] # 原點像素
            p_max = st.session_state.calib_pts[1] # 最大值像素
            
            # 計算縮放比例
            scale_x = real_x_max / (p_max[0] - p0[0])
            scale_y = real_y_max / (p_max[1] - p0[1]) # Y 軸在影像中是反向的

            st.write("---")
            curve_name = st.text_input("曲線名稱", value="Eon_Test")
            st.session_state["current_curve"] = curve_name
            
            if curve_name in st.session_state.curves:
                raw_pts = st.session_state.curves[curve_name]["points"]
                # 轉換像素到真實數值
                real_pts = []
                for px, py in raw_pts:
                    rx = (px - p0[0]) * scale_x
                    ry = (py - p0[1]) * scale_y
                    real_pts.append((rx, ry))
                
                df = pd.DataFrame(real_pts, columns=["Current (A)", "Energy (mJ)"])
                st.write(f"已擷取 {len(df)} 個點")
                st.dataframe(df)

    # --- 4. 擬合與驗證 ---
    if len(st.session_state.calib_pts) == 2 and curve_name in st.session_state.curves:
        st.divider()
        if st.button("🚀 執行 Method SW3 擬合", type="primary"):
            x_data, y_data = df["Current (A)"].values, df["Energy (mJ)"].values
            def sw3_func(i, A, B, C): return A + B * i + C * i**2
            try:
                popt, _ = curve_fit(sw3_func, x_data, y_data)
                st.session_state.curves[curve_name]["params"] = popt
                st.success(f"擬合成功！A={popt[0]:.4f}, B={popt[1]:.4f}, C={popt[2]:.4f}")
            except:
                st.error("擬合失敗，請增加點選的點數。")

        # --- 5. 繪圖驗證 ---
        if st.session_state.curves[curve_name]["params"] is not None:
            p = st.session_state.curves[curve_name]["params"]
            fig, ax = plt.subplots()
            xi = np.linspace(0, real_x_max, 100)
            yi = p[0] + p[1] * xi + p[2] * xi**2
            
            ax.plot(xi, yi, label=f"Fitted {curve_name}")
            ax.scatter(df["Current (A)"], df["Energy (mJ)"], color='red', label="Data Points")
            ax.set_xlim(0, real_x_max)
            ax.set_ylim(0, real_y_max)
            ax.set_xlabel("Current (A)")
            ax.set_ylabel("Energy (mJ)")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)