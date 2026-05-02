import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Total Loss Evaluator", layout="wide")

# ==========================================
# 1. 曲線類別定義 (封裝論文公式)
# ==========================================
class SwitchingCurve:
    """切換損耗類別 (Method SW3)[cite: 1]"""
    def __init__(self, name):
        self.name = name
        self.params = None # [A, B, C]
        self.raw_pts = []
        self.real_pts = []
    def get_val(self, i):
        if self.params is None: return 0.0
        return self.params[0] + self.params[1]*abs(i) + self.params[2]*(i**2)

class ConductionCurve:
    """導通損耗類別 (Method Con1: Linear Approximation)"""
    def __init__(self, name):
        self.name = name
        self.params = None # [Vx, Rx] -> v = Rx*i + Vx
        self.raw_pts = []
        self.real_pts = []
    def get_voltage(self, i):
        """計算導通壓降 v_X"""
        if self.params is None: return 0.0
        return self.params[1]*abs(i) + self.params[0]
    def get_power(self, i_rms, i_avg):
        """計算平均導通損耗 (Equation 16)"""
        if self.params is None: return 0.0
        Vx, Rx = self.params
        return Rx * (i_rms**2) + Vx * i_avg

# ==========================================
# 2. Session State 初始化
# ==========================================
if "sw_curves" not in st.session_state: st.session_state.sw_curves = {}
if "con_curves" not in st.session_state: st.session_state.con_curves = {}
if "calib_sw" not in st.session_state: st.session_state.calib_sw = []
if "calib_con" not in st.session_state: st.session_state.calib_con = []

st.title("🚀 Power Device 綜合損耗建模系統")

# ==========================================
# 3. 側邊欄：統一曲線管理區
# ==========================================
with st.sidebar:
    st.header("📊 曲線管理中心")
    
    # 切換損管理
    st.subheader("🔹 切換損耗 (Switching)")
    sw_name = st.text_input("切換曲線名", "Eon_Test")
    if st.button("➕ 新增切換曲線"):
        st.session_state.sw_curves[sw_name] = SwitchingCurve(sw_name)
    
    # 導通損管理
    st.subheader("🔸 導通損耗 (Conduction)")
    con_name = st.text_input("導通曲線名", "IGBT_Con_150C")
    if st.button("➕ 新增導通曲線"):
        st.session_state.con_curves[con_name] = ConductionCurve(con_name)
    
    st.divider()
    if st.button("🗑️ 清空所有數據"):
        st.session_state.clear()
        st.rerun()

# ==========================================
# 4. 切換損耗作業區 (不修改現行功能)
# ==========================================
st.header("1️⃣ 切換損耗建模 (Method SW3)")
sw_list = list(st.session_state.sw_curves.keys())
if not sw_list:
    st.info("💡 指示：請先在左側新增一條『切換曲線』。")
else:
    active_sw = st.selectbox("當前編輯切換曲線", sw_list)
    up_sw = st.file_uploader("上傳切換損圖檔", type=["png", "jpg"], key="up_sw")
    
    if up_sw:
        img_sw = Image.open(up_sw)
        c1, c2 = st.columns([2, 1])
        with c1:
            st.write(f"📍 點選模式：{'標定座標' if len(st.session_state.calib_sw)<2 else '擷取數據'}")
            v_sw = streamlit_image_coordinates(img_sw, key="click_sw")
            if v_sw:
                p = (v_sw["x"], v_sw["y"])
                if len(st.session_state.calib_sw) < 2:
                    if not st.session_state.calib_sw or p != st.session_state.calib_sw[-1]:
                        st.session_state.calib_sw.append(p)
                        st.rerun()
                else:
                    obj = st.session_state.sw_curves[active_sw]
                    if not obj.raw_pts or p != obj.raw_pts[-1]:
                        obj.raw_pts.append(p)
        with c2:
            x_m = st.number_input("SW X軸最大(A)", 1000.0)
            y_m = st.number_input("SW Y軸最大(mJ)", 125.0)
            if len(st.session_state.calib_sw) == 2:
                p0, pm = st.session_state.calib_sw
                sx, sy = x_m/(pm[0]-p0[0]), y_m/(pm[1]-p0[1])
                obj = st.session_state.sw_curves[active_sw]
                obj.real_pts = [((px-p0[0])*sx, (py-p0[1])*sy) for px, py in obj.raw_pts]
                if st.button(f"擬合 {active_sw}"):
                    x_d, y_d = np.array([p[0] for p in obj.real_pts]), np.array([p[1] for p in obj.real_pts])
                    popt, _ = curve_fit(lambda i,A,B,C: A+B*i+C*i**2, x_d, y_d)
                    obj.params = popt
                    st.success("✅ 擬合成功")

# ==========================================
# 5. 導通損耗作業區 (新功能：Method Con1)
# ==========================================
st.divider()
st.header("2️⃣ 導通損耗建模 (Method Con1)")
con_list = list(st.session_state.con_curves.keys())

if not con_list:
    st.info("💡 指示：請先在左側新增一條『導通曲線』(如 IGBT 或 FWD)。")
else:
    active_con = st.selectbox("當前編輯導通曲線", con_list)
    up_con = st.file_uploader("上傳導通特性圖 (v-i curve)", type=["png", "jpg"], key="up_con")
    
    if up_con:
        img_con = Image.open(up_con)
        c1, c2 = st.columns([2, 1])
        with c1:
            if len(st.session_state.calib_con) < 2:
                st.warning("👉 指示：請點擊圖片中的【原點 0,0】與【最大刻度點】進行標定。")
            else:
                st.success("👉 指示：請沿著 v-i 曲線點選數據點進行線性擬合。")
            
            v_con = streamlit_image_coordinates(img_con, key="click_con")
            if v_con:
                p = (v_con["x"], v_con["y"])
                if len(st.session_state.calib_con) < 2:
                    if not st.session_state.calib_con or p != st.session_state.calib_con[-1]:
                        st.session_state.calib_con.append(p)
                        st.rerun()
                else:
                    obj = st.session_state.con_curves[active_con]
                    if not obj.raw_pts or p != obj.raw_pts[-1]:
                        obj.raw_pts.append(p)
        
        with c2:
            # 論文中 v-i 圖通常 x 軸是電壓 (V)，y 軸是電流 (A)
            con_x_m = st.number_input("V-I圖 電壓最大值(V)", 5.0)
            con_y_m = st.number_input("V-I圖 電流最大值(A)", 800.0)
            
            if len(st.session_state.calib_con) == 2:
                p0, pm = st.session_state.calib_con
                # 計算像素到物理值的縮放 (x: V, y: I)
                sx, sy = con_x_max_scale = con_x_m/(pm[0]-p0[0]), con_y_m/(pm[1]-p0[1])
                obj = st.session_state.con_curves[active_con]
                # 實體數據點轉換 (電壓, 電流)
                obj.real_pts = [((px-p0[0])*sx, (py-p0[1])*sy) for px, py in obj.raw_pts]
                st.write(f"已擷取 {len(obj.real_pts)} 個點")
                
                if st.button(f"擬合導通特性 {active_con}"):
                    # 依據 Eq 13: v = Rx*i + Vx. 我們擬合的是 V(電壓) 對 I(電流)
                    v_data = np.array([p[0] for p in obj.real_pts])
                    i_data = np.array([p[1] for p in obj.real_pts])
                    # 線性擬合: v = p[1]*i + p[0] (p[0]=Vx, p[1]=Rx)
                    z = np.polyfit(i_data, v_data, 1)
                    obj.params = [z[1], z[0]] # [Vx, Rx]
                    st.success(f"✅ 擬合完成：Vx={z[1]:.4f}V, Rx={z[0]*1000:.4f}mΩ")

# ==========================================
# 6. 結果可視化區 (分開呈現)
# ==========================================
st.divider()
st.header("📊 擬合結果視覺化")
res_c1, res_c2 = st.columns(2)

with res_c1:
    st.subheader("切換損耗曲線 (mJ)")
    fig_sw, ax_sw = plt.subplots()
    for name, obj in st.session_state.sw_curves.items():
        if obj.params is not None:
            xi = np.linspace(0, 1000, 100)
            yi = [obj.get_val(x) for x in xi]
            ax_sw.plot(xi, yi, label=name)
    ax_sw.set_xlabel("Current (A)"); ax_sw.set_ylabel("Energy (mJ)"); ax_sw.legend()
    st.pyplot(fig_sw)

with res_c2:
    st.subheader("導通特性曲線 (V-I)")
    fig_con, ax_con = plt.subplots()
    for name, obj in st.session_state.con_curves.items():
        if obj.params is not None:
            ii = np.linspace(0, 800, 100)
            vv = [obj.get_voltage(i) for i in ii]
            ax_con.plot(vv, ii, label=name) # 繪製傳統 V-I 圖
    ax_con.set_xlabel("Voltage (V)"); ax_con.set_ylabel("Current (A)"); ax_con.legend()
    st.pyplot(fig_con)