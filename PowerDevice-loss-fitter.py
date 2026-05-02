import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Total Loss Evaluator", layout="wide")

# ==========================================
# 1. 核心類別定義 (封裝論文公式)
# ==========================================
class SwitchingCurve:
    """切換損耗 (Method SW3: E = A + B*i + C*i^2)"""
    def __init__(self, name):
        self.name = name
        self.params = None 
        self.raw_pts = []
        self.real_pts = []

    def get_val(self, i):
        if self.params is None: return 0.0
        i_abs = abs(i)
        return self.params[0] + self.params[1]*i_abs + self.params[2]*(i_abs**2)

class ConductionDevice:
    """導通損耗 (Method Con1 & Eq 19)"""
    def __init__(self, name):
        self.name = name
        self.t_min_c = 25.0   # 攝氏溫度
        self.t_max_c = 150.0  # 攝氏溫度
        self.fit_tmin = None  # [Vx, Rx]
        self.fit_tmax = None  # [Vx, Rx]
        self.raw_pts_tmin, self.raw_pts_tmax = [], []
        self.real_pts_tmin, self.real_pts_tmax = [], []

    def get_eq19_params(self, tj_c):
        """依據 Tj (°C) 解算 Rx 與 Vx (Eq 17, 18)"""
        if self.fit_tmin is None or self.fit_tmax is None: return None, None
        V1, R1 = self.fit_tmin
        V2, R2 = self.fit_tmax
        # 方程式中使用絕對溫度 (K) 進行線性比例運算
        T1, T2 = self.t_min_c + 273.15, self.t_max_c + 273.15
        Tj = tj_c + 273.15
        
        denom = T1 - T2
        rx_tj = ((R2*T1 - R1*T2) / denom) + ((R1 - R2) / denom) * Tj
        vx_tj = ((V2*T1 - V1*T2) / denom) + ((V1 - V2) / denom) * Tj
        return vx_tj, rx_tj

    def calc_pcon(self, tj_c, i_rms, i_avg):
        """計算導通損耗 (Eq 16/19)"""
        vx, rx = self.get_eq19_params(tj_c)
        if vx is None: return 0.0
        return rx * (i_rms**2) + vx * i_avg

# ==========================================
# 2. Session State 管理
# ==========================================
if "sw_curves" not in st.session_state: st.session_state.sw_curves = {}
if "igbt_obj" not in st.session_state: st.session_state.igbt_obj = ConductionDevice("IGBT")
if "fwd_obj" not in st.session_state: st.session_state.fwd_obj = ConductionDevice("FWD")
for k in ["calib_sw", "calib_igbt", "calib_fwd"]:
    if k not in st.session_state: st.session_state[k] = []

st.title("🚀 Power Device 綜合損耗建模系統 (攝氏單位修復版)")

# ==========================================
# 3. Sidebar 設定
# ==========================================
with st.sidebar:
    st.header("📊 曲線管理")
    sw_n = st.text_input("新增切換曲線 (如 Eon)", "Eon_Test")
    if st.button("➕ 新增切換曲線"): st.session_state.sw_curves[sw_n] = SwitchingCurve(sw_n)
    
    st.divider()
    st.subheader("🌡️ 導通損溫度基準 (°C)")
    tc1 = st.number_input("T_min (°C)", value=25.0)
    tc2 = st.number_input("T_max (°C)", value=150.0)
    st.session_state.igbt_obj.t_min_c = st.session_state.fwd_obj.t_min_c = tc1
    st.session_state.igbt_obj.t_max_c = st.session_state.fwd_obj.t_max_c = tc2

    if st.button("🔄 重置標定", type="primary"):
        st.session_state.calib_sw = []; st.session_state.calib_igbt = []; st.session_state.calib_fwd = []
        st.rerun()

# ==========================================
# 4. 三大作業作業區
# ==========================================

# --- A. 切換損區域 ---
st.header("1️⃣ 切換損耗建模 (Method SW3)")
sw_list = list(st.session_state.sw_curves.keys())
if not sw_list: st.info("💡 指示：請在側邊欄新增『切換曲線』。")
else:
    active_sw = st.selectbox("切換曲線對象", sw_list)
    up_sw = st.file_uploader("上傳切換損耗圖 (mJ)", type=["png", "jpg"], key="up_sw")
    if up_sw:
        img = Image.open(up_sw)
        c1, c2 = st.columns([2, 1])
        with c1:
            if len(st.session_state.calib_sw) < 2: st.warning("👉 Step 1: 標定原點(0,0)與最大刻度點。")
            v = streamlit_image_coordinates(img, key="click_sw")
            if v:
                p = (v["x"], v["y"])
                if len(st.session_state.calib_sw) < 2:
                    if not st.session_state.calib_sw or p != st.session_state.calib_sw[-1]:
                        st.session_state.calib_sw.append(p); st.rerun()
                else:
                    obj = st.session_state.sw_curves[active_sw]
                    if not obj.raw_pts or p != obj.raw_pts[-1]: obj.raw_pts.append(p)
        with c2:
            xm, ym = st.number_input("SW I(A)最大", 1000.0), st.number_input("SW E(mJ)最大", 125.0)
            if len(st.session_state.calib_sw) == 2:
                p0, pm = st.session_state.calib_sw
                sx, sy = xm/(pm[0]-p0[0]), ym/(pm[1]-p0[1])
                obj = st.session_state.sw_curves[active_sw]
                obj.real_pts = [((px-p0[0])*sx, (py-p0[1])*sy) for px, py in obj.raw_pts]
                if st.button(f"🚀 擬合切換損 {active_sw}"):
                    xd, yd = np.array([p[0] for p in obj.real_pts]), np.array([p[1] for p in obj.real_pts])
                    popt, _ = curve_fit(lambda i,A,B,C: A+B*i+C*i**2, xd, yd)
                    obj.params = popt; st.success("擬合成功")

# --- B. IGBT 導通損區域 ---
st.divider()
st.header("2️⃣ IGBT 導通特性建模 (Method Con1)")
tm_igbt = st.radio("IGBT 標定溫度點", [f"{tc1}°C", f"{tc2}°C"], horizontal=True, key="tm_igbt")
up_igbt = st.file_uploader("上傳 IGBT v-i 特性圖", type=["png", "jpg"], key="up_igbt")
if up_igbt:
    img = Image.open(up_igbt)
    c1, c2 = st.columns([2, 1])
    with c1:
        if len(st.session_state.calib_igbt) < 2: st.warning("👉 Step 1: 標定原點與最大點。")
        v = streamlit_image_coordinates(img, key="click_igbt")
        if v:
            p = (v["x"], v["y"])
            if len(st.session_state.calib_igbt) < 2:
                if not st.session_state.calib_igbt or p != st.session_state.calib_igbt[-1]:
                    st.session_state.calib_igbt.append(p); st.rerun()
            else:
                target = st.session_state.igbt_obj.raw_pts_tmin if tm_igbt.startswith(str(tc1)) else st.session_state.igbt_obj.raw_pts_tmax
                if not target or p != target[-1]: target.append(p)
    with c2:
        xm, ym = st.number_input("IGBT V(V)最大", 5.0), st.number_input("IGBT I(A)最大", 800.0)
        if len(st.session_state.calib_igbt) == 2:
            p0, pm = st.session_state.calib_igbt
            sx, sy = xm/(pm[0]-p0[0]), ym/(pm[1]-p0[1])
            is_min = tm_igbt.startswith(str(tc1))
            raw = st.session_state.igbt_obj.raw_pts_tmin if is_min else st.session_state.igbt_obj.raw_pts_tmax
            real = [((px-p0[0])*sx, (py-p0[1])*sy) for px, py in raw]
            if is_min: st.session_state.igbt_obj.real_pts_tmin = real
            else: st.session_state.igbt_obj.real_pts_tmax = real
            if st.button(f"🚀 擬合 IGBT"):
                cur = st.session_state.igbt_obj.real_pts_tmin if is_min else st.session_state.igbt_obj.real_pts_tmax
                vd, id = np.array([p[0] for p in cur]), np.array([p[1] for p in cur])
                z = np.polyfit(id, vd, 1)
                if is_min: st.session_state.igbt_obj.fit_tmin = [z[1], z[0]]
                else: st.session_state.igbt_obj.fit_tmax = [z[1], z[0]]
                st.success(f"{tm_igbt} 擬合成功")

# --- C. FWD 導通損區域 ---
st.divider()
st.header("3️⃣ FWD 導通特性建模 (Method Con1)")
tm_fwd = st.radio("FWD 標定溫度點", [f"{tc1}°C", f"{tc2}°C"], horizontal=True, key="tm_fwd")
up_fwd = st.file_uploader("上傳 FWD v-i 特性圖", type=["png", "jpg"], key="up_fwd")
if up_fwd:
    img = Image.open(up_fwd)
    c1, c2 = st.columns([2, 1])
    with c1:
        if len(st.session_state.calib_fwd) < 2: st.warning("👉 Step 1: 標定原點與最大點。")
        v = streamlit_image_coordinates(img, key="click_fwd")
        if v:
            p = (v["x"], v["y"])
            if len(st.session_state.calib_fwd) < 2:
                if not st.session_state.calib_fwd or p != st.session_state.calib_fwd[-1]:
                    st.session_state.calib_fwd.append(p); st.rerun()
            else:
                target = st.session_state.fwd_obj.raw_pts_tmin if tm_fwd.startswith(str(tc1)) else st.session_state.fwd_obj.raw_pts_tmax
                if not target or p != target[-1]: target.append(p)
    with c2:
        xm, ym = st.number_input("FWD V(V)最大", 5.0, key="fx"), st.number_input("FWD I(A)最大", 800.0, key="fy")
        if len(st.session_state.calib_fwd) == 2:
            p0, pm = st.session_state.calib_fwd
            sx, sy = xm/(pm[0]-p0[0]), ym/(pm[1]-p0[1])
            is_min = tm_fwd.startswith(str(tc1))
            raw = st.session_state.fwd_obj.raw_pts_tmin if is_min else st.session_state.fwd_obj.raw_pts_tmax
            real = [((px-p0[0])*sx, (py-p0[1])*sy) for px, py in raw]
            if is_min: st.session_state.fwd_obj.real_pts_tmin = real
            else: st.session_state.fwd_obj.real_pts_tmax = real
            if st.button(f"🚀 擬合 FWD"):
                cur = st.session_state.fwd_obj.real_pts_tmin if is_min else st.session_state.fwd_obj.real_pts_tmax
                vd, id = np.array([p[0] for p in cur]), np.array([p[1] for p in cur])
                z = np.polyfit(id, vd, 1)
                if is_min: st.session_state.fwd_obj.fit_tmin = [z[1], z[0]]
                else: st.session_state.fwd_obj.fit_tmax = [z[1], z[0]]
                st.success(f"{tm_fwd} 擬合成功")

# ==========================================
# 5. 結果可視化 (修復版)
# ==========================================
st.divider()
st.header("📊 擬合結果視覺化")
res_c1, res_c2, res_c3 = st.columns(3)

with res_c1:
    st.subheader("切換能量擬合 (mJ)")
    fig_sw, ax_sw = plt.subplots()
    xi = np.linspace(0, 1000, 100)
    for n, o in st.session_state.sw_curves.items():
        if o.params is not None: ax_sw.plot(xi, [o.get_val(x) for x in xi], label=n)
    ax_sw.set_xlabel("Current (A)"); ax_sw.legend(); st.pyplot(fig_sw)

with res_c2:
    st.subheader("IGBT Vce 特性")
    fig_igbt, ax_igbt = plt.subplots()
    ii = np.linspace(0, 800, 100)
    for t, p in [(f"{tc1}°C", st.session_state.igbt_obj.fit_tmin), (f"{tc2}°C", st.session_state.igbt_obj.fit_tmax)]:
        if p is not None: ax_igbt.plot(p[0]+p[1]*ii, ii, label=t)
    ax_igbt.set_xlabel("Voltage (V)"); ax_igbt.set_ylabel("Current (A)"); ax_igbt.legend(); st.pyplot(fig_igbt)

with res_c3:
    st.subheader("FWD Vf 特性")
    fig_fwd, ax_fwd = plt.subplots()
    for t, p in [(f"{tc1}°C", st.session_state.fwd_obj.fit_tmin), (f"{tc2}°C", st.session_state.fwd_obj.fit_tmax)]:
        if p is not None: ax_fwd.plot(p[0]+p[1]*ii, ii, label=t)
    ax_fwd.set_xlabel("Voltage (V)"); ax_fwd.set_ylabel("Current (A)"); ax_fwd.legend(); st.pyplot(fig_fwd)

# ==========================================
# 6. 方程式 (19) 即時分析器
# ==========================================
st.divider()
st.header("🔍 全域損耗分析器 (Eq 19)")
if st.session_state.igbt_obj.fit_tmax and st.session_state.fwd_obj.fit_tmax:
    c1, c2, c3, c4 = st.columns(4)
    tj_c = c1.number_input("接面溫度 Tj (°C)", value=100.0)
    i_pk = c2.number_input("峰值電流 I_peak (A)", value=400.0)
    fsw  = c3.number_input("切換頻率 f_sw (Hz)", value=10000.0)
    
    # 自動解算 RMS 與 平均電流[cite: 1]
    irms = i_pk / 2
    iavg = i_pk / np.pi
    st.info(f"💡 正弦波解算：$I_{{rms}} = {irms:.2f}$A, $I_{{avg}} = {iavg:.2f}$A")
    
    # 計算導通損 (Eq 19)[cite: 1]
    pcon_igbt = st.session_state.igbt_obj.calc_pcon(tj_c, irms, iavg)
    pcon_fwd  = st.session_state.fwd_obj.calc_pcon(tj_c, irms, iavg)
    
    # 計算切換損 (模式 B 積分)[cite: 1]
    psw_total = 0.0
    for n, o in st.session_state.sw_curves.items():
        if o.params is not None:
            e_avg = np.mean([o.get_val(i_pk * np.sin(t)) for t in np.linspace(0, np.pi, 100)])
            psw_total += e_avg * fsw * 1e-3
            
    st.metric("IGBT 導通損耗", f"{pcon_igbt:.2f} W")
    st.metric("FWD 導通損耗", f"{pcon_fwd:.2f} W")
    st.metric("總切換損耗 (P_sw)", f"{psw_total:.2f} W")
    st.markdown(f"### **總預估損耗 (P_total): {psw_total + pcon_igbt + pcon_fwd:.2f} W**")
else:
    st.info("💡 指示：完成導通特性擬合後，計算器將自動啟用。")