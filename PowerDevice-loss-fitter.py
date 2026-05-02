import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Power Device Total Loss Evaluator", layout="wide")

# ==========================================
# 1. 核心類別定義
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
    """導通損耗 (Method Con1 & Eq 19)[cite: 1]"""
    def __init__(self, name):
        self.name = name
        self.t_min = 298.15 
        self.t_max = 423.15 
        self.fit_tmin = None # [Vx, Rx]
        self.fit_tmax = None # [Vx, Rx]
        self.raw_pts_tmin = []
        self.raw_pts_tmax = []
        self.real_pts_tmin = []
        self.real_pts_tmax = []

    def calc_eq19(self, tj_k, i_rms, i_avg):
        """實作方程式 (19): Tj 溫度下的導通損耗"""
        if self.fit_tmin is None or self.fit_tmax is None: return 0.0
        V1, R1 = self.fit_tmin
        V2, R2 = self.fit_tmax
        T1, T2 = self.t_min, self.t_max
        denom = T1 - T2
        # Rx(Tj)
        rx_tj = ((R2*T1 - R1*T2) / denom) + ((R1 - R2) / denom) * tj_k
        # Vx(Tj)
        vx_tj = ((V2*T1 - V1*T2) / denom) + ((V1 - V2) / denom) * tj_k
        return rx_tj * (i_rms**2) + vx_tj * i_avg

# ==========================================
# 2. Session State 狀態管理
# ==========================================
if "sw_curves" not in st.session_state: st.session_state.sw_curves = {}
if "igbt_obj" not in st.session_state: st.session_state.igbt_obj = ConductionDevice("IGBT")
if "fwd_obj" not in st.session_state: st.session_state.fwd_obj = ConductionDevice("FWD")

for k in ["calib_sw", "calib_igbt", "calib_fwd"]:
    if k not in st.session_state: st.session_state[k] = []

st.title("🚀 Power Device 綜合損耗建模與動態評估系統")

# ==========================================
# 3. 側邊欄管理區
# ==========================================
with st.sidebar:
    st.header("📊 曲線庫管理")
    sw_n = st.text_input("新增切換曲線名 (如 Eon)", "Eon_150C")
    if st.button("➕ 新增切換曲線"): st.session_state.sw_curves[sw_n] = SwitchingCurve(sw_n)
    
    st.divider()
    st.subheader("🌡️ 導通損溫度基準 (K)")
    t1 = st.number_input("T_min (例如 298.15)", value=298.15)
    t2 = st.number_input("T_max (例如 423.15)", value=423.15)
    st.session_state.igbt_obj.t_min = st.session_state.fwd_obj.t_min = t1
    st.session_state.igbt_obj.t_max = st.session_state.fwd_obj.t_max = t2

    if st.button("🔄 重置所有標定", type="primary"):
        st.session_state.calib_sw = []; st.session_state.calib_igbt = []; st.session_state.calib_fwd = []
        st.rerun()

# ==========================================
# 4. 作業區域 (同頁面分塊)
# ==========================================

# --- A. 切換損耗區域 ---
st.header("1️⃣ 切換損耗建模 (Method SW3)")
sw_list = list(st.session_state.sw_curves.keys())
if not sw_list:
    st.info("💡 指示：請在左側新增『切換曲線』。")
else:
    active_sw = st.selectbox("切換曲線對象", sw_list)
    up_sw = st.file_uploader("上傳切換能量圖 (mJ)", type=["png", "jpg"], key="up_sw")
    if up_sw:
        img = Image.open(up_sw)
        c1, c2 = st.columns([2, 1])
        with c1:
            if len(st.session_state.calib_sw) < 2: st.warning("👉 指示：點擊原點(0,0)與最大刻度點。")
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
            xm, ym = st.number_input("SW X最大(A)", 1000.0), st.number_input("SW Y最大(mJ)", 125.0)
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
t_mode_igbt = st.radio("IGBT 標定溫度", ["T_min", "T_max"], horizontal=True, key="tm_igbt")
up_igbt = st.file_uploader("上傳 IGBT v-i 圖", type=["png", "jpg"], key="up_igbt")
if up_igbt:
    img = Image.open(up_igbt)
    c1, c2 = st.columns([2, 1])
    with c1:
        if len(st.session_state.calib_igbt) < 2: st.warning("👉 指示：標定原點與最大點。")
        v = streamlit_image_coordinates(img, key="click_igbt")
        if v:
            p = (v["x"], v["y"])
            if len(st.session_state.calib_igbt) < 2:
                if not st.session_state.calib_igbt or p != st.session_state.calib_igbt[-1]:
                    st.session_state.calib_igbt.append(p); st.rerun()
            else:
                target = st.session_state.igbt_obj.raw_pts_tmin if t_mode_igbt=="T_min" else st.session_state.igbt_obj.raw_pts_tmax
                if not target or p != target[-1]: target.append(p)
    with c2:
        xm, ym = st.number_input("IGBT V最大(V)", 5.0), st.number_input("IGBT I最大(A)", 800.0)
        if len(st.session_state.calib_igbt) == 2:
            p0, pm = st.session_state.calib_igbt
            sx, sy = xm/(pm[0]-p0[0]), ym/(pm[1]-p0[1])
            raw = st.session_state.igbt_obj.raw_pts_tmin if t_mode_igbt=="T_min" else st.session_state.igbt_obj.raw_pts_tmax
            real = [((px-p0[0])*sx, (py-p0[1])*sy) for px, py in raw]
            if t_mode_igbt=="T_min": st.session_state.igbt_obj.real_pts_tmin = real
            else: st.session_state.igbt_obj.real_pts_tmax = real
            if st.button(f"🚀 擬合 IGBT-{t_mode_igbt}"):
                cur_real = st.session_state.igbt_obj.real_pts_tmin if t_mode_igbt=="T_min" else st.session_state.igbt_obj.real_pts_tmax
                vd, id = np.array([p[0] for p in cur_real]), np.array([p[1] for p in cur_real])
                z = np.polyfit(id, vd, 1) # v = Rx*i + Vx
                if t_mode_igbt=="T_min": st.session_state.igbt_obj.fit_tmin = [z[1], z[0]]
                else: st.session_state.igbt_obj.fit_tmax = [z[1], z[0]]
                st.success(f"{t_mode_igbt} 擬合成功")

# --- C. FWD 導通損區域 ---
st.divider()
st.header("3️⃣ FWD 導通特性建模 (Method Con1)")
t_mode_fwd = st.radio("FWD 標定溫度", ["T_min", "T_max"], horizontal=True, key="tm_fwd")
up_fwd = st.file_uploader("上傳 FWD v-i 圖", type=["png", "jpg"], key="up_fwd")
if up_fwd:
    img = Image.open(up_fwd)
    c1, c2 = st.columns([2, 1])
    with c1:
        if len(st.session_state.calib_fwd) < 2: st.warning("👉 指示：標定原點與最大點。")
        v = streamlit_image_coordinates(img, key="click_fwd")
        if v:
            p = (v["x"], v["y"])
            if len(st.session_state.calib_fwd) < 2:
                if not st.session_state.calib_fwd or p != st.session_state.calib_fwd[-1]:
                    st.session_state.calib_fwd.append(p); st.rerun()
            else:
                target = st.session_state.fwd_obj.raw_pts_tmin if t_mode_fwd=="T_min" else st.session_state.fwd_obj.raw_pts_tmax
                if not target or p != target[-1]: target.append(p)
    with c2:
        xm, ym = st.number_input("FWD V最大(V)", 5.0, key="fx"), st.number_input("FWD I最大(A)", 800.0, key="fy")
        if len(st.session_state.calib_fwd) == 2:
            p0, pm = st.session_state.calib_fwd
            sx, sy = xm/(pm[0]-p0[0]), ym/(pm[1]-p0[1])
            raw = st.session_state.fwd_obj.raw_pts_tmin if t_mode_fwd=="T_min" else st.session_state.fwd_obj.raw_pts_tmax
            real = [((px-p0[0])*sx, (py-p0[1])*sy) for px, py in raw]
            if t_mode_fwd=="T_min": st.session_state.fwd_obj.real_pts_tmin = real
            else: st.session_state.fwd_obj.real_pts_tmax = real
            if st.button(f"🚀 擬合 FWD-{t_mode_fwd}"):
                cur_real = st.session_state.fwd_obj.real_pts_tmin if t_mode_fwd=="T_min" else st.session_state.fwd_obj.real_pts_tmax
                vd, id = np.array([p[0] for p in cur_real]), np.array([p[1] for p in cur_real])
                z = np.polyfit(id, vd, 1)
                if t_mode_fwd=="T_min": st.session_state.fwd_obj.fit_tmin = [z[1], z[0]]
                else: st.session_state.fwd_obj.fit_tmax = [z[1], z[0]]
                st.success(f"{t_mode_fwd} 擬合成功")

# ==========================================
# 5. 結果可視化 (分別呈現)
# ==========================================
st.divider()
st.header("📊 擬合結果視覺化")
res_c1, res_c2, res_c3 = st.columns(3)

with res_c1:
    st.subheader("切換損耗擬合結果 (mJ)")
    fig, ax = plt.subplots()
    xi = np.linspace(0, 1000, 100)
    for n, o in st.session_state.sw_curves.items():
        if o.params is not None: ax.plot(xi, [o.get_val(x) for x in xi], label=n)
    ax.set_xlabel("Current (A)"); ax.legend(); st.pyplot(fig)

with res_c2:
    st.subheader("IGBT 導通特性 (V-I)")
    fig, ax = plt.subplots()
    ii = np.linspace(0, 800, 100)
    for t, p in [("Tmin", st.session_state.igbt_obj.fit_tmin), ("Tmax", st.session_state.igbt_obj.fit_tmax)]:
        if p: ax.plot(p[0]+p[1]*ii, ii, label=t)
    ax.set_xlabel("Voltage (V)"); ax.set_ylabel("Current (A)"); ax.legend(); st.pyplot(fig)

with res_c3:
    st.subheader("FWD 導通特性 (V-I)")
    fig, ax = plt.subplots()
    for t, p in [("Tmin", st.session_state.fwd_obj.fit_tmin), ("Tmax", st.session_state.fwd_obj.fit_tmax)]:
        if p: ax.plot(p[0]+p[1]*ii, ii, label=t)
    ax.set_xlabel("Voltage (V)"); ax.set_ylabel("Current (A)"); ax.legend(); st.pyplot(fig)

# ==========================================
# 6. 全域損耗計算器 (Eq 19)
# ==========================================
st.divider()
st.header("🔍 方程式 (19) 損耗即時評估器")
if st.session_state.igbt_obj.fit_tmax and st.session_state.fwd_obj.fit_tmax:
    c1, c2, c3 = st.columns(3)
    tj = c1.number_input("工作接面溫度 Tj (K)", value=398.15)
    irms = c2.number_input("RMS 電流 IX (A)", value=400.0)
    iavg = c3.number_input("平均電流 IX,ave (A)", value=250.0)
    
    p_igbt = st.session_state.igbt_obj.calc_eq19(tj, irms, iavg)
    p_fwd = st.session_state.fwd_obj.calc_eq19(tj, irms, iavg)
    st.metric("IGBT 導通損", f"{p_igbt:.2f} W")
    st.metric("FWD 導通損", f"{p_fwd:.2f} W")
    st.metric("總導通損耗", f"{p_igbt+p_fwd:.2f} W")
else:
    st.info("💡 指示：完成 IGBT/FWD 之 Tmin/Tmax 擬合後，計算器將自動啟用。")