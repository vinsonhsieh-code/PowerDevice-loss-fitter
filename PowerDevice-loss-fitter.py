# ==============================================================================
# 專案名稱：Power Device Loss Fitter & Thermal Feedback Solver
# 版本編號：v1.0 (Stable Release)
# 更新日期：2026-05-02
# 核心特性：
#   1. [Method SW3] 切換損耗二階多項式擬合 (A, B, C 係數解算)
#   2. [Method Con1] 導通損耗線性化建模 (Vx, Rx 參數提取)
#   3. [Equation 19] 基於 Tj 變化之導通損耗動態補償算法
#   4. [Figure 8] 熱平衡自動疊代器 (Thermal Feedback Loop)
#   5. [Figure 9] 橋臂電流鑑別與 IGBT/FWD 損耗分離邏輯
# ==============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

# 設定 Streamlit 頁面配置
st.set_page_config(page_title="Power Device Loss Fitter - v1.0", layout="wide")

# ==========================================
# 1. 核心模型定義 (Core Physics Models)
# ==========================================
class SwitchingCurve:
    """ 實作 Method SW3: 將切換能量擬合為電流的二階函數 E = A + Bi + Ci^2 """
    def __init__(self, name):
        self.name = name
        self.params = None  # 儲存 [A, B, C] 擬合係數
        self.raw_pts = []   # 像素座標
        self.real_pts = []  # 物理值座標 (A, mJ)

    def get_val(self, i):
        if self.params is None: return 0.0
        i_abs = abs(i)
        return self.params[0] + self.params[1]*i_abs + self.params[2]*(i_abs**2)

class ConductionDevice:
    """ 實作 Method Con1: 提取元件閾值電壓 (Vx) 與動態電阻 (Rx) """
    def __init__(self, name):
        self.name = name
        self.t_min_c, self.t_max_c = 25.0, 150.0 # 預設溫度基準
        self.fit_tmin, self.fit_tmax = None, None # 儲存不同溫度下的 [Vx, Rx]
        self.raw_pts_tmin, self.raw_pts_tmax = [], []
        self.real_pts_tmin, self.real_pts_tmax = [], []

    def get_eq19_params(self, tj_c):
        """ 依據論文 Eq 17, 18 執行線性溫度插值，解算當前 Tj 下的參數 """
        if self.fit_tmin is None or self.fit_tmax is None: return None, None
        V1, R1 = self.fit_tmin
        V2, R2 = self.fit_tmax
        T1, T2 = self.t_min_c + 273.15, self.t_max_c + 273.15
        Tj = tj_c + 273.15
        denom = T1 - T2
        # Tj 溫度下的 Rx
        rx_tj = ((R2*T1 - R1*T2) / denom) + ((R1 - R2) / denom) * Tj
        # Tj 溫度下的 Vx
        vx_tj = ((V2*T1 - V1*T2) / denom) + ((V1 - V2) / denom) * Tj
        return vx_tj, rx_tj

    def calc_pcon(self, tj_c, i_rms, i_avg):
        """ 實作 Eq 19: 結合 Offset 損耗 (Vx) 與 交流動態損耗 (Rx) """
        vx, rx = self.get_eq19_params(tj_c)
        if vx is None: return 0.0
        return rx * (i_rms**2) + vx * i_avg

# ==========================================
# 2. Session State 管理 (狀態保持)
# ==========================================
if "sw_curves" not in st.session_state: st.session_state.sw_curves = {}
if "igbt_obj" not in st.session_state: st.session_state.igbt_obj = ConductionDevice("IGBT")
if "fwd_obj" not in st.session_state: st.session_state.fwd_obj = ConductionDevice("FWD")
for k in ["calib_sw", "calib_igbt", "calib_fwd"]:
    if k not in st.session_state: st.session_state[k] = []

st.title("🚀 Power Device 綜合損耗建模與熱反饋分析系統 (v1.0)")

# ==========================================
# 3. Sidebar (全域設定與曲線庫)
# ==========================================
with st.sidebar:
    st.header("📊 曲線管理與 SOP 設定")
    sw_n = st.text_input("新增切換曲線 (如 Eon_150C)", "Eon_Test")
    if st.button("➕ 新增曲線"): st.session_state.sw_curves[sw_n] = SwitchingCurve(sw_n)
    st.divider()
    st.subheader("🌡️ 導通損溫度基準 (°C)")
    tc1 = st.number_input("T_min (通常為 25°C)", value=25.0)
    tc2 = st.number_input("T_max (通常為 150°C)", value=150.0)
    st.session_state.igbt_obj.t_min_c = st.session_state.fwd_obj.t_min_c = tc1
    st.session_state.igbt_obj.t_max_c = st.session_state.fwd_obj.t_max_c = tc2
    if st.button("🔄 重置所有標定數據"):
        st.session_state.calib_sw = []; st.session_state.calib_igbt = []; st.session_state.calib_fwd = []
        st.rerun()

# ==========================================
# 4. 三大擬合作業區 (Data Fitting Zones)
# ==========================================

# --- A. 切換損耗擬合 ---
st.header("1️⃣ 切換損耗建模 (Method SW3)")
sw_list = list(st.session_state.sw_curves.keys())
if not sw_list: st.info("請在左側 Sidebar 新增曲線。")
else:
    act_sw = st.selectbox("選擇當前編輯曲線", sw_list)
    up_sw = st.file_uploader("上傳 Datasheet 切換能量圖 (mJ)", type=["png", "jpg"], key="u_sw")
    if up_sw:
        img = Image.open(up_sw)
        c1, c2 = st.columns([2, 1])
        with c1:
            if len(st.session_state.calib_sw) < 2: st.warning("👉 Step 1: 標定座標原點 (0,0) 與最大點。")
            v = streamlit_image_coordinates(img, key="c_sw")
            if v:
                p = (v["x"], v["y"])
                if len(st.session_state.calib_sw) < 2:
                    if not st.session_state.calib_sw or p != st.session_state.calib_sw[-1]:
                        st.session_state.calib_sw.append(p); st.rerun()
                else:
                    obj = st.session_state.sw_curves[act_sw]
                    if not obj.raw_pts or p != obj.raw_pts[-1]: obj.raw_pts.append(p)
        with c2:
            xm, ym = st.number_input("SW X Max (A)", 1000.0), st.number_input("SW Y Max (mJ)", 125.0)
            if len(st.session_state.calib_sw) == 2:
                p0, pm = st.session_state.calib_sw
                sx, sy = xm/(pm[0]-p0[0]), ym/(pm[1]-p0[1])
                obj = st.session_state.sw_curves[act_sw]
                obj.real_pts = [((px-p0[0])*sx, (py-p0[1])*sy) for px, py in obj.raw_pts]
                if st.button(f"🚀 擬合曲線 {act_sw}"):
                    xd, yd = np.array([p[0] for p in obj.real_pts]), np.array([p[1] for p in obj.real_pts])
                    popt, _ = curve_fit(lambda i,A,B,C: A+B*i+C*i**2, xd, yd)
                    obj.params = popt; st.success(f"{act_sw} 擬合完成 [A, B, C] = {popt}")

# --- B. IGBT 導通損擬合 ---
st.divider()
st.header("2️⃣ IGBT 導通特性建模 (Method Con1)")
tm_i = st.radio("IGBT 標定溫度", [f"{tc1}°C", f"{tc2}°C"], horizontal=True, key="t_i")
up_i = st.file_uploader("上傳 IGBT v-i 圖", type=["png", "jpg"], key="u_i")
if up_i:
    img = Image.open(up_i)
    c1, c2 = st.columns([2, 1])
    with c1:
        v = streamlit_image_coordinates(img, key="c_i")
        if v:
            p = (v["x"], v["y"])
            if len(st.session_state.calib_igbt) < 2:
                if not st.session_state.calib_igbt or p != st.session_state.calib_igbt[-1]:
                    st.session_state.calib_igbt.append(p); st.rerun()
            else:
                target = st.session_state.igbt_obj.raw_pts_tmin if tm_i.startswith(str(tc1)) else st.session_state.igbt_obj.raw_pts_tmax
                if not target or p != target[-1]: target.append(p)
    with c2:
        xm, ym = st.number_input("IGBT V Max (V)", 5.0), st.number_input("IGBT I Max (A)", 800.0)
        if len(st.session_state.calib_igbt) == 2:
            p0, pm = st.session_state.calib_igbt
            sx, sy = xm/(pm[0]-p0[0]), ym/(pm[1]-p0[1])
            is_min = tm_i.startswith(str(tc1))
            raw = st.session_state.igbt_obj.raw_pts_tmin if is_min else st.session_state.igbt_obj.raw_pts_tmax
            real = [((px-p0[0])*sx, (py-p0[1])*sy) for px, py in raw]
            if is_min: st.session_state.igbt_obj.real_pts_tmin = real
            else: st.session_state.igbt_obj.real_pts_tmax = real
            if st.button("🚀 執行 IGBT 擬合"):
                cur = st.session_state.igbt_obj.real_pts_tmin if is_min else st.session_state.igbt_obj.real_pts_tmax
                vd, id = np.array([p[0] for p in cur]), np.array([p[1] for p in cur])
                z = np.polyfit(id, vd, 1) # V = Rx*I + Vx
                if is_min: st.session_state.igbt_obj.fit_tmin = [z[1], z[0]]
                else: st.session_state.igbt_obj.fit_tmax = [z[1], z[0]]
                st.success(f"IGBT {tm_i} 擬合成功 [Vx, Rx]")

# --- C. FWD 導通損擬合 ---
st.divider()
st.header("3️⃣ FWD 導通特性建模 (Method Con1)")
tm_f = st.radio("FWD 標定溫度", [f"{tc1}°C", f"{tc2}°C"], horizontal=True, key="t_f")
up_f = st.file_uploader("上傳 FWD順向圖", type=["png", "jpg"], key="u_f")
if up_f:
    img = Image.open(up_f)
    c1, c2 = st.columns([2, 1])
    with c1:
        v = streamlit_image_coordinates(img, key="c_f")
        if v:
            p = (v["x"], v["y"])
            if len(st.session_state.calib_fwd) < 2:
                if not st.session_state.calib_fwd or p != st.session_state.calib_fwd[-1]:
                    st.session_state.calib_fwd.append(p); st.rerun()
            else:
                target = st.session_state.fwd_obj.raw_pts_tmin if tm_f.startswith(str(tc1)) else st.session_state.fwd_obj.raw_pts_tmax
                if not target or p != target[-1]: target.append(p)
    with c2:
        xm, ym = st.number_input("FWD V Max (V)", 5.0, key="fx"), st.number_input("FWD I Max (A)", 800.0, key="fy")
        if len(st.session_state.calib_fwd) == 2:
            p0, pm = st.session_state.calib_fwd
            sx, sy = xm/(pm[0]-p0[0]), ym/(pm[1]-p0[1])
            is_min = tm_f.startswith(str(tc1))
            raw = st.session_state.fwd_obj.raw_pts_tmin if is_min else st.session_state.fwd_obj.raw_pts_tmax
            real = [((px-p0[0])*sx, (py-p0[1])*sy) for px, py in raw]
            if is_min: st.session_state.fwd_obj.real_pts_tmin = real
            else: st.session_state.fwd_obj.real_pts_tmax = real
            if st.button("🚀 執行 FWD 擬合"):
                cur = st.session_state.fwd_obj.real_pts_tmin if is_min else st.session_state.fwd_obj.real_pts_tmax
                vd, id = np.array([p[0] for p in cur]), np.array([p[1] for p in cur])
                z = np.polyfit(id, vd, 1)
                if is_min: st.session_state.fwd_obj.fit_tmin = [z[1], z[0]]
                else: st.session_state.fwd_obj.fit_tmax = [z[1], z[0]]
                st.success(f"FWD {tm_f} 擬合成功 [Vx, Rx]")

# ==========================================
# 5. 分析器：熱平衡疊代器 (Thermal Loop)
# ==========================================
st.divider()
st.header("📊 擬合結果視覺化與全域熱分析")

# A. 靜態數據繪製
res_c1, res_c2, res_c3 = st.columns(3)
with res_c1:
    st.caption("切換能量擬合結果 (mJ)")
    fig_s, ax_s = plt.subplots(figsize=(5,3.5))
    xi = np.linspace(0, 1000, 100)
    for n, o in st.session_state.sw_curves.items():
        if o.params is not None: ax_s.plot(xi, [o.get_val(x) for x in xi], label=n)
    ax_s.set_xlabel("Current (A)"); ax_s.legend(); st.pyplot(fig_s)
with res_c2:
    st.caption("IGBT 導通特性 (Vce)")
    fig_i, ax_i = plt.subplots(figsize=(5,3.5))
    ii = np.linspace(0, 800, 100)
    for t, p in [(f"{tc1}°C", st.session_state.igbt_obj.fit_tmin), (f"{tc2}°C", st.session_state.igbt_obj.fit_tmax)]:
        if p is not None: ax_i.plot(p[0]+p[1]*ii, ii, label=t)
    ax_i.set_xlabel("Voltage (V)"); ax_i.set_ylabel("Current (A)"); ax_i.legend(); st.pyplot(fig_i)
with res_c3:
    st.caption("FWD 順向特性 (Vf)")
    fig_f, ax_f = plt.subplots(figsize=(5,3.5))
    for t, p in [(f"{tc1}°C", st.session_state.fwd_obj.fit_tmin), (f"{tc2}°C", st.session_state.fwd_obj.fit_tmax)]:
        if p is not None: ax_f.plot(p[0]+p[1]*ii, ii, label=t)
    ax_f.set_xlabel("Voltage (V)"); ax_f.set_ylabel("Current (A)"); ax_f.legend(); st.pyplot(fig_f)

st.divider()
st.subheader("🔍 全域損耗解算器 (基於 Figure 8 熱反饋流程)")

if st.session_state.igbt_obj.fit_tmax and st.session_state.fwd_obj.fit_tmax:
    # --- 參數輸入區 ---
    with st.container():
        input_c1, input_c2, input_c3 = st.columns(3)
        tj_init = input_c1.number_input("環境/初始溫度 Ta (°C)", value=25.0)
        i_pk = input_c2.number_input("峰值電流 I_peak (A)", value=400.0)
        fsw  = input_c3.number_input("切換頻率 f_sw (Hz)", value=15000.0)
    
    with st.container():
        th_c1, th_c2, th_c3 = st.columns(3)
        rth_igbt = th_c1.number_input("IGBT 熱阻 Rth(j-c) (K/W)", value=0.1, step=0.01)
        rth_fwd = th_c2.number_input("FWD 熱阻 Rth(j-c) (K/W)", value=0.15, step=0.01)
        epsilon = th_c3.number_input("收斂誤差 ε (°C)", value=0.1, step=0.1)
    
    # --- 熱反饋疊代核心 (Thermal Loop Implementation) ---
    irms, iavg = i_pk / 2, i_pk / np.pi # 半波正弦指標
    tj_loop = tj_init
    iteration_history = []
    converged = False

    for i in range(20): # Max 20 iters
        pcon_igbt_curr = st.session_state.igbt_obj.calc_pcon(tj_loop, irms, iavg)
        pcon_fwd_curr = st.session_state.fwd_obj.calc_pcon(tj_loop, irms, iavg)
        psw_curr = 0.0
        for n, o in st.session_state.sw_curves.items():
            if o.params is not None:
                # 弦波週期內 SW 損耗積分 (Method SW3)
                e_avg = np.mean([o.get_val(i_pk * np.sin(t)) for t in np.linspace(0, np.pi, 100)])
                psw_curr += e_avg * fsw * 1e-3
        
        ploss_total = pcon_igbt_curr + pcon_fwd_curr + psw_curr
        # 更新溫度 Eq 27: Tj = Ploss * Rth + Ta
        tj_new = ploss_total * (rth_igbt + rth_fwd) + tj_init
        
        iteration_history.append({
            "疊代": i + 1, "Tj_in": f"{tj_loop:.2f}", "Ploss": f"{ploss_total:.2f}", 
            "Tj_out": f"{tj_new:.2f}", "Err": f"{abs(tj_new - tj_loop):.3f}"
        })
        
        if abs(tj_new - tj_loop) <= epsilon:
            tj_loop = tj_new; converged = True; break
        tj_loop = tj_new

    # --- 結果展示 ---
    st.markdown("---")
    st.info("🔄 **熱平衡疊代日誌**")
    st.table(pd.DataFrame(iteration_history))
    
    # 公式展示與對齊
    st.markdown("---")
    r1, r2 = st.columns([1, 2.5])
    r1.metric("最終平衡 Tj", f"{tj_loop:.2f} °C")
    r2.latex(r"T_j = P_{loss} \cdot (R_{th,IGBT} + R_{th,FWD}) + T_a")
    
    r1.metric("IGBT 導通損耗", f"{pcon_igbt_curr:.2f} W")
    r2.latex(r"P_{con,IGBT} = R_{ce}(T_j) \cdot I_{rms}^2 + V_{ce}(T_j) \cdot I_{avg}")

    r1.metric("FWD 導通損耗", f"{pcon_fwd_curr:.2f} W")
    r2.latex(r"P_{con,FWD} = R_{f}(T_j) \cdot I_{rms}^2 + V_{f}(T_j) \cdot I_{avg}")
    
    r1.metric("總切換損耗 (P_sw)", f"{psw_curr:.2f} W")
    r2.latex(r"P_{sw} = \left( \frac{1}{\pi} \int_{0}^{\pi} E_{sw}(i(\theta)) d\theta \right) \cdot f_{sw}")

    st.success(f"### **⚡ 穩態總損耗 (P_total): {ploss_total:.2f} W**")
    
else:
    st.info("💡 請先完成所有曲線擬合作業後，系統將自動啟動熱反饋分析。")