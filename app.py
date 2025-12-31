import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp

# --- 有意差ラベル判定用関数 ---
def get_sig_label(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

# 1. ページ構成
st.set_page_config(page_title="Scientific Stat Engine Pro", layout="wide")
st.title("🔬 Scientific Stat Engine: International Edition")
st.markdown("Automatic statistical analysis with professional reporting in Japanese and English.")

# 2. グループ管理
if 'g_count' not in st.session_state: st.session_state.g_count = 3
c1, _ = st.columns([1, 4])
with c1:
    if st.button("＋ Add Group"): st.session_state.g_count += 1
    if st.session_state.g_count > 2 and st.button("－ Remove Group"): st.session_state.g_count -= 1

st.divider()

# 3. データ入力
data_dict = {}
cols = st.columns(3)
for i in range(st.session_state.g_count):
    with cols[i % 3]:
        name = st.text_input(f"Group {i+1} Name", value=f"Group {i+1}", key=f"n{i}")
        raw = st.text_area(f"Input Data for {name} (Line separated)", key=f"d{i}", height=120)
        vals = [float(x.strip()) for x in raw.replace(',', '\n').split('\n') if x.strip()]
        if len(vals) >= 3: data_dict[name] = vals

# 4. 解析エンジン
if len(data_dict) >= 2:
    st.header("📊 Results & Analysis Log")
    
    all_normal = True
    shapiro_log = []
    for n, v in data_dict.items():
        _, p_s = stats.shapiro(v)
        all_normal &= (p_s > 0.05)
        shapiro_log.append(f"{n}(p={p_s:.4f})")
    
    _, p_lev = stats.levene(*data_dict.values())
    is_equal_var = (p_lev > 0.05)

    method = ""
    p_final = 0.0
    p_disp = ""

    # A. 2-Group Comparison
    if len(data_dict) == 2:
        gn = list(data_dict.keys())
        v1, v2 = data_dict[gn[0]], data_dict[gn[1]]
        if all_normal:
            method = "Student's t-test" if is_equal_var else "Welch's t-test"
            _, p_final = stats.ttest_ind(v1, v2, equal_var=is_equal_var)
        else:
            method = "Mann-Whitney U-test"
            _, p_final = stats.mannwhitneyu(v1, v2, alternative='two-sided')
        
        st.success(f"**Method Selected: {method}**")
        p_disp = f"{p_final:.2e}" if p_final < 0.001 else f"{p_final:.4f}"
        st.metric("P-value", p_disp)

    # B. 3+ Group Comparison
    else:
        if all_normal and is_equal_var:
            method = "One-way ANOVA + Tukey's HSD"
            _, p_anova = stats.f_oneway(*data_dict.values())
            p_final = p_anova
            st.success(f"**Method Selected: {method}**")
            p_disp = f"{p_anova:.2e}" if p_anova < 0.001 else f"{p_anova:.4f}"
            st.write(f"Overall P-value (ANOVA): **{p_disp}**")
            
            if p_anova < 0.05:
                flat_data = [v for sub in data_dict.values() for v in sub]
                labels = [n for n, sub in data_dict.items() for _ in sub]
                res = pairwise_tukeyhsd(flat_data, labels)
                df_res = pd.DataFrame(data=res._results_table.data[1:], columns=res._results_table.data[0])
                st.table(df_res)
        else:
            method = "Kruskal-Wallis test (Non-parametric)"
            _, p_kw = stats.kruskal(*data_dict.values())
            p_final = p_kw
            st.warning(f"**Method Selected: {method}**")
            p_disp = f"{p_kw:.4f}"
            st.write(f"Overall P-value (Kruskal-Wallis): **{p_disp}**")
            
            if p_kw < 0.05:
                st.write("Post-hoc (Dunn's test):")
                df_dunn = sp.posthoc_dunn(list(data_dict.values()), p_adjust='bonferroni')
                df_dunn.columns = df_dunn.index = data_dict.keys()
                st.table(df_dunn)

    with st.expander("Detailed Diagnostic Log"):
        st.write(f"・Normality (Shapiro-Wilk): {', '.join(shapiro_log)}")
        st.write(f"・Equal Variance (Levene): p = {p_lev:.4f}")

    # --- 5. レポート生成 (日英併記) ---
    st.divider()
    st.header("📝 Generated Reports")
    
    # 判定ロジック
    is_sig = (p_final < 0.05)
    
    # 日本語レポート作成
    if all_normal and is_equal_var:
        jp_reason = "データの分布が偏っておらず、バラツキも均一だったため、標準的なt検定/ANOVAを選択しました。"
        en_reason = "Since the data followed a normal distribution with equal variance, a parametric test (t-test/ANOVA) was selected."
    elif not all_normal:
        jp_reason = "データに外れ値や偏りが見られたため、順位を重視するノンパラメトリック検定を選択しました。"
        en_reason = "Due to the presence of outliers or non-normal distribution, a non-parametric test was selected for robustness."
    else:
        jp_reason = "バラツキが群間で異なっていたため、ウェルチの補正を行いました。"
        en_reason = "Due to unequal variances, Welch's correction was applied."

    jp_report = f"""【解析報告書】
1. 手法: {method}
2. 理由: {jp_reason}
3. 結果: {"有意差あり" if is_sig else "有意差なし"} (P={p_disp})
"""

    en_report = f"""【Statistical Analysis Report】
1. Method: {method}
2. Rationale: {en_reason}
3. Results: {"Significant Difference Found" if is_sig else "No Significant Difference"} (P={p_disp})
4. Conclusion: Based on the {method}, the null hypothesis was {"rejected" if is_sig else "not rejected"}.
"""

    tab_jp, tab_en = st.tabs(["日本語レポート", "English Report"])
    with tab_jp:
        st.text_area("JP Report", jp_report, height=200)
        st.download_button("レポート(JP)を保存", jp_report, "report_jp.txt")
    with tab_en:
        st.text_area("EN Report", en_report, height=200)
        st.download_button("Download Report (EN)", en_report, "report_en.txt")

else:
    st.info("Please input data for at least 2 groups.")
    # --- Footer Disclaimer for English Version ---
    st.divider()
    st.caption("【Disclaimer】")
    st.caption("""
    This tool is intended for assistive purposes in statistical analysis and data visualization. 
    While the calculations are based on reliable libraries, final interpretations and conclusions 
    must be made by the user based on professional expertise. 
    The developer assumes no responsibility for any outcomes resulting from the use of this software.
    """)
