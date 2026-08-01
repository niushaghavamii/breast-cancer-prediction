"""
اپلیکیشن پیش‌بینی خوش‌خیم/بدخیم بودن تومور سرطان پستان
مدل: sklearn pipeline  |  رابط کاربری: Streamlit  |  تفسیرپذیری: Permutation Importance + Plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer
from sklearn.inspection import permutation_importance

st.set_page_config(
    page_title="سرطان سینه",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/Vazirmatn-font-face.css');

html, body, [class*="css"], .stApp, .stMarkdown, .stButton, .stSlider, .stDataFrame,
.stTextInput, .stSelectbox, div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
    font-family: 'Vazirmatn', Tahoma, sans-serif !important;
}

.stApp { direction: rtl; }
section[data-testid="stSidebar"] { direction: rtl; }
div[data-testid="stMarkdownContainer"] { text-align: right; }
.stSlider { direction: ltr; }
.stSlider label { direction: rtl; text-align: right; display: block; }

.main-title {
    font-size: 2.3rem; font-weight: 800; color: #1e293b;
    text-align: center; margin-bottom: 0.1rem;
}
.sub-title {
    font-size: 1.05rem; color: #64748b;
    text-align: center; margin-bottom: 1.8rem;
}
.info-card {
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    border: 1px solid #e2e8f0; border-radius: 16px;
    padding: 1.3rem 1.5rem; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.info-card h3 { margin: 0; font-size: 1.7rem; color: #1e293b; }
.info-card p { margin: 0.3rem 0 0 0; color: #64748b; font-size: 0.9rem; }
.result-card-benign {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
    border: 2px solid #10b981; border-radius: 18px; padding: 2rem;
    text-align: center; box-shadow: 0 4px 14px rgba(16,185,129,0.15);
}
.result-card-malignant {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border: 2px solid #ef4444; border-radius: 18px; padding: 2rem;
    text-align: center; box-shadow: 0 4px 14px rgba(239,68,68,0.15);
}
.result-title { font-size: 1.9rem; font-weight: 800; margin: 0.3rem 0; }
.result-sub { font-size: 1.05rem; color: #475569; margin: 0; }
.section-header {
    font-size: 1.35rem; font-weight: 700; color: #1e293b;
    border-right: 5px solid #4f46e5; padding-right: 0.7rem;
    margin: 1.6rem 0 0.8rem 0;
}
div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
    color: white; font-weight: 700; font-size: 1.05rem;
    border-radius: 10px; padding: 0.7rem; border: none; transition: 0.2s;
}
div.stButton > button:hover {
    background: linear-gradient(135deg, #4338ca 0%, #3730a3 100%);
    box-shadow: 0 4px 12px rgba(79,70,229,0.35);
}
.stExpander { direction: rtl; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🩺 سامانه هوشمند پیش‌بینی سرطان سینه</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">مبتنی بر دیتاست Wisconsin Breast Cancer و مدل یادگیری‌ماشین</div>', unsafe_allow_html=True)

FEATURE_FA = {
    'mean radius': 'شعاع (میانگین)', 'mean texture': 'بافت (میانگین)',
    'mean perimeter': 'محیط (میانگین)', 'mean area': 'مساحت (میانگین)',
    'mean smoothness': 'صافی سطح (میانگین)', 'mean compactness': 'فشردگی (میانگین)',
    'mean concavity': 'تقعر (میانگین)', 'mean concave points': 'نقاط مقعر (میانگین)',
    'mean symmetry': 'تقارن (میانگین)', 'mean fractal dimension': 'بعد فراکتالی (میانگین)',
    'radius error': 'خطای شعاع', 'texture error': 'خطای بافت',
    'perimeter error': 'خطای محیط', 'area error': 'خطای مساحت',
    'smoothness error': 'خطای صافی سطح', 'compactness error': 'خطای فشردگی',
    'concavity error': 'خطای تقعر', 'concave points error': 'خطای نقاط مقعر',
    'symmetry error': 'خطای تقارن', 'fractal dimension error': 'خطای بعد فراکتالی',
    'worst radius': 'شعاع (بدترین حالت)', 'worst texture': 'بافت (بدترین حالت)',
    'worst perimeter': 'محیط (بدترین حالت)', 'worst area': 'مساحت (بدترین حالت)',
    'worst smoothness': 'صافی سطح (بدترین حالت)', 'worst compactness': 'فشردگی (بدترین حالت)',
    'worst concavity': 'تقعر (بدترین حالت)', 'worst concave points': 'نقاط مقعر (بدترین حالت)',
    'worst symmetry': 'تقارن (بدترین حالت)', 'worst fractal dimension': 'بعد فراکتالی (بدترین حالت)',
}

@st.cache_resource
def load_trained_model():
    return joblib.load('joblib.model_best')

@st.cache_data
def load_reference_data():
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    X = df.drop(columns=['target'])
    return X, df

@st.cache_data
def get_feature_importance(_model, X, y):
    """محاسبه اهمیت ویژگی‌ها با permutation importance"""
    result = permutation_importance(_model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    return result.importances_mean

try:
    model = load_trained_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

X_ref, full_df = load_reference_data()
feature_names = list(X_ref.columns)
y_ref = full_df['target']

if not model_loaded:
    st.error("فایل مدل 'joblib.model_best' پیدا نشد.")
    st.stop()

# کارت‌های اطلاعاتی
c1, c2, c3, c4 = st.columns(4)
benign_count = int((full_df['target'] == 1).sum())
malignant_count = int((full_df['target'] == 0).sum())

with c1:
    st.markdown(f'<div class="info-card"><h3>{len(full_df)}</h3><p>تعداد کل نمونه‌ها</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="info-card"><h3>{benign_count}</h3><p>نمونه‌های خوش‌خیم</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="info-card"><h3>{malignant_count}</h3><p>نمونه‌های بدخیم</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="info-card"><h3>۳۰</h3><p>ویژگی ورودی</p></div>', unsafe_allow_html=True)

st.write("")

# سایدبار
st.sidebar.markdown("## 📋 ویژگی‌های تومور بیمار")
st.sidebar.caption("مقادیر را برای هر ویژگی تنظیم کنید")

mean_features = [f for f in feature_names if f.startswith('mean')]
error_features = [f for f in feature_names if 'error' in f]
worst_features = [f for f in feature_names if f.startswith('worst')]

input_values = {}

def render_group(feature_list):
    for feat in feature_list:
        col_min = float(X_ref[feat].min())
        col_max = float(X_ref[feat].max())
        col_mean = float(X_ref[feat].mean())
        step = (col_max - col_min) / 200 if col_max > col_min else 0.01
        input_values[feat] = st.slider(
            FEATURE_FA.get(feat, feat),
            min_value=col_min, max_value=col_max,
            value=col_mean, step=step, key=feat
        )

with st.sidebar.expander("📏 ویژگی‌های میانگین (Mean)", expanded=True):
    render_group(mean_features)
with st.sidebar.expander("📉 ویژگی‌های خطا (Standard Error)", expanded=False):
    render_group(error_features)
with st.sidebar.expander("⚠️ ویژگی‌های بدترین حالت (Worst)", expanded=False):
    render_group(worst_features)

st.sidebar.write("")
predict_button = st.sidebar.button("🔍  اجرای پیش‌بینی")

input_df = pd.DataFrame([input_values], columns=feature_names)

if predict_button:
    with st.spinner("در حال تحلیل داده‌ها..."):
        pred_proba = model.predict_proba(input_df)[0]
        pred_class = model.predict(input_df)[0]
        is_benign = (pred_class == 1)
        confidence = pred_proba[1] if is_benign else pred_proba[0]

    st.markdown('<div class="section-header">📊 نتیجه پیش‌بینی</div>', unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1, 1.3])
    with res_col1:
        box_class = "result-card-benign" if is_benign else "result-card-malignant"
        icon = "✅" if is_benign else "⚠️"
        label_fa = "خوش‌خیم (Benign)" if is_benign else "بدخیم (Malignant)"
        st.markdown(f"""
            <div class="{box_class}">
                <div style="font-size:2.5rem;">{icon}</div>
                <div class="result-title">{label_fa}</div>
                <div class="result-sub">درصد اطمینان مدل: <b>{confidence*100:.1f}٪</b></div>
            </div>
        """, unsafe_allow_html=True)

    with res_col2:
        gauge_color = "#10b981" if is_benign else "#ef4444"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={'suffix': "٪", 'font': {'size': 42, 'family': 'Vazirmatn'}},
            title={'text': "میزان اطمینان پیش‌بینی", 'font': {'size': 16, 'family': 'Vazirmatn'}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 50], 'color': '#f1f5f9'},
                    {'range': [50, 80], 'color': '#e2e8f0'},
                    {'range': [80, 100], 'color': '#cbd5e1'},
                ],
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10),
                                 font=dict(family="Vazirmatn"))
        st.plotly_chart(fig_gauge, use_container_width=True)

    prob_fig = go.Figure(go.Bar(
        x=[pred_proba[0] * 100, pred_proba[1] * 100],
        y=["بدخیم (Malignant)", "خوش‌خیم (Benign)"],
        orientation='h',
        marker_color=["#ef4444", "#10b981"],
        text=[f"{pred_proba[0]*100:.1f}٪", f"{pred_proba[1]*100:.1f}٪"],
        textposition="outside"
    ))
    prob_fig.update_layout(
        title="توزیع احتمال بین دو کلاس",
        xaxis_title="درصد احتمال", yaxis_title="",
        font=dict(family="Vazirmatn", size=14),
        height=260, margin=dict(l=10, r=10, t=50, b=30),
        xaxis=dict(range=[0, 110])
    )
    st.plotly_chart(prob_fig, use_container_width=True)

    # تفسیرپذیری با Permutation Importance (جایگزین SHAP)
    st.markdown('<div class="section-header">🔬 تفسیر پیش‌بینی - اهمیت ویژگی‌ها</div>', unsafe_allow_html=True)
    st.write(
        "نمودار زیر نشان می‌دهد کدام ویژگی‌های تومور بیشترین تأثیر را در پیش‌بینی مدل داشته‌اند."
    )

    with st.spinner("در حال محاسبه اهمیت ویژگی‌ها..."):
        importances = get_feature_importance(model, X_ref, y_ref)

    fa_labels = [FEATURE_FA.get(f, f) for f in feature_names]
    order = np.argsort(importances)[::-1][:10]

    top_labels = [fa_labels[i] for i in order][::-1]
    top_values = [float(importances[i]) for i in order][::-1]

    # مقایسه مقدار ورودی با میانگین دیتاست
    input_vs_mean = []
    for i in order:
        feat = feature_names[i]
        diff = (input_values[feat] - float(X_ref[feat].mean())) / (float(X_ref[feat].std()) + 1e-8)
        input_vs_mean.append(diff)
    input_vs_mean = input_vs_mean[::-1]

    bar_colors = ["#ef4444" if v > 0 else "#3b82f6" for v in input_vs_mean]

    imp_col, diff_col = st.columns(2)

    with imp_col:
        imp_fig = go.Figure(go.Bar(
            x=top_values,
            y=top_labels,
            orientation='h',
            marker_color="#4f46e5",
            text=[f"{v:.3f}" for v in top_values],
            textposition="outside"
        ))
        imp_fig.update_layout(
            title="۱۰ ویژگی مهم‌تر در مدل",
            xaxis_title="اهمیت ویژگی",
            font=dict(family="Vazirmatn", size=13),
            height=460, margin=dict(l=10, r=10, t=50, b=30)
        )
        st.plotly_chart(imp_fig, use_container_width=True)

    with diff_col:
        diff_fig = go.Figure(go.Bar(
            x=input_vs_mean,
            y=top_labels,
            orientation='h',
            marker_color=bar_colors,
            text=[f"{v:+.2f}" for v in input_vs_mean],
            textposition="outside"
        ))
        diff_fig.update_layout(
            title="انحراف مقادیر بیمار از میانگین (Z-score)",
            xaxis_title="انحراف از میانگین",
            font=dict(family="Vazirmatn", size=13),
            height=460, margin=dict(l=10, r=10, t=50, b=30)
        )
        st.plotly_chart(diff_fig, use_container_width=True)

    top_feat = feature_names[order[0]]
    top_name_fa = fa_labels[order[0]]
    st.info(f"🔑 **مهم‌ترین ویژگی در این مدل:** «**{top_name_fa}**» بیشترین تأثیر را در پیش‌بینی دارد.")

    with st.expander("📥 مشاهده‌ی جزئیات کامل مقادیر واردشده"):
        display_df = input_df.T.reset_index()
        display_df.columns = ["ویژگی", "مقدار واردشده"]
        display_df["ویژگی"] = display_df["ویژگی"].map(lambda f: FEATURE_FA.get(f, f))
        st.dataframe(display_df, use_container_width=True, height=400)

else:
    st.info("👈 ویژگی‌های تومور را از نوار کناری تنظیم کرده و روی دکمه «اجرای پیش‌بینی» کلیک کنید.")

st.divider()
st.caption("ساخته‌شده با Streamlit و scikit-learn — صرفاً جهت اهداف آموزشی و پژوهشی، جایگزین تشخیص پزشکی نیست.")
        
