"""
========================================================
 نظام الذكاء الاصطناعي البيئي - EnviroAI Pro
 مُعَدَّل للنشر على Render / Railway / أي استضافة سحابية
========================================================
الاستخدام المحلي:
    python enviro_ai_full.py  ثم افتح http://localhost:8050

النشر أونلاين:
    gunicorn enviro_ai_full:server
"""

import os, warnings, base64, io
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBClassifier
import shap

import dash
from dash import dcc, html, Input, Output, State

warnings.filterwarnings("ignore")

# ============================================================
# الخطوة 1: معالجة البيانات
# ============================================================
def process_data(df):
    df.fillna(df.median(numeric_only=True), inplace=True)
    df["RiskIndex"] = 0.40*df["AQI"] + 0.30*df["CO"] + 0.30*df["SMOKE"]
    bins   = [-np.inf, 50, 100, 150, 200, np.inf]
    labels = ["آمن", "متوسط", "تحذير", "حرج", "طوارئ"]
    df["RiskLevel"] = pd.cut(df["RiskIndex"], bins=bins, labels=labels)
    features = ["AQI", "CO", "SMOKE", "Temperature", "Humidity"]
    iso = IsolationForest(contamination=0.03, random_state=42)
    df["Anomaly"] = iso.fit_predict(df[features])
    return df

# ============================================================
# الخطوة 2: تدريب النموذج
# ============================================================
def train_model(df):
    features = ["AQI", "CO", "SMOKE", "Temperature", "Humidity"]
    X = df[features]
    le = LabelEncoder()
    y = le.fit_transform(df["RiskLevel"].astype(str))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", verbosity=0
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, le, X_test, y_test, acc, features

# ============================================================
# الخطوة 3: الرسوم ثلاثية الأبعاد
# ============================================================
def chart_3d_scatter(df):
    color_map = {
        "آمن":    "#2ecc71",
        "متوسط":  "#f1c40f",
        "تحذير":  "#e67e22",
        "حرج":    "#e74c3c",
        "طوارئ":  "#8e44ad",
    }
    fig = go.Figure()
    for level, color in color_map.items():
        sub = df[df["RiskLevel"] == level]
        fig.add_trace(go.Scatter3d(
            x=sub["AQI"], y=sub["CO"], z=sub["SMOKE"],
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.7),
            name=level,
        ))
    fig.update_layout(
        title="🌐 رسم ثلاثي الأبعاد: AQI × CO × SMOKE",
        scene=dict(
            xaxis_title="AQI – جودة الهواء",
            yaxis_title="CO – أول أكسيد الكربون",
            zaxis_title="SMOKE – الدخان",
            bgcolor="#0d1117",
        ),
        paper_bgcolor="#0d1117",
        font=dict(color="white"),
        legend=dict(bgcolor="#1c2333"),
    )
    return fig


def chart_3d_surface(df):
    xi = np.linspace(df["AQI"].min(), df["AQI"].max(), 60)
    yi = np.linspace(df["Humidity"].min(), df["Humidity"].max(), 60)
    XX, YY = np.meshgrid(xi, yi)
    knn = KNeighborsRegressor(n_neighbors=10)
    knn.fit(df[["AQI", "Humidity"]], df["Temperature"])
    ZZ = knn.predict(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
    fig = go.Figure(go.Surface(
        x=xi, y=yi, z=ZZ,
        colorscale="Inferno",
        colorbar=dict(title="°C"),
    ))
    fig.update_layout(
        title="🌡️ سطح ثلاثي الأبعاد: درجة الحرارة عبر AQI والرطوبة",
        scene=dict(
            xaxis_title="AQI",
            yaxis_title="الرطوبة %",
            zaxis_title="درجة الحرارة °C",
            bgcolor="#0d1117",
        ),
        paper_bgcolor="#0d1117",
        font=dict(color="white"),
    )
    return fig


def chart_3d_bar(df):
    summary = (
        df.groupby("RiskLevel", observed=True)[["AQI", "CO", "SMOKE"]]
        .mean()
        .reset_index()
    )
    colors  = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
    metrics = ["AQI", "CO", "SMOKE"]
    y_pos   = {"AQI": 0, "CO": 1, "SMOKE": 2}
    fig = go.Figure()
    for i, row in summary.iterrows():
        color = colors[i % len(colors)]
        level = str(row["RiskLevel"])
        for metric in metrics:
            val = row[metric]
            yp  = y_pos[metric]
            fig.add_trace(go.Scatter3d(
                x=[i, i], y=[yp, yp], z=[0, val],
                mode="lines",
                line=dict(color=color, width=8),
                name=level,
                showlegend=(metric == "AQI"),
            ))
            fig.add_trace(go.Scatter3d(
                x=[i], y=[yp], z=[val],
                mode="markers+text",
                marker=dict(size=6, color=color),
                text=[f"{val:.0f}"],
                textposition="top center",
                textfont=dict(size=9, color="white"),
                showlegend=False,
            ))
    fig.update_layout(
        title="📊 أعمدة 3D: متوسط القراءات لكل مستوى خطر",
        scene=dict(
            xaxis=dict(
                title="مستوى الخطر",
                tickvals=list(range(len(summary))),
                ticktext=summary["RiskLevel"].tolist(),
            ),
            yaxis=dict(title="المقياس", tickvals=[0, 1, 2], ticktext=metrics),
            zaxis_title="القيمة المتوسطة",
            bgcolor="#0d1117",
        ),
        paper_bgcolor="#0d1117",
        font=dict(color="white"),
        legend=dict(bgcolor="#1c2333"),
    )
    return fig


def chart_anomaly_timeline(df):
    fig = go.Figure()
    normal = df[df["Anomaly"] ==  1]
    anom   = df[df["Anomaly"] == -1]
    fig.add_trace(go.Scatter(
        x=normal.index, y=normal["AQI"],
        mode="lines", name="طبيعي",
        line=dict(color="#2ecc71", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=anom.index, y=anom["AQI"],
        mode="markers", name="شاذ",
        marker=dict(color="red", size=7, symbol="x"),
    ))
    fig.update_layout(
        title="⚠️ الشذوذات في مؤشر AQI عبر الزمن",
        xaxis_title="رقم القراءة",
        yaxis_title="AQI",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="white"),
    )
    return fig


def chart_shap(model, X_test, features):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test[:500])
    sv = np.array(shap_values)
    if sv.ndim == 3:       # (samples, features, classes)
        importance = np.abs(sv).mean(axis=(0, 2))
    elif sv.ndim == 2:     # (samples, features)
        importance = np.abs(sv).mean(axis=0)
    else:
        importance = np.abs(sv).mean(axis=0)
    df_shap = (
        pd.DataFrame({"feature": features, "importance": importance})
        .sort_values("importance")
    )
    fig = go.Figure(go.Bar(
        x=df_shap["importance"],
        y=df_shap["feature"],
        orientation="h",
        marker=dict(color=df_shap["importance"], colorscale="Viridis"),
    ))
    fig.update_layout(
        title="🧠 أهمية المتغيرات (SHAP)",
        xaxis_title="متوسط |SHAP value|",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="white"),
    )
    return fig


def _step_item(num, title, desc, color):
    return html.Div(
        style={"display":"flex","gap":"10px","alignItems":"flex-start","marginBottom":"8px"},
        children=[
            html.Span(num, style={"background":color,"color":"#0d1117","borderRadius":"50%",
                                   "width":"22px","height":"22px","display":"flex",
                                   "alignItems":"center","justifyContent":"center",
                                   "fontSize":"0.75rem","fontWeight":"700","flexShrink":"0"}),
            html.Div([
                html.Span(title, style={"color":color,"fontWeight":"700","fontSize":"0.88rem"}),
                html.Span(" – " + desc, style={"color":"#888","fontSize":"0.82rem"}),
            ]),
        ]
    )

# ============================================================
# الخطوة 4: لوحة تحكم Dash
# ============================================================
app = dash.Dash(__name__, title="EnviroAI Pro 🌍")

# ← هذا السطر ضروري للنشر أونلاين على Render / Railway / Gunicorn
server = app.server

app.index_string = '''<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
*{box-sizing:border-box}
body{margin:0;font-family:Cairo,Arial,sans-serif}

@keyframes progressAnim{
  0%{width:0%} 15%{width:20%} 40%{width:50%}
  70%{width:75%} 90%{width:90%} 100%{width:100%}
}
@keyframes shimmer{
  0%{background-position:-400px 0}
  100%{background-position:400px 0}
}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.35}}
@keyframes slideDown{
  from{transform:translateY(-12px);opacity:0}
  to{transform:translateY(0);opacity:1}
}
@keyframes popIn{
  0%{transform:scale(0.85);opacity:0}
  100%{transform:scale(1);opacity:1}
}

.progress-wrap{
  background:#111d2b;border-radius:50px;height:16px;
  width:100%;overflow:hidden;margin:12px 0 6px;
  border:1px solid #2ecc71;
}
.progress-fill{
  height:100%;border-radius:50px;
  background:linear-gradient(90deg,#145a32,#2ecc71,#145a32);
  background-size:400px 100%;
  animation:progressAnim 4s ease-out forwards,shimmer 1.2s linear infinite;
}
.steps-row{
  display:flex;justify-content:space-between;
  margin-top:8px;flex-wrap:wrap;gap:4px;
}
.step{color:#3a4a5a;font-size:0.8rem;transition:color 0.6s}
.step.active{color:#2ecc71;animation:pulse 1s infinite}
.step.done{color:#2ecc71}

.loading-box{
  background:#1c2333;border:1px solid #2ecc71;
  border-radius:12px;padding:18px 22px;
  margin-top:18px;animation:slideDown 0.3s ease;
}
.success-banner{
  background:linear-gradient(135deg,#1a472a,#0d2818);
  border:2px solid #2ecc71;border-radius:12px;
  padding:16px 22px;margin-top:18px;text-align:center;
  color:#2ecc71;font-size:1.1rem;font-weight:700;
  animation:popIn 0.4s ease;letter-spacing:0.4px;
}
.upload-zone{
  border:2px dashed #2ecc71 !important;
  border-radius:8px !important;
  padding:28px !important;
  cursor:pointer;
  background:#0d1117;
  margin-top:15px;
  transition:background 0.2s;
}
.upload-zone:hover{background:#111d2b !important}
</style>
</head>
<body>
{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>'''

app.layout = html.Div(
    style={
        "background":  "#0d1117",
        "minHeight":   "100vh",
        "fontFamily":  "Cairo, Arial",
        "direction":   "rtl",
    },
    children=[

        # ── رأس الصفحة ──────────────────────────────────────
        html.Div(
            style={
                "background":    "linear-gradient(135deg,#1a472a,#0d1117)",
                "padding":       "30px",
                "textAlign":     "center",
                "borderBottom":  "2px solid #2ecc71",
            },
            children=[
                html.H1(
                    "🌍 نظام الذكاء الاصطناعي البيئي – EnviroAI Pro",
                    style={"color": "#2ecc71", "margin": 0, "fontSize": "2rem"},
                ),
                html.P(
                    "رفع ملف Excel ← تحليل فوري ← رسومات ثلاثية الأبعاد ← نموذج XGBoost",
                    style={"color": "#aaa", "marginTop": "8px"},
                ),
            ],
        ),

        # ── قسم شرح الخوارزميات ─────────────────────────────
        html.Div(
            style={"padding": "30px 30px 0", "maxWidth": "1200px", "margin": "0 auto"},
            children=[
                html.H2("🧬 كيف يعمل النظام؟ — الخوارزميات المستخدمة",
                        style={"color":"#2ecc71","textAlign":"center","marginBottom":"25px","fontSize":"1.5rem"}),
                html.Div(
                    style={"display":"grid","gridTemplateColumns":"repeat(auto-fit, minmax(320px, 1fr))","gap":"20px"},
                    children=[

                        # 1 XGBoost
                        html.Div(style={"background":"#1c2333","borderRadius":"14px","padding":"22px","border":"1px solid #2ecc71","borderTop":"4px solid #2ecc71"},children=[
                            html.Div(style={"display":"flex","alignItems":"center","gap":"12px","marginBottom":"12px"},children=[
                                html.Span("🤖",style={"fontSize":"2rem"}),
                                html.Div([html.H3("XGBoost Classifier",style={"color":"#2ecc71","margin":0,"fontSize":"1.1rem"}),html.Span("تصنيف مستوى الخطر",style={"color":"#888","fontSize":"0.8rem"})]),
                            ]),
                            html.P("خوارزمية تعزيز متدرج تبني مئات الأشجار القرارية بشكل تسلسلي، كل شجرة تصحح أخطاء السابقة. تُعدّ من أقوى خوارزميات التصنيف في بيانات الجداول.",style={"color":"#ccc","fontSize":"0.9rem","lineHeight":"1.7","marginBottom":"12px"}),
                            html.Div(style={"background":"#0d1117","borderRadius":"8px","padding":"12px"},children=[
                                html.P("⚙️ الإعدادات:",style={"color":"#2ecc71","margin":"0 0 6px","fontSize":"0.85rem","fontWeight":"700"}),
                                html.Ul([html.Li("n_estimators = 300 شجرة"),html.Li("max_depth = 6 مستويات"),html.Li("learning_rate = 0.05"),html.Li("subsample = 80%")],style={"color":"#aaa","fontSize":"0.82rem","lineHeight":"1.9","margin":0,"paddingRight":"18px"}),
                            ]),
                            html.Div(style={"marginTop":"10px","background":"#0a2a1a","borderRadius":"6px","padding":"8px 12px","border":"1px solid #1a5a2a"},children=[
                                html.Span("📤 المخرج: ",style={"color":"#2ecc71","fontWeight":"700","fontSize":"0.82rem"}),
                                html.Span("آمن / متوسط / تحذير / حرج / طوارئ",style={"color":"#aaa","fontSize":"0.82rem"}),
                            ]),
                        ]),

                        # 2 Isolation Forest
                        html.Div(style={"background":"#1c2333","borderRadius":"14px","padding":"22px","border":"1px solid #e74c3c","borderTop":"4px solid #e74c3c"},children=[
                            html.Div(style={"display":"flex","alignItems":"center","gap":"12px","marginBottom":"12px"},children=[
                                html.Span("🔍",style={"fontSize":"2rem"}),
                                html.Div([html.H3("Isolation Forest",style={"color":"#e74c3c","margin":0,"fontSize":"1.1rem"}),html.Span("كشف الشذوذات",style={"color":"#888","fontSize":"0.8rem"})]),
                            ]),
                            html.P("تعزل القراءات الشاذة بناءً على مدى سهولة عزلها في شجرة عشوائية. القراءات النادرة تُعزل بخطوات أقل. تعمل بدون تسميات مسبقة (Unsupervised).",style={"color":"#ccc","fontSize":"0.9rem","lineHeight":"1.7","marginBottom":"12px"}),
                            html.Div(style={"background":"#0d1117","borderRadius":"8px","padding":"12px"},children=[
                                html.P("⚙️ الإعدادات:",style={"color":"#e74c3c","margin":"0 0 6px","fontSize":"0.85rem","fontWeight":"700"}),
                                html.Ul([html.Li("contamination = 3% (نسبة الشذوذات المتوقعة)"),html.Li("random_state = 42 للتكرارية"),html.Li("يعمل على 5 متغيرات بيئية")],style={"color":"#aaa","fontSize":"0.82rem","lineHeight":"1.9","margin":0,"paddingRight":"18px"}),
                            ]),
                            html.Div(style={"marginTop":"10px","background":"#2a0a0a","borderRadius":"6px","padding":"8px 12px","border":"1px solid #5a1a1a"},children=[
                                html.Span("📤 المخرج: ",style={"color":"#e74c3c","fontWeight":"700","fontSize":"0.82rem"}),
                                html.Span("+1 طبيعي  /  -1 شاذ",style={"color":"#aaa","fontSize":"0.82rem"}),
                            ]),
                        ]),

                        # 3 KNN
                        html.Div(style={"background":"#1c2333","borderRadius":"14px","padding":"22px","border":"1px solid #e67e22","borderTop":"4px solid #e67e22"},children=[
                            html.Div(style={"display":"flex","alignItems":"center","gap":"12px","marginBottom":"12px"},children=[
                                html.Span("🌡️",style={"fontSize":"2rem"}),
                                html.Div([html.H3("KNN Regressor",style={"color":"#e67e22","margin":0,"fontSize":"1.1rem"}),html.Span("تقدير درجة الحرارة – السطح 3D",style={"color":"#888","fontSize":"0.8rem"})]),
                            ]),
                            html.P("يحسب متوسط أقرب 10 نقاط مجاورة لتقدير درجة الحرارة عند أي تركيبة من AQI والرطوبة، مما يُنتج السطح ثلاثي الأبعاد المرئي.",style={"color":"#ccc","fontSize":"0.9rem","lineHeight":"1.7","marginBottom":"12px"}),
                            html.Div(style={"background":"#0d1117","borderRadius":"8px","padding":"12px"},children=[
                                html.P("⚙️ الإعدادات:",style={"color":"#e67e22","margin":"0 0 6px","fontSize":"0.85rem","fontWeight":"700"}),
                                html.Ul([html.Li("n_neighbors = 10 جيران"),html.Li("شبكة 60×60 نقطة للسطح"),html.Li("المدخلات: AQI + Humidity")],style={"color":"#aaa","fontSize":"0.82rem","lineHeight":"1.9","margin":0,"paddingRight":"18px"}),
                            ]),
                            html.Div(style={"marginTop":"10px","background":"#2a1a0a","borderRadius":"6px","padding":"8px 12px","border":"1px solid #5a3a1a"},children=[
                                html.Span("📤 المخرج: ",style={"color":"#e67e22","fontWeight":"700","fontSize":"0.82rem"}),
                                html.Span("سطح ثلاثي الأبعاد لدرجة الحرارة",style={"color":"#aaa","fontSize":"0.82rem"}),
                            ]),
                        ]),

                        # 4 SHAP
                        html.Div(style={"background":"#1c2333","borderRadius":"14px","padding":"22px","border":"1px solid #9b59b6","borderTop":"4px solid #9b59b6"},children=[
                            html.Div(style={"display":"flex","alignItems":"center","gap":"12px","marginBottom":"12px"},children=[
                                html.Span("🧠",style={"fontSize":"2rem"}),
                                html.Div([html.H3("SHAP – Shapley Values",style={"color":"#9b59b6","margin":0,"fontSize":"1.1rem"}),html.Span("تفسير قرارات النموذج",style={"color":"#888","fontSize":"0.8rem"})]),
                            ]),
                            html.P("تقنية من نظرية الألعاب تحسب مساهمة كل متغير في قرار النموذج. تجيب: لماذا صُنِّفت هذه القراءة كـ خطر؟ مهمة للشفافية والثقة بالنموذج.",style={"color":"#ccc","fontSize":"0.9rem","lineHeight":"1.7","marginBottom":"12px"}),
                            html.Div(style={"background":"#0d1117","borderRadius":"8px","padding":"12px"},children=[
                                html.P("⚙️ طريقة العمل:",style={"color":"#9b59b6","margin":"0 0 6px","fontSize":"0.85rem","fontWeight":"700"}),
                                html.Ul([html.Li("TreeExplainer مُحسَّن لـ XGBoost"),html.Li("يحلل أول 500 قراءة اختبارية"),html.Li("يعطي قيمة لكل متغير في كل قراءة")],style={"color":"#aaa","fontSize":"0.82rem","lineHeight":"1.9","margin":0,"paddingRight":"18px"}),
                            ]),
                            html.Div(style={"marginTop":"10px","background":"#1a0a2a","borderRadius":"6px","padding":"8px 12px","border":"1px solid #3a1a5a"},children=[
                                html.Span("📤 المخرج: ",style={"color":"#9b59b6","fontWeight":"700","fontSize":"0.82rem"}),
                                html.Span("رسم أهمية المتغيرات بالترتيب",style={"color":"#aaa","fontSize":"0.82rem"}),
                            ]),
                        ]),

                        # 5 معادلة الخطر
                        html.Div(style={"background":"#1c2333","borderRadius":"14px","padding":"22px","border":"1px solid #3498db","borderTop":"4px solid #3498db"},children=[
                            html.Div(style={"display":"flex","alignItems":"center","gap":"12px","marginBottom":"12px"},children=[
                                html.Span("📐",style={"fontSize":"2rem"}),
                                html.Div([html.H3("معادلة مؤشر الخطر",style={"color":"#3498db","margin":0,"fontSize":"1.1rem"}),html.Span("RiskIndex – الحساب الأولي",style={"color":"#888","fontSize":"0.8rem"})]),
                            ]),
                            html.P("قبل تدريب النموذج، نحسب مؤشراً مركّباً يدمج ثلاثة مقاييس بيئية بأوزان مدروسة بناءً على خطورة كل منها على الصحة العامة.",style={"color":"#ccc","fontSize":"0.9rem","lineHeight":"1.7","marginBottom":"12px"}),
                            html.Div(style={"background":"#0d1117","borderRadius":"8px","padding":"14px","textAlign":"center"},children=[
                                html.P("RiskIndex =",style={"color":"#888","margin":"0 0 4px","fontSize":"0.82rem"}),
                                html.P("0.40 × AQI  +  0.30 × CO  +  0.30 × SMOKE",style={"color":"#3498db","fontFamily":"monospace","fontSize":"1rem","fontWeight":"700","margin":"0 0 12px"}),
                                html.Div(style={"display":"flex","justifyContent":"center","gap":"8px","flexWrap":"wrap"},children=[
                                    html.Span("< 50 آمن",      style={"background":"#1a3a1a","color":"#2ecc71","padding":"3px 8px","borderRadius":"4px","fontSize":"0.78rem"}),
                                    html.Span("50–100 متوسط",  style={"background":"#3a3a1a","color":"#f1c40f","padding":"3px 8px","borderRadius":"4px","fontSize":"0.78rem"}),
                                    html.Span("100–150 تحذير", style={"background":"#3a2a1a","color":"#e67e22","padding":"3px 8px","borderRadius":"4px","fontSize":"0.78rem"}),
                                    html.Span("150–200 حرج",   style={"background":"#3a1a1a","color":"#e74c3c","padding":"3px 8px","borderRadius":"4px","fontSize":"0.78rem"}),
                                    html.Span("> 200 طوارئ",   style={"background":"#2a1a3a","color":"#8e44ad","padding":"3px 8px","borderRadius":"4px","fontSize":"0.78rem"}),
                                ]),
                            ]),
                        ]),

                        # 6 تدفق العمل
                        html.Div(style={"background":"#1c2333","borderRadius":"14px","padding":"22px","border":"1px solid #1abc9c","borderTop":"4px solid #1abc9c"},children=[
                            html.Div(style={"display":"flex","alignItems":"center","gap":"12px","marginBottom":"12px"},children=[
                                html.Span("🔄",style={"fontSize":"2rem"}),
                                html.Div([html.H3("تدفق العمل الكامل",style={"color":"#1abc9c","margin":0,"fontSize":"1.1rem"}),html.Span("Pipeline – خطوة بخطوة",style={"color":"#888","fontSize":"0.8rem"})]),
                            ]),
                            html.Div(children=[
                                _step_item("1","📥 رفع الملف","قراءة Excel وفحص الأعمدة","#1abc9c"),
                                _step_item("2","🧹 التنظيف","تعبئة القيم المفقودة بالوسيط","#1abc9c"),
                                _step_item("3","📐 RiskIndex","المعادلة المركّبة AQI+CO+SMOKE","#1abc9c"),
                                _step_item("4","🔍 Isolation Forest","كشف القراءات الشاذة","#1abc9c"),
                                _step_item("5","🤖 XGBoost","تدريب 300 شجرة، 80/20 split","#1abc9c"),
                                _step_item("6","🧠 SHAP","تفسير أهمية كل متغير","#1abc9c"),
                                _step_item("7","📊 رسوم 3D","Scatter + Surface + Bars","#1abc9c"),
                            ]),
                        ]),

                    ]
                ),
                html.Hr(style={"borderColor":"#2a3a4a","margin":"30px 0"}),
            ]
        ),

        html.Div(
            style={"padding": "30px", "maxWidth": "900px", "margin": "0 auto"},
            children=[

                # بطاقة الرفع
                html.Div(
                    style={
                        "background":    "#1c2333",
                        "borderRadius":  "12px",
                        "padding":       "25px",
                        "border":        "2px solid #2ecc71",
                        "textAlign":     "center",
                    },
                    children=[
                        html.H3(
                            "📂 ارفع ملف Excel البيئي",
                            style={"color": "#2ecc71", "marginTop": 0},
                        ),
                        html.P("الأعمدة المطلوبة:", style={"color": "#ccc", "marginBottom": "6px"}),
                        html.Ul(
                            [
                                html.Li("AQI – مؤشر جودة الهواء (0–500)"),
                                html.Li("CO – تركيز أول أكسيد الكربون (ppm)"),
                                html.Li("SMOKE – كثافة الدخان (ppm)"),
                                html.Li("Temperature – درجة الحرارة (°C)"),
                                html.Li("Humidity – الرطوبة النسبية (%)"),
                            ],
                            style={
                                "color":        "#aaa",
                                "textAlign":    "right",
                                "lineHeight":   "2",
                                "marginBottom": "10px",
                            },
                        ),
                        dcc.Upload(
                            id="upload-data",
                            children=html.Div(
                                id="upload-label",
                                children=[
                                    html.Span("📁  اسحب الملف هنا أو "),
                                    html.A(
                                        "انقر للاختيار",
                                        style={"color": "#2ecc71", "textDecoration": "underline"},
                                    ),
                                ],
                                style={"color": "#ccc", "fontSize": "1.1rem"},
                            ),
                            className="upload-zone",
                            accept=".xlsx",
                        ),
                    ],
                ),

                # شريط التحميل
                html.Div(id="loading-indicator"),

                # بانر النجاح
                html.Div(id="status-bar",  style={"marginTop": "20px"}),

                # بطاقات الإحصاء
                html.Div(id="stat-cards",  style={"marginTop": "20px"}),
            ],
        ),

        # ── منطقة الرسوم ────────────────────────────────────
        html.Div(id="charts-section", style={"padding": "0 30px 30px"}),
    ],
)


# ── Callback 1: شريط التحميل يظهر فور اختيار الملف ─────────
@app.callback(
    Output("loading-indicator", "children"),
    Output("upload-label",      "children"),
    Input("upload-data",        "contents"),
    State("upload-data",        "filename"),
    prevent_initial_call=True,
)
def show_loading(contents, filename):
    if contents is None:
        return "", [
            html.Span("📁  اسحب الملف هنا أو "),
            html.A("انقر للاختيار",
                   style={"color": "#2ecc71", "textDecoration": "underline"}),
        ]

    steps = [
        "📥 قراءة الملف",
        "🔧 معالجة البيانات",
        "🤖 تدريب النموذج",
        "📊 رسم المخططات",
        "✅ اكتمل",
    ]

    loading_box = html.Div(
        className="loading-box",
        children=[
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                children=[
                    html.Span("⏳ جاري تحليل الملف...",
                              style={"color": "#2ecc71", "fontWeight": "700", "fontSize": "1rem"}),
                    html.Span(filename, style={"color": "#888", "fontSize": "0.85rem"}),
                ],
            ),
            html.Div(className="progress-wrap",
                     children=[html.Div(className="progress-fill")]),
            html.Div(
                className="steps-row",
                children=[
                    html.Span(s, className="step active" if i == 0 else "step")
                    for i, s in enumerate(steps)
                ],
            ),
        ],
    )

    new_label = [
        html.Span("✅  تم اختيار: "),
        html.Strong(filename, style={"color": "#2ecc71"}),
    ]
    return loading_box, new_label


# ── Callback 2: التحليل الكامل وعرض النتائج ─────────────────
@app.callback(
    Output("status-bar",     "children"),
    Output("stat-cards",     "children"),
    Output("charts-section", "children"),
    Input("upload-data",     "contents"),
    State("upload-data",     "filename"),
    prevent_initial_call=True,
)
def analyze(contents, filename):
    if contents is None:
        return "", "", ""

    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_excel(io.BytesIO(decoded))

    df = process_data(df)
    model, le, X_test, y_test, acc, features = train_model(df)

    n_anomaly  = int((df["Anomaly"] == -1).sum())
    n_critical = int(df["RiskLevel"].isin(["حرج", "طوارئ"]).sum())
    avg_aqi    = round(df["AQI"].mean(), 1)
    max_smoke  = int(df["SMOKE"].max())

    # بانر النجاح
    success = html.Div(
        className="success-banner",
        children=[
            f"✅  تم التحليل بنجاح!  |  "
            f"عدد القراءات: {len(df):,}  |  "
            f"دقة النموذج: {acc*100:.1f}%  |  "
            f"شذوذات مكتشفة: {n_anomaly}"
        ],
    )

    # بطاقات الإحصاء
    cards_data = [
        ("📊", "متوسط AQI",           avg_aqi,              "#3498db"),
        ("☁️", "أعلى SMOKE",          max_smoke,            "#e67e22"),
        ("⚠️", "قراءات حرجة/طوارئ",  n_critical,           "#e74c3c"),
        ("🔍", "شذوذات مكتشفة",       n_anomaly,            "#9b59b6"),
        ("🎯", "دقة النموذج",          f"{acc*100:.1f}%",    "#2ecc71"),
    ]
    cards = html.Div(
        style={
            "display":        "flex",
            "gap":            "15px",
            "flexWrap":       "wrap",
            "justifyContent": "center",
        },
        children=[
            html.Div(
                style={
                    "background":    "#1c2333",
                    "borderRadius":  "10px",
                    "padding":       "20px",
                    "flex":          "1",
                    "minWidth":      "150px",
                    "textAlign":     "center",
                    "border":        f"1px solid {c}",
                },
                children=[
                    html.Div(icon, style={"fontSize": "2rem"}),
                    html.Div(lbl,  style={"color": "#aaa", "fontSize": "0.85rem", "margin": "5px 0"}),
                    html.Div(str(val), style={"color": c, "fontSize": "1.5rem", "fontWeight": "bold"}),
                ],
            )
            for icon, lbl, val, c in cards_data
        ],
    )

    # دالة مساعدة لبناء بطاقة الرسم
    def chart_card(title, fig, desc):
        return html.Div(
            style={
                "background":    "#1c2333",
                "borderRadius":  "12px",
                "padding":       "20px",
                "marginBottom":  "25px",
                "border":        "1px solid #2a3a4a",
            },
            children=[
                html.H3(title, style={"color": "#2ecc71", "marginTop": 0}),
                html.P(desc,   style={"color": "#888", "fontSize": "0.9rem"}),
                dcc.Graph(figure=fig, config={"displayModeBar": False}),
            ],
        )

    charts = html.Div([
        chart_card(
            "🌐 رسم ثلاثي الأبعاد – AQI × CO × SMOKE",
            chart_3d_scatter(df),
            "توزيع القراءات في فضاء ثلاثي الأبعاد. النقاط الحمراء/البنفسجية = مناطق خطرة تستوجب التدخل الفوري.",
        ),
        chart_card(
            "🌡️ سطح ثلاثي الأبعاد – درجة الحرارة عبر AQI والرطوبة",
            chart_3d_surface(df),
            "سطح مُحاكى بخوارزمية KNN. الألوان الداكنة = حرارة منخفضة، الألوان المضيئة = حرارة مرتفعة.",
        ),
        chart_card(
            "📊 أعمدة 3D – متوسط القراءات لكل مستوى",
            chart_3d_bar(df),
            "مقارنة متوسطات AQI وCO وSMOKE لكل تصنيف خطر. يساعد في فهم الفوارق بين المستويات.",
        ),
        chart_card(
            "⚠️ الشذوذات عبر الزمن",
            chart_anomaly_timeline(df),
            "الخط الأخضر = قراءات طبيعية، النقاط الحمراء ✕ = شذوذات اكتشفها Isolation Forest.",
        ),
        chart_card(
            "🧠 أهمية المتغيرات (SHAP)",
            chart_shap(model, X_test, features),
            "قيم SHAP تفسّر قرارات النموذج: المتغير الأطول شريطاً هو الأكثر تأثيراً في التصنيف.",
        ),
    ])

    return success, cards, charts


# ============================================================
# تشغيل التطبيق
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print("\n" + "=" * 55)
    print("  🌍 EnviroAI Pro – نظام الذكاء الاصطناعي البيئي")
    print("=" * 55)
    print(f"  افتح المتصفح على: http://localhost:{port}")
    print("=" * 55 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port)
