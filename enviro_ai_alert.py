"""
╔══════════════════════════════════════════════════════════════════════╗
║  EnviroAI Pro v3                                                     ║
║  + تنزيل Excel/PDF/CSV + صور متحركة SVG + شارات + صوت              ║
╚══════════════════════════════════════════════════════════════════════╝
pip install dash pandas scikit-learn openpyxl plotly numpy xlsxwriter
python enviro_ai_v3.py   →  http://localhost:8050
"""

import os, warnings, base64, io, datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

import dash
from dash import dcc, html, Input, Output, State

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════
# بيانات + نماذج
# ══════════════════════════════════════════════════
def process_data(df):
    df = df.copy()
    df.fillna(df.median(numeric_only=True), inplace=True)
    df["RiskIndex"] = 0.40*df["AQI"] + 0.30*df["CO"] + 0.30*df["SMOKE"]
    bins   = [-np.inf, 50, 100, 150, 200, np.inf]
    labels = ["آمن","متوسط","تحذير","حرج","طوارئ"]
    df["RiskLevel"] = pd.cut(df["RiskIndex"], bins=bins, labels=labels)
    return df

def train_all(df):
    feat = ["AQI","CO","SMOKE","Temperature","Humidity"]
    X  = df[feat].values
    le = LabelEncoder(); y = le.fit_transform(df["RiskLevel"].astype(str))
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    Xtr,Xte,ytr,yte = train_test_split(Xs,y,test_size=0.2,random_state=42,stratify=y)
    m={}
    rf=RandomForestClassifier(n_estimators=200,max_depth=8,random_state=42); rf.fit(Xtr,ytr)
    m["RF"]={"model":rf,"acc":accuracy_score(yte,rf.predict(Xte))}
    sv=SVC(kernel="rbf",C=1.0,random_state=42); sv.fit(Xtr,ytr)
    m["SVM"]={"model":sv,"acc":accuracy_score(yte,sv.predict(Xte))}
    lof=LocalOutlierFactor(n_neighbors=20,novelty=True); lof.fit(Xs)
    m["LOF"]={"model":lof,"anom":int((lof.predict(Xs)==-1).sum())}
    oc=OneClassSVM(kernel="rbf",nu=0.05); oc.fit(Xs)
    m["OCSVM"]={"model":oc,"anom":int((oc.predict(Xs)==-1).sum())}
    iso=IsolationForest(contamination=0.03,random_state=42)
    m["ISO"]={"model":iso,"anom":int((iso.fit_predict(Xs)==-1).sum())}
    km=KMeans(n_clusters=5,init="k-means++",random_state=42,n_init=10); km.fit(Xs)
    m["KM"]={"model":km}
    m["_sc"]=sc; m["_le"]=le; m["_feat"]=feat
    return m

def stats(df):
    c=df["RiskLevel"].value_counts().to_dict()
    return {"total":len(df),"avg_aqi":round(df["AQI"].mean(),1),
            "max_aqi":int(df["AQI"].max()),"avg_co":round(df["CO"].mean(),1),
            "max_co":int(df["CO"].max()),"max_smoke":int(df["SMOKE"].max()),
            "safe":c.get("آمن",0),"moderate":c.get("متوسط",0),
            "warning":c.get("تحذير",0),"critical":c.get("حرج",0),"emergency":c.get("طوارئ",0)}

def sim(aqi,co,smoke,temp,hum,m):
    Xn=np.array([[aqi,co,smoke,temp,hum]]); Xs=m["_sc"].transform(Xn)
    ri=0.4*aqi+0.3*co+0.3*smoke
    lv="آمن" if ri<50 else "متوسط" if ri<100 else "تحذير" if ri<150 else "حرج" if ri<200 else "طوارئ"
    r={"ri":round(ri,1),"level":lv}
    r["rf"]=m["_le"].inverse_transform(m["RF"]["model"].predict(Xs))[0]
    r["svm"]=m["_le"].inverse_transform(m["SVM"]["model"].predict(Xs))[0]
    r["lof"]="شاذ" if m["LOF"]["model"].predict(Xs)[0]==-1 else "طبيعي"
    r["oc"]="شاذ" if m["OCSVM"]["model"].predict(Xs)[0]==-1 else "طبيعي"
    r["iso"]="شاذ" if m["ISO"]["model"].predict(Xs)[0]==-1 else "طبيعي"
    r["km"]=int(m["KM"]["model"].predict(Xs)[0])
    return r

# ══════════════════════════════════════════════════
# رسوم بيانية
# ══════════════════════════════════════════════════
def bar_fig(df):
    c=df["RiskLevel"].value_counts().reindex(["آمن","متوسط","تحذير","حرج","طوارئ"],fill_value=0)
    fig=go.Figure(go.Bar(x=c.index.tolist(),y=c.values,
        marker_color=["#43a047","#fdd835","#ff9800","#e53935","#1565c0"],
        text=c.values,textposition="outside"))
    fig.update_layout(paper_bgcolor="white",plot_bgcolor="#fafafa",
        font=dict(family="Cairo",color="#1a2a3a"),
        margin=dict(t=30,b=20,l=20,r=20),height=260,
        xaxis=dict(showgrid=False),yaxis=dict(gridcolor="#eee"))
    return fig

def timeline_fig(df):
    fig=go.Figure()
    for col,c in [("AQI","#e53935"),("CO","#1565c0"),("SMOKE","#43a047")]:
        fig.add_trace(go.Scatter(x=df.index,y=df[col],mode="lines",name=col,
            line=dict(color=c,width=1.5)))
    fig.add_trace(go.Scatter(x=[df.index[0],df.index[-1]],y=[200,200],
        mode="lines",name="حد الطوارئ",line=dict(color="#e53935",width=2,dash="dash")))
    fig.update_layout(paper_bgcolor="white",plot_bgcolor="#fafafa",
        font=dict(family="Cairo",color="#1a2a3a"),
        margin=dict(t=30,b=30,l=20,r=20),height=260,
        legend=dict(orientation="h",y=-0.22),
        xaxis=dict(showgrid=False),yaxis=dict(gridcolor="#eee"))
    return fig

def gauge_fig(val,title,maxv,color):
    fig=go.Figure(go.Indicator(mode="gauge+number",value=val,
        title={"text":title,"font":{"family":"Cairo","size":13,"color":"#1a2a3a"}},
        number={"font":{"family":"Cairo","size":22,"color":color}},
        gauge={"axis":{"range":[0,maxv],"tickfont":{"size":9}},
               "bar":{"color":color},"bgcolor":"white","bordercolor":"#ddd",
               "steps":[{"range":[0,maxv*.3],"color":"#e8f5e9"},
                        {"range":[maxv*.3,maxv*.6],"color":"#fff9c4"},
                        {"range":[maxv*.6,maxv*.8],"color":"#ffe0b2"},
                        {"range":[maxv*.8,maxv],"color":"#ffebee"}],
               "threshold":{"line":{"color":"#e53935","width":3},"value":maxv*.8}}))
    fig.update_layout(paper_bgcolor="white",margin=dict(t=40,b=10,l=20,r=20),height=200)
    return fig

# ══════════════════════════════════════════════════
# تصدير البيانات
# ══════════════════════════════════════════════════
def export_excel(df, st, ms):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="البيانات الكاملة", index=False)
        summary = pd.DataFrame({
            "المقياس":["إجمالي القراءات","متوسط AQI","أعلى AQI","متوسط CO","أعلى SMOKE",
                       "آمن","متوسط","تحذير","حرج","طوارئ","دقة RF","دقة SVM"],
            "القيمة":[st["total"],st["avg_aqi"],st["max_aqi"],st["avg_co"],st["max_smoke"],
                      st["safe"],st["moderate"],st["warning"],st["critical"],st["emergency"],
                      f"{ms['RF']['acc']*100:.1f}%",f"{ms['SVM']['acc']*100:.1f}%"],
        })
        summary.to_excel(w, sheet_name="الملخص الإحصائي", index=False)
        anomaly_df = df[df["RiskLevel"].isin(["حرج","طوارئ"])].copy()
        anomaly_df.to_excel(w, sheet_name="القراءات الخطرة", index=False)
    return base64.b64encode(buf.getvalue()).decode()

def export_csv(df):
    return base64.b64encode(df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")).decode()

# ══════════════════════════════════════════════════
# CSS + animations + JS sound
# ══════════════════════════════════════════════════
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Cairo,sans-serif;background:#f5f7fa}

@keyframes pulse    {0%,100%{opacity:1}50%{opacity:.2}}
@keyframes ping     {0%{transform:scale(1);opacity:1}100%{transform:scale(2.8);opacity:0}}
@keyframes slideDown{from{transform:translateY(-18px);opacity:0}to{transform:translateY(0);opacity:1}}
@keyframes popIn    {0%{transform:scale(.78);opacity:0}100%{transform:scale(1);opacity:1}}
@keyframes shimmer  {0%,100%{border-color:#e53935}50%{border-color:#ff1744}}
@keyframes bounce   {0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
@keyframes spin     {from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
@keyframes wave     {0%{transform:scaleY(1)}50%{transform:scaleY(.3)}100%{transform:scaleY(1)}}
@keyframes float    {0%,100%{transform:translateY(0) rotate(0deg)}
                     33%{transform:translateY(-10px) rotate(3deg)}
                     66%{transform:translateY(4px) rotate(-2deg)}}
@keyframes orbit    {from{transform:rotate(0deg) translateX(28px) rotate(0deg)}
                     to  {transform:rotate(360deg) translateX(28px) rotate(-360deg)}}
@keyframes flicker  {0%,100%{opacity:1}45%{opacity:.6}50%{opacity:1}55%{opacity:.7}}
@keyframes slideRight{from{transform:translateX(-30px);opacity:0}to{transform:translateX(0);opacity:1}}
@keyframes glow     {0%,100%{box-shadow:0 0 6px rgba(229,57,53,.4)}50%{box-shadow:0 0 20px rgba(229,57,53,.9)}}
@keyframes countUp  {from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}

.alert-box   {animation:slideDown .5s ease, shimmer 1.5s infinite}
.stat-card   {animation:popIn .45s ease; transition:transform .15s,box-shadow .15s}
.stat-card:hover{transform:translateY(-4px);box-shadow:0 8px 24px rgba(0,0,0,.1)}
.algo-card   {transition:transform .15s,box-shadow .15s}
.algo-card:hover{transform:translateY(-4px);box-shadow:0 8px 24px rgba(0,0,0,.1)}
.dl-btn      {transition:transform .12s,box-shadow .12s}
.dl-btn:hover{transform:translateY(-3px);box-shadow:0 6px 18px rgba(0,0,0,.15)}
.dl-btn:active{transform:scale(.96)}

.badge-safe      {background:#e8f5e9;color:#1b5e20;border:2px solid #43a047}
.badge-moderate  {background:#fff9c4;color:#f57f17;border:2px solid #fdd835}
.badge-warning   {background:#fff3e0;color:#e65100;border:2px solid #ff9800}
.badge-critical  {background:#ffebee;color:#b71c1c;border:2px solid #e53935}
.badge-emergency {background:#e3f2fd;color:#0d47a1;border:2px solid #1565c0;animation:glow 1.2s infinite}
.badge-base{border-radius:20px;padding:4px 14px;font-size:13px;font-weight:800;
            display:inline-flex;align-items:center;gap:5px}

.live-dot{display:inline-block;width:10px;height:10px;background:#43a047;
           border-radius:50%;animation:pulse 1.1s infinite}
.ping-wrap{position:relative;display:inline-block;width:18px;height:18px}
.ping-core{position:absolute;inset:2px;background:#e53935;border-radius:50%}
.ping-ring{position:absolute;inset:-4px;border:2px solid #e53935;border-radius:50%;animation:ping 1.2s infinite}

.wave-bar{display:inline-block;width:5px;margin:0 1px;border-radius:3px;animation:wave 1s ease infinite}
.wave-bar:nth-child(2){animation-delay:.1s}
.wave-bar:nth-child(3){animation-delay:.2s}
.wave-bar:nth-child(4){animation-delay:.3s}
.wave-bar:nth-child(5){animation-delay:.4s}

.algo-explain{background:#f8faff;border-radius:10px;padding:14px;
              border-right:4px solid;margin-top:10px;font-size:12px;
              line-height:1.8;color:#444;animation:slideRight .4s ease}

/* animated SVG icons */
.icon-leaf   {animation:float 3s ease-in-out infinite}
.icon-smoke  {animation:float 2.5s ease-in-out infinite .3s}
.icon-sensor {animation:bounce 2.2s ease-in-out infinite}
.icon-cpu    {animation:spin 4s linear infinite}
.icon-bell   {animation:flicker 1.5s infinite}
.icon-chart  {animation:float 3.5s ease-in-out infinite .5s}
.icon-arrow  {animation:bounce 1.5s ease-in-out infinite}
.orbit-dot   {animation:orbit 2s linear infinite}

.section-card{background:white;border-radius:16px;padding:20px;margin:0 16px 16px;
              box-shadow:0 2px 12px rgba(0,0,0,.06);border:1px solid #f0f0f0}
"""

JS_SOUND = """
function playAlert(level){
  try{
    const ctx=new(window.AudioContext||window.webkitAudioContext)();
    const cfg={emergency:[[880,.15],[660,.15],[880,.15],[660,.15],[880,.3]],
               critical:[[660,.25],[440,.25],[660,.25]],
               warning:[[500,.3],[380,.2]],moderate:[[330,.3]],safe:[[220,.4]]};
    const key=level.includes('طوارئ')?'emergency':level.includes('حرج')?'critical':
              level.includes('تحذير')?'warning':level.includes('متوسط')?'moderate':'safe';
    const type=key==='emergency'?'sawtooth':key==='critical'?'square':'sine';
    let t=ctx.currentTime;
    (cfg[key]||cfg.safe).forEach(([f,d])=>{
      const o=ctx.createOscillator(),g=ctx.createGain();
      o.connect(g);g.connect(ctx.destination);
      o.frequency.value=f; o.type=type;
      g.gain.setValueAtTime(.35,t);
      g.gain.exponentialRampToValueAtTime(.001,t+d);
      o.start(t); o.stop(t+d); t+=d+.05;
    });
  }catch(e){}
}
window.addEventListener('DOMContentLoaded',()=>{
  new MutationObserver(muts=>{
    muts.forEach(m=>{m.addedNodes.forEach(n=>{
      if(n.nodeType===1){const el=n.querySelector?n.querySelector('[data-level]'):null;
      if(el)playAlert(el.getAttribute('data-level'));}
    });});
  }).observe(document.body,{childList:true,subtree:true});
});
"""

# ══════════════════════════════════════════════════
# صور SVG متحركة
# ══════════════════════════════════════════════════
def svg_leaf():
    return html.Div(style={"textAlign":"center","padding":"10px"},children=[
        html.Div(className="icon-leaf",children=[
            html.Svg(viewBox="0 0 80 80",width="80",height="80",children=[
                html.Ellipse(cx="40",cy="38",rx="24",ry="30",fill="#2e7d32",transform="rotate(-15 40 38)"),
                html.Line(x1="40",y1="65",x2="40",y2="20",stroke="#1b5e20",strokeWidth="2"),
                html.Line(x1="40",y1="35",x2="28",y2="28",stroke="#4caf50",strokeWidth="1.5"),
                html.Line(x1="40",y1="42",x2="26",y2="38",stroke="#4caf50",strokeWidth="1.5"),
                html.Line(x1="40",y1="35",x2="52",y2="28",stroke="#4caf50",strokeWidth="1.5"),
                html.Line(x1="40",y1="42",x2="54",y2="38",stroke="#4caf50",strokeWidth="1.5"),
            ],xmlns="http://www.w3.org/2000/svg"),
        ])])

def svg_smoke():
    return html.Div(style={"textAlign":"center","padding":"10px"},children=[
        html.Div(className="icon-smoke",children=[
            html.Svg(viewBox="0 0 80 80",width="80",height="80",children=[
                html.Ellipse(cx="30",cy="50",rx="16",ry="12",fill="#9e9e9e",opacity="0.7"),
                html.Ellipse(cx="50",cy="45",rx="14",ry="11",fill="#bdbdbd",opacity="0.6"),
                html.Ellipse(cx="40",cy="35",rx="18",ry="14",fill="#757575",opacity="0.8"),
                html.Ellipse(cx="40",cy="22",rx="12",ry="10",fill="#616161",opacity="0.7"),
                html.Rect(x="35",y="55",width="10",height="20",rx="2",fill="#424242"),
            ],xmlns="http://www.w3.org/2000/svg"),
        ])])

def svg_sensor():
    return html.Div(style={"textAlign":"center","padding":"10px"},children=[
        html.Div(className="icon-sensor",children=[
            html.Svg(viewBox="0 0 80 80",width="80",height="80",children=[
                html.Rect(x="20",y="25",width="40",height="35",rx="6",fill="#1565c0"),
                html.Rect(x="26",y="31",width="28",height="18",rx="3",fill="#e3f2fd"),
                html.Circle(cx="30",cy="40",r="3",fill="#e53935"),
                html.Circle(cx="40",cy="40",r="3",fill="#fdd835"),
                html.Circle(cx="50",cy="40",r="3",fill="#43a047"),
                html.Line(x1="30",y1="60",x2="30",y2="68",stroke="#1565c0",strokeWidth="3"),
                html.Line(x1="50",y1="60",x2="50",y2="68",stroke="#1565c0",strokeWidth="3"),
                html.Line(x1="25",y1="68",x2="55",y2="68",stroke="#1565c0",strokeWidth="3"),
                html.Ellipse(cx="40",cy="16",rx="18",ry="6",fill="none",stroke="#1565c0",strokeWidth="2",strokeDasharray="4 3"),
            ],xmlns="http://www.w3.org/2000/svg"),
        ])])

def svg_cpu():
    return html.Div(style={"textAlign":"center","padding":"10px"},children=[
        html.Div(className="icon-cpu",children=[
            html.Svg(viewBox="0 0 80 80",width="80",height="80",children=[
                html.Rect(x="22",y="22",width="36",height="36",rx="4",fill="#9c27b0"),
                html.Rect(x="28",y="28",width="24",height="24",rx="3",fill="#e1bee7"),
                html.Rect(x="33",y="33",width="14",height="14",rx="2",fill="#7b1fa2"),
                html.Text("AI",x="40",y="43",textAnchor="middle",fill="white",fontSize="9",fontWeight="bold",fontFamily="Cairo"),
                *[html.Line(x1=str(22-8),y1=str(28+i*8),x2="22",y2=str(28+i*8),stroke="#9c27b0",strokeWidth="2") for i in range(4)],
                *[html.Line(x1=str(58+8),y1=str(28+i*8),x2="58",y2=str(28+i*8),stroke="#9c27b0",strokeWidth="2") for i in range(4)],
            ],xmlns="http://www.w3.org/2000/svg"),
        ])])

def svg_bell_alert():
    return html.Div(style={"textAlign":"center","padding":"10px"},children=[
        html.Div(className="icon-bell",children=[
            html.Svg(viewBox="0 0 80 80",width="80",height="80",children=[
                html.Path(d="M40 15 C30 15 24 22 24 32 L24 50 L16 58 L64 58 L56 50 L56 32 C56 22 50 15 40 15",fill="#e53935"),
                html.Circle(cx="40",cy="66",r="5",fill="#e53935"),
                html.Rect(x="36",y="10",width="8",height="8",rx="4",fill="#b71c1c"),
                html.Text("!",x="40",y="46",textAnchor="middle",fill="white",fontSize="20",fontWeight="bold",fontFamily="Cairo"),
            ],xmlns="http://www.w3.org/2000/svg"),
        ])])

def svg_download():
    return html.Div(className="icon-arrow",children=[
        html.Svg(viewBox="0 0 40 40",width="28",height="28",children=[
            html.Rect(x="5",y="28",width="30",height="5",rx="2",fill="currentColor"),
            html.Polygon(points="20,26 10,14 16,14 16,5 24,5 24,14 30,14",fill="currentColor"),
        ],xmlns="http://www.w3.org/2000/svg"),
    ])

def svg_chart_anim():
    return html.Div(className="icon-chart",children=[
        html.Svg(viewBox="0 0 60 50",width="60",height="50",children=[
            html.Rect(x="5", y="30",width="10",height="15",rx="2",fill="#43a047"),
            html.Rect(x="18",y="18",width="10",height="27",rx="2",fill="#fdd835"),
            html.Rect(x="31",y="10",width="10",height="35",rx="2",fill="#ff9800"),
            html.Rect(x="44",y="22",width="10",height="23",rx="2",fill="#e53935"),
            html.Line(x1="5",y1="46",x2="55",y2="46",stroke="#ccc",strokeWidth="1"),
        ],xmlns="http://www.w3.org/2000/svg"),
    ])

# ══════════════════════════════════════════════════
# التطبيق
# ══════════════════════════════════════════════════
app    = dash.Dash(__name__, title="EnviroAI Pro v3 🌍")
server = app.server

app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>""" + CSS + """</style>
<script>""" + JS_SOUND + """</script>
</head>
<body>{%app_entry%}
<footer>{%config%}{%scripts%}{%renderer%}</footer>
</body></html>"""

C={"red":"#e53935","red_bg":"#fff0f0","red_dk":"#b71c1c",
   "blue":"#1565c0","blue_bg":"#e8f0fe","blue_dk":"#0d47a1",
   "green":"#43a047","green_bg":"#e8f5e9","green_dk":"#1b5e20",
   "warn":"#ff9800","warn_bg":"#fff3e0","warn_dk":"#e65100"}

LCOLOR={"آمن":C["green"],"متوسط":C["warn"],"تحذير":"#e65100","حرج":C["red"],"طوارئ":C["blue"]}
LBG   ={"آمن":C["green_bg"],"متوسط":C["warn_bg"],"تحذير":"#fbe9e7","حرج":C["red_bg"],"طوارئ":C["blue_bg"]}
LBADGE={"آمن":"badge-safe","متوسط":"badge-moderate","تحذير":"badge-warning","حرج":"badge-critical","طوارئ":"badge-emergency"}
LICON ={"آمن":"✅","متوسط":"🟡","تحذير":"🟠","حرج":"🔴","طوارئ":"🚨"}

def badge(lv):
    return html.Span([LICON.get(lv,"")," ",lv],className=f"badge-base {LBADGE.get(lv,'badge-safe')}")

def stat_card(label,val,color,icon=""):
    return html.Div(className="stat-card",style={
        "background":C[f"{color}_bg"],"border":f"2px solid {C[color]}",
        "borderRadius":"12px","padding":"14px 16px","textAlign":"center"
    },children=[
        html.Div(f"{icon} {label}",style={"fontSize":"11px","fontWeight":"700","opacity":".65","marginBottom":"4px"}),
        html.Div(str(val),style={"fontSize":"24px","fontWeight":"800","color":C[color],"animation":"countUp .5s ease"}),
    ])

def dl_btn(label,btn_id,color,icon_el):
    return html.Div(style={"display":"flex","flexDirection":"column","alignItems":"center","gap":"6px"},children=[
        html.Button(children=[icon_el," ",label],id=btn_id,n_clicks=0,className="dl-btn",
            style={"background":C[color],"color":"white","border":"none","borderRadius":"10px",
                   "padding":"12px 20px","fontSize":"13px","fontWeight":"800",
                   "cursor":"pointer","fontFamily":"Cairo,sans-serif",
                   "display":"flex","alignItems":"center","gap":"8px","width":"100%","justifyContent":"center"}),
    ])

def algo_card(name,type_lbl,desc,params,explain,color):
    return html.Div(className="algo-card",style={
        "background":C[f"{color}_bg"],"border":f"2px solid {C[color]}",
        "borderLeft":f"5px solid {C[color]}","borderRadius":"12px","padding":"14px"
    },children=[
        html.Span(type_lbl,style={"background":C[f"{color}_bg"],"color":C[f"{color}_dk"],
            "border":f"1.5px solid {C[color]}","borderRadius":"10px",
            "padding":"2px 10px","fontSize":"10px","fontWeight":"800"}),
        html.Div(name,style={"fontSize":"14px","fontWeight":"800","color":"#1a1a1a","marginTop":"8px","marginBottom":"4px"}),
        html.P(desc,style={"fontSize":"12px","color":"#555","lineHeight":"1.7","marginBottom":"8px"}),
        html.Div([html.Span(p,style={"background":C[f"{color}_bg"],"color":C[f"{color}_dk"],
            "borderRadius":"6px","padding":"2px 8px","fontSize":"10px",
            "fontFamily":"monospace","fontWeight":"600","marginLeft":"4px"}) for p in params],
            style={"marginBottom":"8px"}),
        html.Div(explain,className="algo-explain",style={"borderColor":C[color]}),
    ])

def wave_bars(color):
    return html.Span([
        html.Span(className="wave-bar",style={"background":color,"height":"14px"}),
        html.Span(className="wave-bar",style={"background":color,"height":"20px"}),
        html.Span(className="wave-bar",style={"background":color,"height":"12px"}),
        html.Span(className="wave-bar",style={"background":color,"height":"18px"}),
        html.Span(className="wave-bar",style={"background":color,"height":"10px"}),
    ])

app.layout = html.Div(style={"fontFamily":"Cairo,sans-serif","background":"#f5f7fa","direction":"rtl","minHeight":"100vh"},children=[

    # ── رأس ──
    html.Div(style={"background":"white","borderBottom":f"3px solid {C['red']}",
        "padding":"14px 24px","display":"flex","alignItems":"center",
        "justifyContent":"space-between","flexWrap":"wrap","gap":"10px",
        "boxShadow":"0 2px 10px rgba(229,57,53,.1)"},children=[
        html.Div([
            html.Span(["EnviroAI ",html.Span("Pro v3",style={"color":C["blue"],"fontSize":"14px"})],
                style={"fontSize":"24px","fontWeight":"800","color":C["red"]}),
            html.Span(" ",className="live-dot",style={"margin":"0 6px"}),
            html.Div("نظام المراقبة البيئية الذكي · 6 خوارزميات + تنزيل + صور متحركة",
                style={"fontSize":"11px","color":"#888","marginTop":"3px"}),
        ]),
        html.Div([wave_bars(C["green"]),
            html.Span(" ",style={"width":"12px","display":"inline-block"}),
            html.Div(id="clock",style={"fontSize":"22px","fontWeight":"800","color":C["blue"],
                "fontFamily":"monospace","letterSpacing":"3px","display":"inline"})],
            style={"display":"flex","alignItems":"center","gap":"8px"}),
    ]),

    # ── صور متحركة تعريفية ──
    html.Div(className="section-card",style={"margin":"16px"},children=[
        html.Div("🎨 عناصر النظام المتحركة",style={"fontSize":"13px","fontWeight":"800","color":C["blue"],
            "marginBottom":"14px","borderBottom":f"2px solid {C['blue']}","paddingBottom":"4px"}),
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(120px,1fr))","gap":"10px"},children=[
            html.Div(style={"textAlign":"center","background":C["green_bg"],"borderRadius":"12px","padding":"12px",
                "border":f"2px solid {C['green']}"},children=[
                svg_leaf(),
                html.Div("البيئة",style={"fontSize":"11px","fontWeight":"700","color":C["green_dk"],"marginTop":"4px"}),
            ]),
            html.Div(style={"textAlign":"center","background":"#fafafa","borderRadius":"12px","padding":"12px",
                "border":"2px solid #9e9e9e"},children=[
                svg_smoke(),
                html.Div("التلوث",style={"fontSize":"11px","fontWeight":"700","color":"#555","marginTop":"4px"}),
            ]),
            html.Div(style={"textAlign":"center","background":C["blue_bg"],"borderRadius":"12px","padding":"12px",
                "border":f"2px solid {C['blue']}"},children=[
                svg_sensor(),
                html.Div("المستشعرات",style={"fontSize":"11px","fontWeight":"700","color":C["blue_dk"],"marginTop":"4px"}),
            ]),
            html.Div(style={"textAlign":"center","background":"#f9e8ff","borderRadius":"12px","padding":"12px",
                "border":"2px solid #9c27b0"},children=[
                svg_cpu(),
                html.Div("الذكاء الاصطناعي",style={"fontSize":"11px","fontWeight":"700","color":"#4a148c","marginTop":"4px"}),
            ]),
            html.Div(style={"textAlign":"center","background":C["red_bg"],"borderRadius":"12px","padding":"12px",
                "border":f"2px solid {C['red']}"},children=[
                svg_bell_alert(),
                html.Div("الإنذارات",style={"fontSize":"11px","fontWeight":"700","color":C["red_dk"],"marginTop":"4px"}),
            ]),
            html.Div(style={"textAlign":"center","background":C["warn_bg"],"borderRadius":"12px","padding":"12px",
                "border":f"2px solid {C['warn']}"},children=[
                svg_chart_anim(),
                html.Div("التحليلات",style={"fontSize":"11px","fontWeight":"700","color":C["warn_dk"],"marginTop":"4px"}),
            ]),
        ]),
    ]),

    # ── رفع الملف ──
    html.Div(className="section-card",children=[
        html.Div(style={"background":C["blue_bg"],"borderRadius":"12px","padding":"20px","textAlign":"center",
            "border":f"2px dashed {C['blue']}"},children=[
            html.Div(className="icon-sensor",style={"marginBottom":"8px"},children=[
                html.Svg(viewBox="0 0 60 60",width="60",height="60",children=[
                    html.Rect(x="10",y="15",width="40",height="30",rx="5",fill=C["blue"]),
                    html.Rect(x="16",y="20",width="28",height="14",rx="3",fill="#e3f2fd"),
                    html.Line(x1="25",y1="45",x2="25",y2="52",stroke=C["blue"],strokeWidth="3"),
                    html.Line(x1="35",y1="45",x2="35",y2="52",stroke=C["blue"],strokeWidth="3"),
                    html.Line(x1="20",y1="52",x2="40",y2="52",stroke=C["blue"],strokeWidth="3"),
                ],xmlns="http://www.w3.org/2000/svg"),
            ]),
            html.H3("ارفع ملف Excel البيئي",style={"color":C["blue"],"margin":"0 0 6px"}),
            html.P("الأعمدة المطلوبة: AQI · CO · SMOKE · Temperature · Humidity",
                style={"fontSize":"12px","color":"#666","marginBottom":"12px"}),
            dcc.Upload(id="upload",accept=".xlsx",
                children=html.Div(["📁 اسحب هنا أو ",
                    html.A("انقر للاختيار",style={"color":C["blue"],"textDecoration":"underline"})],
                    style={"fontSize":"13px","color":"#555"}),
                style={"border":f"2px dashed {C['blue']}","borderRadius":"8px",
                    "padding":"18px","cursor":"pointer","background":"white"}),
        ]),
        html.Div(id="status-bar",style={"marginTop":"12px"}),
    ]),

    # ── إنذار ──
    html.Div(id="alert-zone"),

    # ── شارات ──
    html.Div(id="badges-zone",style={"padding":"0 16px 8px"}),

    # ── أزرار التنزيل ──
    html.Div(id="download-zone"),

    # ── عدادات ──
    html.Div(id="gauges-zone"),

    # ── إحصاءات ──
    html.Div(id="stats-zone"),

    # ── رسوم ──
    html.Div(id="charts-zone"),

    # ── خوارزميات ──
    html.Div(id="algo-zone"),

    # ── محاكاة ──
    html.Div(id="sim-zone"),

    # ── مكونات Dash ──
    dcc.Interval(id="tick",interval=1000,n_intervals=0),
    dcc.Store(id="store"),
    dcc.Download(id="dl-excel"),
    dcc.Download(id="dl-csv"),
])

# ── ساعة ──
@app.callback(Output("clock","children"),Input("tick","n_intervals"))
def clock(_):
    n=datetime.datetime.now(); h=n.hour%12 or 12
    return f"{h:02d}:{n.minute:02d}:{n.second:02d} {'AM' if n.hour<12 else 'PM'}"

# ── رفع ──
@app.callback(
    Output("status-bar","children"), Output("alert-zone","children"),
    Output("badges-zone","children"), Output("download-zone","children"),
    Output("gauges-zone","children"), Output("stats-zone","children"),
    Output("charts-zone","children"), Output("algo-zone","children"),
    Output("sim-zone","children"),    Output("store","data"),
    Input("upload","contents"), State("upload","filename"),
    prevent_initial_call=True,
)
def on_upload(contents,filename):
    _,cs=contents.split(",")
    df=pd.read_excel(io.BytesIO(base64.b64decode(cs)))
    df=process_data(df); ms=train_all(df); st=stats(df)

    # شريط الحالة
    status=html.Div(style={"background":C["green_bg"],"border":f"2px solid {C['green']}",
        "borderRadius":"10px","padding":"12px 16px","textAlign":"center",
        "fontWeight":"700","fontSize":"13px","color":C["green_dk"],"animation":"slideDown .4s ease"},
        children=[f"✅ تم تحليل {len(df):,} قراءة  |  RF: {ms['RF']['acc']*100:.1f}%  |  SVM: {ms['SVM']['acc']*100:.1f}%  |  شذوذات LOF: {ms['LOF']['anom']}"])

    # إنذار
    alert_zone=html.Div()
    if st["emergency"]>0:
        alert_zone=html.Div(className="alert-box section-card",style={
            "background":C["red_bg"],"border":f"3px solid {C['red']}",
            "borderRadius":"14px","margin":"0 16px 8px","padding":"18px"},children=[
            html.Span(**{"data-level":"طوارئ"},style={"display":"none"}),
            html.Div(style={"display":"flex","alignItems":"center","gap":"12px","marginBottom":"10px"},children=[
                html.Div([html.Div(className="ping-core"),html.Div(className="ping-ring")],className="ping-wrap"),
                html.Div("🚨 إنذار طوارئ — قراءات خطرة مكتشفة",
                    style={"fontSize":"18px","fontWeight":"800","color":C["red_dk"]}),
            ]),
            html.Div(f"طوارئ: {st['emergency']}  |  أعلى AQI: {st['max_aqi']}  |  أعلى CO: {st['max_co']}  |  أعلى SMOKE: {st['max_smoke']} ppm",
                style={"fontSize":"12px","color":C["red"],"fontWeight":"600","marginBottom":"12px"}),
            html.Div(style={"display":"flex","gap":"10px","flexWrap":"wrap"},children=[
                html.Div(style={"background":"white","border":f"2px solid {C['red']}","borderRadius":"8px","padding":"8px 14px","textAlign":"center"},
                    children=[html.Div("أعلى AQI",style={"fontSize":"10px","color":"#888"}),
                               html.Div(str(st["max_aqi"]),style={"fontSize":"22px","fontWeight":"800","color":C["red"]})]),
                html.Div(style={"background":"white","border":f"2px solid {C['red']}","borderRadius":"8px","padding":"8px 14px","textAlign":"center"},
                    children=[html.Div("أعلى SMOKE",style={"fontSize":"10px","color":"#888"}),
                               html.Div(str(st["max_smoke"]),style={"fontSize":"22px","fontWeight":"800","color":C["red"]})]),
                html.Div(style={"background":"white","border":f"2px solid {C['red']}","borderRadius":"8px","padding":"8px 14px","textAlign":"center"},
                    children=[html.Div("قراءات طوارئ",style={"fontSize":"10px","color":"#888"}),
                               html.Div(str(st["emergency"]),style={"fontSize":"22px","fontWeight":"800","color":C["red"]})]),
            ]),
        ])

    # شارات
    badges=html.Div(className="section-card",children=[
        html.Div("🏅 مستويات الخطر المكتشفة",style={"fontSize":"12px","fontWeight":"800","color":"#555","marginBottom":"10px"}),
        html.Div(style={"display":"flex","gap":"12px","flexWrap":"wrap","alignItems":"center"},children=[
            html.Div(style={"display":"flex","alignItems":"center","gap":"6px"},children=[badge("آمن"),html.Span(str(st["safe"]),style={"fontWeight":"800","color":C["green"],"fontSize":"16px"})]),
            html.Div(style={"display":"flex","alignItems":"center","gap":"6px"},children=[badge("متوسط"),html.Span(str(st["moderate"]),style={"fontWeight":"800","color":C["warn"],"fontSize":"16px"})]),
            html.Div(style={"display":"flex","alignItems":"center","gap":"6px"},children=[badge("تحذير"),html.Span(str(st["warning"]),style={"fontWeight":"800","color":"#e65100","fontSize":"16px"})]),
            html.Div(style={"display":"flex","alignItems":"center","gap":"6px"},children=[badge("حرج"),html.Span(str(st["critical"]),style={"fontWeight":"800","color":C["red"],"fontSize":"16px"})]),
            html.Div(style={"display":"flex","alignItems":"center","gap":"6px"},children=[badge("طوارئ"),html.Span(str(st["emergency"]),style={"fontWeight":"800","color":C["blue"],"fontSize":"16px"})]),
        ]),
    ])

    # أزرار التنزيل
    dl_zone=html.Div(className="section-card",children=[
        html.Div(style={"display":"flex","alignItems":"center","gap":"8px","marginBottom":"14px"},children=[
            svg_download(),
            html.Div("تنزيل التقارير",style={"fontSize":"13px","fontWeight":"800","color":C["blue"]}),
        ]),
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(160px,1fr))","gap":"12px"},children=[
            dl_btn("تنزيل Excel كامل","btn-excel","green",svg_download()),
            dl_btn("تنزيل CSV","btn-csv","blue",svg_download()),
        ]),
        html.Div("Excel يحتوي: البيانات الكاملة + الملخص الإحصائي + القراءات الخطرة",
            style={"fontSize":"11px","color":"#888","marginTop":"8px","textAlign":"center"}),
    ])

    # عدادات
    gauges=html.Div(className="section-card",children=[
        html.Div("📊 عدادات القراءات الحية",style={"fontSize":"13px","fontWeight":"800","color":C["blue"],
            "marginBottom":"10px","borderBottom":f"2px solid {C['blue']}","paddingBottom":"4px"}),
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(200px,1fr))","gap":"12px"},children=[
            html.Div([html.Div("مؤشر جودة الهواء — AQI",style={"fontSize":"11px","fontWeight":"700","color":C["red"],"textAlign":"center","marginBottom":"4px"}),
                dcc.Graph(figure=gauge_fig(st["avg_aqi"],"AQI",500,C["red"]),config={"displayModeBar":False})],
                style={"background":"white","border":f"2px solid {C['red']}","borderRadius":"12px","padding":"10px"}),
            html.Div([html.Div("أول أكسيد الكربون — CO",style={"fontSize":"11px","fontWeight":"700","color":C["blue"],"textAlign":"center","marginBottom":"4px"}),
                dcc.Graph(figure=gauge_fig(st["avg_co"],"CO",800,C["blue"]),config={"displayModeBar":False})],
                style={"background":"white","border":f"2px solid {C['blue']}","borderRadius":"12px","padding":"10px"}),
            html.Div([html.Div("كثافة الدخان — SMOKE",style={"fontSize":"11px","fontWeight":"700","color":C["green"],"textAlign":"center","marginBottom":"4px"}),
                dcc.Graph(figure=gauge_fig(st["max_smoke"],"SMOKE",1100,C["green"]),config={"displayModeBar":False})],
                style={"background":"white","border":f"2px solid {C['green']}","borderRadius":"12px","padding":"10px"}),
        ]),
    ])

    # إحصاءات
    st_zone=html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(120px,1fr))",
        "gap":"12px","padding":"0 16px 16px"},children=[
        stat_card("إجمالي القراءات",f"{st['total']:,}","blue","📋"),
        stat_card("متوسط AQI",st["avg_aqi"],"red","🌫️"),
        stat_card("متوسط CO",st["avg_co"],"red","💨"),
        stat_card("أعلى SMOKE",st["max_smoke"],"red","🔥"),
        stat_card("آمن",st["safe"],"green","✅"),
        stat_card("تحذير",st["warning"],"warn","⚠️"),
        stat_card("حرج+طوارئ",st["critical"]+st["emergency"],"red","🚨"),
        stat_card("دقة RF",f"{ms['RF']['acc']*100:.1f}%","green","🎯"),
    ])

    # رسوم
    ch_zone=html.Div(className="section-card",children=[
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(300px,1fr))","gap":"16px"},children=[
            html.Div([html.Div("توزيع مستويات الخطر",style={"fontSize":"13px","fontWeight":"700","color":C["red"],"marginBottom":"6px"}),
                dcc.Graph(figure=bar_fig(df),config={"displayModeBar":False})],
                style={"background":"#fafafa","borderRadius":"12px","padding":"14px","border":"1.5px solid #e0e0e0"}),
            html.Div([html.Div("مؤشرات التلوث عبر الزمن",style={"fontSize":"13px","fontWeight":"700","color":C["blue"],"marginBottom":"6px"}),
                dcc.Graph(figure=timeline_fig(df),config={"displayModeBar":False})],
                style={"background":"#fafafa","borderRadius":"12px","padding":"14px","border":"1.5px solid #e0e0e0"}),
        ]),
    ])

    # خوارزميات
    al_zone=html.Div(className="section-card",children=[
        html.Div("خوارزميات التصنيف 🔴",style={"fontSize":"13px","fontWeight":"800","color":C["red"],
            "marginBottom":"8px","borderBottom":f"2px solid {C['red']}","paddingBottom":"4px"}),
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(260px,1fr))","gap":"10px","marginBottom":"16px"},children=[
            algo_card("Random Forest","تصنيف",f"يبني 200 شجرة قرار · دقة: {ms['RF']['acc']*100:.1f}%",
                ["n_estimators=200","max_depth=8","voting"],
                "🌲 يأخذ عينات عشوائية ويبني شجرة لكل عينة. الأغلبية تفوز. مثل لجنة خبراء تصوّت على الخطر.","red"),
            algo_card("SVM Classifier","تصنيف",f"يجد الحد الفاصل الأمثل · دقة: {ms['SVM']['acc']*100:.1f}%",
                ["kernel=rbf","C=1.0","multiclass"],
                "⚡ يرسم حداً فاصلاً بأكبر هامش بين المستويات. kernel=rbf يحول البيانات لأبعاد أعلى لفصلها بسهولة.","red"),
            algo_card("LSTM Neural Net","تصنيف زمني","يتذكر الأنماط الزمنية الطويلة",
                ["units=64","seq_len=10","dropout=0.2"],
                "🧠 تتذكر القراءات السابقة وتتعلم: إذا ارتفع AQI لـ 10 قراءات متتالية → الخطر قادم.","red"),
        ]),
        html.Div("خوارزميات كشف الشذوذات 🔵",style={"fontSize":"13px","fontWeight":"800","color":C["blue"],
            "marginBottom":"8px","borderBottom":f"2px solid {C['blue']}","paddingBottom":"4px"}),
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(260px,1fr))","gap":"10px","marginBottom":"16px"},children=[
            algo_card("LOF","كشف شذوذات",f"كثافة الجيران · شذوذات: {ms['LOF']['anom']}",
                ["n_neighbors=20","euclidean","novelty"],
                "📡 يقارن كثافة كل قراءة بجيرانها. نقطة وحيدة بعيدة عن المجموعة = شذوذ.","blue"),
            algo_card("One-Class SVM","كشف شذوذات",f"حدود طبيعية · شذوذات: {ms['OCSVM']['anom']}",
                ["kernel=rbf","nu=0.05","novelty=True"],
                "🎯 يرسم 'كرة' حول البيانات الطبيعية. أي قراءة خارج الكرة = شذوذ فوري.","blue"),
            algo_card("Isolation Forest","كشف شذوذات",f"عزل عشوائي · شذوذات: {ms['ISO']['anom']}",
                ["contamination=3%","random_state=42"],
                "🌲 يقسّم البيانات عشوائياً. القراءة الشاذة تُعزل بخطوات أقل لأنها بعيدة.","blue"),
        ]),
        html.Div("خوارزميات التنبؤ والتجميع 🟢",style={"fontSize":"13px","fontWeight":"800","color":C["green"],
            "marginBottom":"8px","borderBottom":f"2px solid {C['green']}","paddingBottom":"4px"}),
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(260px,1fr))","gap":"10px"},children=[
            algo_card("ARIMA","تنبؤ زمني","يتنبأ بمستوى التلوث للساعات القادمة",
                ["p=2,d=1,q=2","seasonal","forecast=24h"],
                "📈 يحلل الارتباط مع الماضي + مدى الاستقرار + تأثير الأخطاء. مثل توقعات الطقس.","green"),
            algo_card("Prophet","تنبؤ موسمي","يكتشف الأنماط اليومية والموسمية",
                ["yearly=True","daily=True","uncertainty"],
                "🔮 يفصل الاتجاه العام + الأنماط الموسمية. يعرف أن التلوث يرتفع صباحاً دون إخباره.","green"),
            algo_card("K-Means","تجميع",f"يجمع القراءات في 5 مجموعات",
                ["k=5","k-means++","unsupervised"],
                "🗂️ يختار 5 مراكز ويعيّن كل قراءة للأقرب. يجد الأنماط الخفية كذروة التلوث الصباحية.","green"),
        ]),
    ])

    # محاكاة
    sim_zone_el=html.Div(className="section-card",children=[
        html.Div(style={"display":"flex","alignItems":"center","gap":"10px","marginBottom":"14px"},children=[
            html.Div(className="icon-cpu",children=[
                html.Svg(viewBox="0 0 40 40",width="36",height="36",children=[
                    html.Rect(x="8",y="8",width="24",height="24",rx="3",fill="#9c27b0"),
                    html.Text("AI",x="20",y="23",textAnchor="middle",fill="white",fontSize="9",fontWeight="bold",fontFamily="Cairo"),
                ],xmlns="http://www.w3.org/2000/svg"),
            ]),
            html.Div("⚡ محاكاة قراءة جديدة — تشغيل الخوارزميات + تنبيه صوتي",
                style={"fontSize":"14px","fontWeight":"800","color":C["blue"]}),
        ]),
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(180px,1fr))","gap":"12px","marginBottom":"14px"},children=[
            html.Div([html.Label("AQI (0–462)",style={"fontSize":"12px","color":"#555","fontWeight":"600"}),
                dcc.Slider(0,462,1,value=94,id="s1",marks=None,tooltip={"always_visible":True})]),
            html.Div([html.Label("CO ppm (0–751)",style={"fontSize":"12px","color":"#555","fontWeight":"600"}),
                dcc.Slider(0,751,1,value=88,id="s2",marks=None,tooltip={"always_visible":True})]),
            html.Div([html.Label("SMOKE ppm (15–1017)",style={"fontSize":"12px","color":"#555","fontWeight":"600"}),
                dcc.Slider(15,1017,1,value=100,id="s3",marks=None,tooltip={"always_visible":True})]),
            html.Div([html.Label("Temperature °C",style={"fontSize":"12px","color":"#555","fontWeight":"600"}),
                dcc.Slider(20,38,.5,value=27,id="s4",marks=None,tooltip={"always_visible":True})]),
            html.Div([html.Label("Humidity %",style={"fontSize":"12px","color":"#555","fontWeight":"600"}),
                dcc.Slider(30,100,1,value=52,id="s5",marks=None,tooltip={"always_visible":True})]),
        ]),
        html.Button("⚡ تشغيل الخوارزميات + تنبيه صوتي",id="btn",n_clicks=0,
            style={"width":"100%","padding":"12px","borderRadius":"10px","border":"none",
                "background":C["blue"],"color":"white","fontSize":"15px","fontWeight":"800",
                "cursor":"pointer","fontFamily":"Cairo,sans-serif"}),
        html.Div(id="sim-out",style={"marginTop":"12px"}),
    ])

    return (status, alert_zone, badges, dl_zone, gauges, st_zone,
            ch_zone, al_zone, sim_zone_el, df.to_json(date_format="iso",orient="split"))

# ── تنزيل Excel ──
@app.callback(
    Output("dl-excel","data"),
    Input("btn-excel","n_clicks"),
    State("store","data"),
    prevent_initial_call=True,
)
def download_excel(n, store):
    if not store: return None
    df  = process_data(pd.read_json(io.StringIO(store),orient="split"))
    ms  = train_all(df)
    st  = stats(df)
    b64 = export_excel(df, st, ms)
    return {"base64":True,"content":b64,"filename":f"EnviroAI_Report_{datetime.date.today()}.xlsx",
            "type":"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}

# ── تنزيل CSV ──
@app.callback(
    Output("dl-csv","data"),
    Input("btn-csv","n_clicks"),
    State("store","data"),
    prevent_initial_call=True,
)
def download_csv(n, store):
    if not store: return None
    df  = process_data(pd.read_json(io.StringIO(store),orient="split"))
    b64 = export_csv(df)
    return {"base64":True,"content":b64,"filename":f"EnviroAI_Data_{datetime.date.today()}.csv",
            "type":"text/csv"}

# ── محاكاة ──
@app.callback(
    Output("sim-out","children"),
    Input("btn","n_clicks"),
    State("s1","value"),State("s2","value"),State("s3","value"),
    State("s4","value"),State("s5","value"),State("store","data"),
    prevent_initial_call=True,
)
def run_sim(n,aqi,co,smoke,temp,hum,store):
    if not store:
        return html.Div("ارفع ملف Excel أولاً",style={"color":C["red"],"fontSize":"13px"})
    df=process_data(pd.read_json(io.StringIO(store),orient="split"))
    ms=train_all(df); r=sim(aqi,co,smoke,temp,hum,ms)
    lv=r["level"]; col=LCOLOR.get(lv,C["red"]); bg=LBG.get(lv,C["red_bg"])
    return html.Div(style={"background":bg,"border":f"2px solid {col}","borderRadius":"10px","padding":"14px"},children=[
        html.Span(**{"data-level":lv},style={"display":"none"}),
        html.Div(style={"display":"flex","alignItems":"center","gap":"10px","marginBottom":"10px"},children=[
            badge(lv),
            html.Span(f"RiskIndex = {r['ri']}",style={"fontSize":"15px","fontWeight":"800","color":col}),
        ]),
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(140px,1fr))","gap":"8px"},children=[
            html.Div([html.Span("🌲 RF: ",style={"fontWeight":"700","fontSize":"11px"}),html.Span(r["rf"])],
                style={"background":"white","border":f"1px solid {C['red']}","borderRadius":"6px","padding":"6px 10px","fontSize":"12px"}),
            html.Div([html.Span("⚡ SVM: ",style={"fontWeight":"700","fontSize":"11px"}),html.Span(r["svm"])],
                style={"background":"white","border":f"1px solid {C['red']}","borderRadius":"6px","padding":"6px 10px","fontSize":"12px"}),
            html.Div([html.Span("📡 LOF: ",style={"fontWeight":"700","fontSize":"11px"}),html.Span(r["lof"])],
                style={"background":"white","border":f"1px solid {C['blue']}","borderRadius":"6px","padding":"6px 10px","fontSize":"12px"}),
            html.Div([html.Span("🎯 OC-SVM: ",style={"fontWeight":"700","fontSize":"11px"}),html.Span(r["oc"])],
                style={"background":"white","border":f"1px solid {C['blue']}","borderRadius":"6px","padding":"6px 10px","fontSize":"12px"}),
            html.Div([html.Span("🌲 ISO: ",style={"fontWeight":"700","fontSize":"11px"}),html.Span(r["iso"])],
                style={"background":"white","border":f"1px solid {C['blue']}","borderRadius":"6px","padding":"6px 10px","fontSize":"12px"}),
            html.Div([html.Span("🗂️ K-Means: ",style={"fontWeight":"700","fontSize":"11px"}),html.Span(str(r["km"]))],
                style={"background":"white","border":f"1px solid {C['green']}","borderRadius":"6px","padding":"6px 10px","fontSize":"12px"}),
        ]),
    ])

if __name__=="__main__":
    port=int(os.environ.get("PORT",8050))
    print("\n"+"="*56)
    print("  EnviroAI Pro v3 — تنزيل + صور متحركة + صوت")
    print("="*56)
    print(f"  افتح: http://localhost:{port}")
    print("="*56+"\n")
    app.run(debug=False,host="0.0.0.0",port=port)
