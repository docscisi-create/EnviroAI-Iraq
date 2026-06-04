"""
EnviroAI Chatbot v3 — طالب جامعي + شرح الخوارزميات + تصميم محسّن
"""
import os, warnings, base64, io, datetime
import numpy as np, pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import dash
from dash import dcc, html, Input, Output, State, ctx
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
# قاعدة المعرفة
# ══════════════════════════════════════════════════════
KB = {
    "aqi": {
        "ar": "مؤشر جودة الهواء (Air Quality Index — AQI):\n• آمن (Safe): 0–50 ✅ — لا يوجد خطر\n• متوسط (Moderate): 51–100 🟡 — تأثير على الحساسين\n• تحذير (Unhealthy for Sensitive): 101–150 🟠 — تجنب الخارج\n• حرج (Unhealthy): 151–200 🔴 — تأثير على الجميع\n• طوارئ (Hazardous): فوق 200 🚨 — إخلاء فوري\nالمصدر: US EPA & WHO",
        "en": "Air Quality Index (AQI) Levels:\n• Safe: 0-50 ✅ — No risk\n• Moderate: 51-100 🟡 — Minor effect on sensitive groups\n• Unhealthy for Sensitive: 101-150 🟠 — Avoid outdoor activity\n• Unhealthy: 151-200 🔴 — Affects everyone\n• Hazardous: >200 🚨 — Emergency evacuation\nSource: US EPA & WHO"
    },
    "co": {
        "ar": "أول أكسيد الكربون (Carbon Monoxide — CO) — مستشعر MQ-7:\n• آمن (Safe): أقل من 9 ppm (معيار WHO)\n• تحذير (Warning): 9–35 ppm — صداع\n• خطر (Danger): 35–200 ppm — دوار وتعب\n• حرج (Critical): 200–400 ppm — خطر على الحياة\n• طوارئ (Emergency): فوق 400 ppm — مميت ⚠️",
        "en": "Carbon Monoxide (CO) — MQ-7 Sensor:\n• Safe: <9 ppm (WHO standard)\n• Warning: 9-35 ppm — headache on exposure\n• Danger: 35-200 ppm — dizziness & fatigue\n• Critical: 200-400 ppm — life-threatening\n• Emergency: >400 ppm — lethal ⚠️"
    },
    "smoke": {
        "ar": "الدخان (Smoke Density) — مستشعر MQ-2:\n• طبيعي (Normal): أقل من 50 ppm\n• تحذير (Warning): 50–100 ppm — تهيج خفيف\n• متوسط (Moderate): 100–200 ppm — سعال\n• خطر (Danger): 200–500 ppm — صعوبة تنفس\n• طوارئ (Emergency): فوق 500 ppm — إخلاء فوري 🔥",
        "en": "Smoke Density — MQ-2 Sensor:\n• Normal: <50 ppm\n• Warning: 50-100 ppm — mild irritation\n• Moderate: 100-200 ppm — coughing\n• Danger: 200-500 ppm — breathing difficulty\n• Emergency: >500 ppm — evacuate immediately 🔥"
    },
    "temp": {
        "ar": "درجة الحرارة والرطوبة (Temperature & Humidity) — DHT22:\n• مثالي للمختبرات (Ideal): 20–24 °C\n• مقبول (Acceptable): 18–27 °C\n• تحذير (Warning): فوق 27°C أو تحت 18°C\n• حرج (Critical): فوق 35°C — خطر على الأجهزة\n• الرطوبة المثلى (Ideal RH): 40–60%",
        "en": "Temperature & Humidity (Temp/RH) — DHT22 Sensor:\n• Ideal for labs: 20-24°C\n• Acceptable: 18-27°C\n• Warning: above 27°C or below 18°C\n• Critical: above 35°C — risk to electronics\n• Ideal Relative Humidity (RH): 40-60%"
    },
    "sensors": {
        "ar": "المستشعرات (Sensors) المستخدمة:\n🌡️ DHT22 — Temperature & Humidity (الحرارة والرطوبة)\n💨 MQ-135 — Air Quality Index (جودة الهواء العامة)\n🔥 MQ-2 — Smoke & Flammable Gas (الدخان والغاز)\n☠️ MQ-7 — Carbon Monoxide (أول أكسيد الكربون)\n🔌 Arduino UNO — Microcontroller (معالجة البيانات بـ C++)",
        "en": "Sensors Used:\n🌡️ DHT22 — Temperature & Humidity sensor\n💨 MQ-135 — Air Quality Index (AQI) sensor\n🔥 MQ-2 — Smoke & Flammable Gas sensor\n☠️ MQ-7 — Carbon Monoxide (CO) sensor\n🔌 Arduino UNO — Microcontroller (C++ programming)"
    },
    "ai": {
        "ar": "خوارزميات الذكاء الاصطناعي (AI Algorithms) في النظام:\n\n🌲 Random Forest (غابة عشوائية) — يبني 200 Decision Tree ويأخذ تصويت الأغلبية (Majority Vote). دقة: 99%.\n\n⚡ SVM — Support Vector Machine — يجد الـ Hyperplane الأمثل بين مستويات الخطر. دقة: 99%.\n\n📡 LOF — Local Abnormal Factor — يكتشف الـ Abnormal Readings بمقارنة كثافة كل نقطة بـ Neighbors. بدون تسميات.\n\n🎯 One-Class SVM — يرسم Boundary حول البيانات الطبيعية. أي قراءة خارجها = Abnormal قراءة غير طبيعية.\n\n🌳 Isolation Forest — يعزل الغير طبيعي بـ Random Trees. الغير طبيعي يُعزل بخطوات أقل (Shorter Path).\n\n🗂️ K-Means Clustering — يجمع القراءات في 5 Clusters لاكتشاف الأنماط الخفية.",
        "en": "AI Algorithms Used in the System:\n\n🌲 Random Forest — Builds 200 Decision Trees and takes Majority Vote for classification. Accuracy: 99%.\n\n⚡ SVM — Support Vector Machine — Finds optimal Hyperplane boundary between risk levels. Accuracy: 99%.\n\n📡 LOF — Local Abnormal Factor — Detects Abnormal Readings by comparing each point's density to its Neighbors. Unsupervised.\n\n🎯 One-Class SVM — Draws a Boundary around normal data. Any reading outside = Abnormal (Abnormal Reading).\n\n🌳 Isolation Forest — Isolates abnormals using Random Trees. Abnormal Readings need shorter Isolation Paths.\n\n🗂️ K-Means Clustering — Groups readings into 5 Clusters to discover hidden patterns."
    },
    "risk": {
        "ar": "معادلة مؤشر الخطر (Risk Index Formula):\nRiskIndex = 0.40×AQI + 0.30×CO + 0.30×SMOKE\n\n• AQI (Air Quality Index) وزنه 40%\n• CO (Carbon Monoxide) وزنه 30%\n• SMOKE (Smoke Density) وزنه 30%\n\nمثال تطبيقي (Practical Example):\nAQI=100, CO=80, SMOKE=60\nRiskIndex = (0.40×100)+(0.30×80)+(0.30×60) = 40+24+18 = 82 → متوسط 🟡\n\n🟢 <50 Safe آمن\n🟡 50–100 Moderate متوسط\n🟠 100–150 Warning تحذير\n🔴 150–200 Critical حرج\n🚨 >200 Emergency طوارئ",
        "en": "Risk Index Formula:\nRiskIndex = 0.40×AQI + 0.30×CO + 0.30×SMOKE\n\n• AQI (Air Quality Index) = 40% weight — highest health impact\n• CO (Carbon Monoxide) = 30% weight — immediate danger\n• SMOKE (Smoke Density) = 30% weight — cumulative damage\n\nPractical Example:\nAQI=100, CO=80, SMOKE=60\nRiskIndex = (0.40×100)+(0.30×80)+(0.30×60) = 40+24+18 = 82 → Moderate 🟡\n\n🟢 <50 Safe | 🟡 50-100 Moderate\n🟠 100-150 Warning | 🔴 150-200 Critical | 🚨 >200 Emergency"
    },
    "system": {
        "ar": "نظام EnviroAI Pro:\n🎓 بحث تخرج (Graduation Project) — المستنصرية\n🏛️ الجامعة المستنصرية (Mustansiriyah University)\n📚 قسم علوم الحاسوب (Department of Computer Science)\n👩‍🏫 إشراف: أ.م. ياسمين مكي محيالدين\n📅 2025–2026\n\nأجهزة (Hardware): Arduino UNO + DHT22 + MQ-135 + MQ-2 + MQ-7\nبرمجيات (Software): Python + Dash + 6 خوارزميات AI\nهدف (Goal): مراقبة جودة الهواء الداخلي في المختبرات",
        "en": "EnviroAI Pro System:\n🎓 Graduation Project — Noor Basim Majeed\n🏛️ Mustansiriyah University\n📚 Department of Computer Science\n👩‍🏫 Supervisor: Asst.Prof. Yasmin Makki Mohialden\n📅 2025–2026\n\nHardware: Arduino UNO + DHT22 + MQ-135 + MQ-2 + MQ-7\nSoftware: Python + Dash + 6 AI Algorithms\nGoal: Monitor indoor air quality in Computer Science labs"
    }
}

def chatbot_reply(q, lang="ar"):
    q = q.lower()
    if any(w in q for w in ["aqi","جودة الهواء","air quality","مؤشر","هواء","air"]):
        return KB["aqi"][lang]
    if any(w in q for w in ["co","كربون","carbon","mq-7","mq7","أكسيد","monoxide"]):
        return KB["co"][lang]
    if any(w in q for w in ["smoke","دخان","غاز","mq-2","mq2","حريق","smok"]):
        return KB["smoke"][lang]
    if any(w in q for w in ["temp","حرارة","temperature","رطوبة","humidity","dht","درجة","rh","humid"]):
        return KB["temp"][lang]
    if any(w in q for w in ["sensor","مستشعر","arduino","جهاز","hardware","mq","أجهزة","sensors"]):
        return KB["sensors"][lang]
    if any(w in q for w in ["algorithm","خوارزمية","forest","svm","kmeans","lof","ذكاء","خوارزم","cluster","isolation"]):
        return KB["ai"][lang]
    if any(w in q for w in ["risk","خطر","riskindex","معادلة","formula","حساب","مؤشر الخطر","index"]):
        return KB["risk"][lang]
    if any(w in q for w in ["system","نظام","نور","مشروع","project","enviro","تخرج","جامعة","university"]):
        return KB["system"][lang]
    if lang == "ar":
        return ("لم أجد إجابة محددة لسؤالك.\n\n"
                "يمكنني الإجابة عن:\n"
                "• AQI — جودة الهواء (Air Quality Index)\n"
                "• CO — أول أكسيد الكربون (Carbon Monoxide)\n"
                "• دخان — Smoke مستويات الأمان\n"
                "• حرارة — Temperature & Humidity\n"
                "• مستشعرات — Sensors: DHT22·MQ-135·MQ-2·MQ-7\n"
                "• خوارزميات — AI Algorithms: RF·SVM·LOF·KMeans\n"
                "• معادلة — Risk Index Formula\n"
                "• نظام — System info")
    return ("No specific answer found.\n\n"
            "I can answer about:\n"
            "• AQI — Air Quality Index levels\n"
            "• CO — Carbon Monoxide danger\n"
            "• Smoke — Safety levels (MQ-2)\n"
            "• Temperature & Humidity (DHT22)\n"
            "• Sensors: DHT22·MQ-135·MQ-2·MQ-7\n"
            "• AI Algorithms: RF·SVM·LOF·IsoForest·KMeans\n"
            "• Risk Index Formula with example\n"
            "• System overview")


# ══ بيانات ══
def process_data(df):
    df=df.copy(); df.fillna(df.median(numeric_only=True),inplace=True)
    df["RiskIndex"]=0.40*df["AQI"]+0.30*df["CO"]+0.30*df["SMOKE"]
    df["RiskLevel"]=pd.cut(df["RiskIndex"],[-np.inf,50,100,150,200,np.inf],
        labels=["آمن","متوسط","تحذير","حرج","طوارئ"])
    return df

def train_all(df):
    feat=["AQI","CO","SMOKE","Temperature","Humidity"]; X=df[feat].values
    le=LabelEncoder(); y=le.fit_transform(df["RiskLevel"].astype(str))
    sc=StandardScaler(); Xs=sc.fit_transform(X)
    Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.2,random_state=42,stratify=y)
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
    m["KM"]={"model":km}; m["_sc"]=sc; m["_le"]=le; m["_feat"]=feat
    return m

def get_stats(df):
    c=df["RiskLevel"].value_counts().to_dict()
    return {"total":len(df),"avg_aqi":round(df["AQI"].mean(),1),"max_aqi":int(df["AQI"].max()),
            "avg_co":round(df["CO"].mean(),1),"max_smoke":int(df["SMOKE"].max()),
            "safe":c.get("آمن",0),"moderate":c.get("متوسط",0),"warning":c.get("تحذير",0),
            "critical":c.get("حرج",0),"emergency":c.get("طوارئ",0)}

def bar_fig(df):
    c=df["RiskLevel"].value_counts().reindex(["آمن","متوسط","تحذير","حرج","طوارئ"],fill_value=0)
    fig=go.Figure(go.Bar(x=c.index.tolist(),y=c.values,
        marker_color=["#43a047","#fdd835","#ff9800","#e53935","#1565c0"],text=c.values,textposition="outside"))
    fig.update_layout(paper_bgcolor="white",plot_bgcolor="#f8f9fa",
        font=dict(family="Cairo",color="#1a2a3a"),margin=dict(t=20,b=20,l=20,r=20),height=230,
        xaxis=dict(showgrid=False),yaxis=dict(gridcolor="#eee"))
    return fig

def timeline_fig(df):
    fig=go.Figure()
    for col,c in [("AQI","#e53935"),("CO","#1565c0"),("SMOKE","#43a047")]:
        fig.add_trace(go.Scatter(x=df.index,y=df[col],mode="lines",name=col,line=dict(color=c,width=1.5)))
    fig.add_trace(go.Scatter(x=[df.index[0],df.index[-1]],y=[200,200],mode="lines",
        name="حد الطوارئ",line=dict(color="#e53935",width=2,dash="dash")))
    fig.update_layout(paper_bgcolor="white",plot_bgcolor="#f8f9fa",
        font=dict(family="Cairo",color="#1a2a3a"),margin=dict(t=20,b=30,l=20,r=20),height=230,
        legend=dict(orientation="h",y=-0.3),xaxis=dict(showgrid=False),yaxis=dict(gridcolor="#eee"))
    return fig

def export_excel(df,st,ms):
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as w:
        df.to_excel(w,sheet_name="البيانات",index=False)
        pd.DataFrame({"المقياس":["إجمالي","متوسط AQI","أعلى AQI","متوسط CO","أعلى SMOKE",
            "آمن","متوسط","تحذير","حرج","طوارئ","دقة RF"],
            "القيمة":[st["total"],st["avg_aqi"],st["max_aqi"],st["avg_co"],st["max_smoke"],
                st["safe"],st["moderate"],st["warning"],st["critical"],st["emergency"],
                f"{ms['RF']['acc']*100:.1f}%"]}).to_excel(w,sheet_name="الملخص",index=False)
        df[df["RiskLevel"].isin(["حرج","طوارئ"])].to_excel(w,sheet_name="الخطرة",index=False)
    return base64.b64encode(buf.getvalue()).decode()

# ══ CSS ══
CSS="""
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Cairo,sans-serif;background:#f1f8e9}

@keyframes pulse{0%,100%{opacity:1}50%{opacity:.2}}
@keyframes ping{0%{transform:scale(1);opacity:1}100%{transform:scale(2.6);opacity:0}}
@keyframes slideDown{from{transform:translateY(-14px);opacity:0}to{transform:translateY(0);opacity:1}}
@keyframes popIn{0%{transform:scale(.8);opacity:0}100%{transform:scale(1);opacity:1}}
@keyframes shimmer{0%,100%{border-color:#e53935}50%{border-color:#ff1744}}
@keyframes float{0%,100%{transform:translateY(0)}40%{transform:translateY(-8px)}70%{transform:translateY(2px)}}
@keyframes wiggle{0%,100%{transform:rotate(0deg)}25%{transform:rotate(-10deg)}75%{transform:rotate(10deg)}}
@keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
@keyframes wave{0%{transform:scaleY(1)}50%{transform:scaleY(.2)}100%{transform:scaleY(1)}}
@keyframes glow{0%,100%{box-shadow:0 0 4px rgba(229,57,53,.3)}50%{box-shadow:0 0 18px rgba(229,57,53,.85)}}
@keyframes msgIn{from{transform:translateY(6px);opacity:0}to{transform:translateY(0);opacity:1}}
@keyframes typingDot{0%,80%,100%{transform:scale(0)}40%{transform:scale(1)}}
@keyframes countUp{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}
@media print{
  header, .card:first-child, #chat-win, .speak-btn, button:not(#btn-print) {display:none!important}
  body{background:white!important}
  .card{box-shadow:none!important;border:1px solid #ccc!important}
}
/* 3D Text effect */
.text-3d {
  color: #fff;
  text-shadow:
    0 1px 0 #ccc,
    0 2px 0 #c9c9c9,
    0 3px 0 #bbb,
    0 4px 0 #b9b9b9,
    0 5px 0 #aaa,
    0 6px 1px rgba(0,0,0,.1),
    0 0 5px rgba(0,0,0,.1),
    0 1px 3px rgba(0,0,0,.3),
    0 3px 5px rgba(0,0,0,.2),
    0 5px 10px rgba(0,0,0,.25),
    0 10px 10px rgba(0,0,0,.2),
    0 20px 20px rgba(0,0,0,.15);
}
.text-3d-green {
  color: #2e7d32;
  text-shadow:
    1px 1px 0 #a5d6a7,
    2px 2px 0 #81c784,
    3px 3px 0 #66bb6a,
    4px 4px 1px rgba(0,0,0,.15),
    0 0 8px rgba(67,160,71,.3);
}
.text-3d-blue {
  color: #1565c0;
  text-shadow:
    1px 1px 0 #90caf9,
    2px 2px 0 #64b5f6,
    3px 3px 0 #42a5f5,
    4px 4px 1px rgba(0,0,0,.15);
}
.text-3d-red {
  color: #c62828;
  text-shadow:
    1px 1px 0 #ef9a9a,
    2px 2px 0 #e57373,
    3px 3px 0 #ef5350,
    4px 4px 1px rgba(0,0,0,.15);
}

/* 3D Cards */
.card-3d {
  background: white;
  border-radius: 16px;
  padding: 18px;
  margin: 0 14px 14px;
  box-shadow:
    0 2px 0 #e0e0e0,
    0 4px 0 #d6d6d6,
    0 6px 0 #ccc,
    0 8px 15px rgba(0,0,0,.15),
    0 10px 25px rgba(0,0,0,.1);
  transform: translateY(-2px);
  transition: transform .2s, box-shadow .2s;
  border: 1px solid #e8e8e8;
}
.card-3d:hover {
  transform: translateY(-5px);
  box-shadow:
    0 5px 0 #e0e0e0,
    0 8px 0 #d6d6d6,
    0 10px 0 #ccc,
    0 14px 20px rgba(0,0,0,.18),
    0 16px 30px rgba(0,0,0,.12);
}

/* 3D algo cards */
.algo-card-3d {
  border-radius: 14px;
  padding: 16px;
  border: 2px solid;
  box-shadow:
    3px 3px 0 rgba(0,0,0,.1),
    5px 5px 0 rgba(0,0,0,.06),
    7px 7px 12px rgba(0,0,0,.12);
  transform: perspective(400px) rotateX(1deg);
  transition: transform .2s, box-shadow .2s;
}
.algo-card-3d:hover {
  transform: perspective(400px) rotateX(0deg) translateY(-4px);
  box-shadow:
    4px 6px 0 rgba(0,0,0,.12),
    6px 8px 0 rgba(0,0,0,.08),
    8px 12px 18px rgba(0,0,0,.15);
}

/* Sensor shapes */
/* Sensor shapes */
.sensor-shape{display:flex;justify-content:center;align-items:center;margin:4px auto}
.sensor-chip{width:28px;height:20px;background:#1565c0;border-radius:3px;border:2px solid #0d47a1;
  display:flex;align-items:center;justify-content:center;font-size:7px;color:white;font-weight:800}
.sensor-dht{width:18px;height:28px;background:#e53935;border-radius:3px 3px 6px 6px;
  border:2px solid #b71c1c;display:flex;align-items:center;justify-content:center;
  font-size:7px;color:white;font-weight:800}
.sensor-mq{width:22px;height:22px;border-radius:50%;background:#ff9800;border:3px solid #e65100;
  display:flex;align-items:center;justify-content:center;font-size:7px;color:white;font-weight:800}
.sensor-arduino{width:32px;height:20px;background:#00979d;border-radius:2px;border:2px solid #007a80;
  display:flex;align-items:center;justify-content:center;font-size:7px;color:white;font-weight:800}

/* Environmental color palette */
/* Environmental color palette */
:root {
  --env-dark:   #388e3c;
  --env-mid:    #388e3c;
  --env-light:  #43a047;
  --env-pale:   #e8f5e9;
  --env-sky:    #e3f2fd;
  --env-earth:  #795548;
  --env-sun:    #f9a825;
  --env-water:  #0288d1;
  --env-alert:  #e53935;
}

/* Override card background to light green */
.card { background: white; border-left: 4px solid #81c784; }

/* Marquee ticker with env colors */
.ticker-wrap { background: linear-gradient(90deg, #388e3c, #388e3c, #388e3c); }

/* Chat window env style */
.chat-window { border: 2px solid var(--env-light) !important; }
.msg-bot .bubble { background: #f1f8e9; border-color: #aed581; }

/* Quick buttons env */
.quick-btn-env {
  background: var(--env-pale) !important;
  color: var(--env-dark) !important;
  border-color: var(--env-light) !important;
}
@keyframes studentBob{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
@keyframes eyeBlink{0%,88%,100%{transform:scaleY(1)}94%{transform:scaleY(.05)}}
@keyframes typing{0%,100%{opacity:1}50%{opacity:0.3}}

/* marquee */
@keyframes marquee{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}

/* student figure */
.student-fig{animation:studentBob 3s ease-in-out infinite;display:block;text-align:center}

/* wave bars */
.wave-bar{display:inline-block;width:4px;height:18px;margin:0 1px;border-radius:3px;animation:wave 1s ease-in-out infinite}
.wave-bar:nth-child(2){animation-delay:.1s;height:12px}
.wave-bar:nth-child(3){animation-delay:.2s;height:22px}
.wave-bar:nth-child(4){animation-delay:.3s;height:14px}
.wave-bar:nth-child(5){animation-delay:.4s;height:9px}

/* live dot */
.live-dot{display:inline-block;width:9px;height:9px;background:#43a047;border-radius:50%;animation:pulse 1.1s infinite;vertical-align:middle}
/* ping */
.ping-wrap{position:relative;display:inline-block;width:16px;height:16px}
.ping-core{position:absolute;inset:2px;background:#e53935;border-radius:50%}
.ping-ring{position:absolute;inset:-4px;border:2px solid #e53935;border-radius:50%;animation:ping 1.2s infinite}

/* icon cards */
.icon-card{border-radius:12px;padding:12px 8px;text-align:center;cursor:pointer;
  border:3px solid #43a047;transition:transform .18s,box-shadow .18s,border-color .18s}
.icon-card:hover{transform:translateY(-4px);box-shadow:0 6px 20px rgba(67,160,71,.25)}
.icon-card.selected{transform:translateY(-4px);box-shadow:0 6px 24px rgba(67,160,71,.35);border-color:#388e3c;border-width:3px}
.icon-emoji{font-size:32px;line-height:1;animation:float 3s ease-in-out infinite;display:block}
.icon-label{font-size:11px;font-weight:800;margin-top:6px}

/* chat */
.chat-window{height:420px;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:10px;scroll-behavior:smooth}
.msg-bot{animation:msgIn .3s ease;align-self:flex-start;max-width:86%}
.msg-user{animation:msgIn .3s ease;align-self:flex-end;max-width:86%}
.msg-bot .bubble{background:#f1f8e9;border:1.5px solid #aed581;border-radius:12px 12px 12px 3px;
  padding:12px 16px;font-size:13.5px;line-height:1.9;color:#1a2a3a;white-space:pre-wrap}
.msg-user .bubble{background:#1565c0;color:white;border-radius:12px 12px 3px 12px;
  padding:12px 16px;font-size:13.5px;line-height:1.75}
.msg-label{font-size:10px;margin-bottom:2px;font-weight:700;opacity:.5}
.speak-btn{background:none;border:1px solid #90caf9;border-radius:10px;padding:3px 9px;
  font-size:10px;cursor:pointer;color:#1565c0;margin-top:3px;font-family:Cairo,sans-serif;
  transition:background .15s}
.speak-btn:hover{background:#e8f0fe}
.typing-dot{display:inline-block;width:6px;height:6px;background:#1565c0;border-radius:50%;margin:0 2px}
.typing-dot:nth-child(1){animation:typingDot 1.2s .0s infinite}
.typing-dot:nth-child(2){animation:typingDot 1.2s .2s infinite}
.typing-dot:nth-child(3){animation:typingDot 1.2s .4s infinite}

/* lang buttons */
.lang-btn{border:2px solid;border-radius:20px;padding:5px 14px;font-size:12px;font-weight:800;
  cursor:pointer;transition:all .15s;font-family:Cairo,sans-serif;white-space:nowrap}

/* badges */
.badge-base{border-radius:16px;padding:4px 12px;font-size:12px;font-weight:800;display:inline-flex;align-items:center;gap:4px;white-space:nowrap}
.badge-safe{background:#e8f5e9;color:#388e3c;border:2px solid #43a047}
.badge-moderate{background:#fff9c4;color:#f57f17;border:2px solid #fdd835}
.badge-warning{background:#fff3e0;color:#e65100;border:2px solid #ff9800}
.badge-critical{background:#ffebee;color:#b71c1c;border:2px solid #e53935}
.badge-emergency{background:#e3f2fd;color:#0d47a1;border:2px solid #1565c0;animation:glow 1.2s infinite}
.alert-anim{animation:slideDown .5s ease,shimmer 1.5s infinite}
.stat-card{animation:countUp .4s ease;transition:transform .15s,box-shadow .15s;border-radius:12px;padding:12px 14px;text-align:center}
.stat-card:hover{transform:translateY(-3px);box-shadow:0 6px 20px rgba(0,0,0,.09)}
.card{background:white;border-radius:14px;padding:18px;margin:0 14px 14px;}
"""

# ══ JS ══
JS="""
var _arVoice=null, _enVoice=null, _muted=false;

function loadVoices(){
  var vv=window.speechSynthesis.getVoices();
  _arVoice=vv.find(v=>v.lang==='ar-SA')||vv.find(v=>v.lang.startsWith('ar'))||null;
  _enVoice=vv.find(v=>v.lang==='en-US')||vv.find(v=>v.lang.startsWith('en'))||null;
  console.log('Voices:',vv.length,'| AR:',_arVoice&&_arVoice.name,'| EN:',_enVoice&&_enVoice.name);
}
loadVoices();
if(window.speechSynthesis.onvoiceschanged!==undefined) window.speechSynthesis.onvoiceschanged=loadVoices;
setTimeout(loadVoices,1500);
setTimeout(loadVoices,3000);

function speakNow(text,lang){
  if(_muted)return;
  loadVoices();
  try{
    window.speechSynthesis.cancel();
    var u=new SpeechSynthesisUtterance(text);
    u.lang=lang==='en'?'en-US':'ar-SA';
    u.voice=lang==='en'?_enVoice:_arVoice;
    u.rate=0.85; u.pitch=1.0; u.volume=1.0;
    u.onerror=function(e){console.warn('TTS:',e.error);};
    window.speechSynthesis.speak(u);
  }catch(e){console.warn(e);}
}

function playAlert(level){
  try{
    var ctx=new(window.AudioContext||window.webkitAudioContext)();
    var cfg={emergency:[[880,.13],[660,.13],[880,.13],[660,.13],[880,.26]],
             critical:[[660,.22],[440,.22]],warning:[[500,.26],[380,.16]],
             moderate:[[330,.26]],safe:[[220,.36]]};
    var key=level.includes('طوارئ')||level.includes('Emergency')?'emergency':
            level.includes('حرج')||level.includes('Critical')?'critical':
            level.includes('تحذير')||level.includes('Warning')?'warning':
            level.includes('متوسط')||level.includes('Moderate')?'moderate':'safe';
    var type=key==='emergency'?'sawtooth':key==='critical'?'square':'sine';
    var t=ctx.currentTime;
    (cfg[key]||cfg.safe).forEach(function(fd){
      var o=ctx.createOscillator(),g=ctx.createGain();
      o.connect(g);g.connect(ctx.destination);
      o.frequency.value=fd[0];o.type=type;
      g.gain.setValueAtTime(.3,t);
      g.gain.exponentialRampToValueAtTime(.001,t+fd[1]);
      o.start(t);o.stop(t+fd[1]);t+=fd[1]+.04;
    });
  }catch(e){}
}

window.addEventListener('DOMContentLoaded',function(){
  // Only English voice for speak buttons - Arabic TTS unreliable
  document.addEventListener('click',function(e){
    loadVoices();
    // speak button
    var spk=e.target.closest('[data-speak-text]');
    if(spk){ speakNow(spk.getAttribute('data-speak-text'),'en'); }
    // print button
    var prn=e.target.closest('[data-action="print"]');
    if(prn){
      var style = '<style>body{font-family:Arial,sans-serif;direction:rtl;padding:20px}' +
        '.no-print{display:none!important}.print-only{display:block}' +
        'h2{color:#2e7d32;border-bottom:3px solid #43a047;padding-bottom:8px}' +
        '.result-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}' +
        '.result-box{border:2px solid #ccc;border-radius:8px;padding:12px;text-align:center}' +
        '.result-box .val{font-size:24px;font-weight:800;color:#1565c0}' +
        '.result-box .lbl{font-size:11px;color:#888}' +
        '</style>';
      var printContent = document.getElementById('stats-zone') ? 
        document.getElementById('stats-zone').innerHTML : '';
      var summaryContent = document.querySelector('[data-print="summary"]') ?
        document.querySelector('[data-print="summary"]').innerHTML : '';
      var win = window.open('','_blank','width=900,height=700');
      win.document.write('<html><head><meta charset="UTF-8">' + style + 
        '</head><body dir="rtl">' +
        '<h2>🌿 تقرير نتائج التحليل البيئي — EnviroAI Pro</h2>' +
        '<p style="color:#888;font-size:12px">تاريخ التقرير: ' + new Date().toLocaleDateString('ar-IQ') + '</p>' +
        summaryContent + printContent +
        '<p style="margin-top:20px;font-size:11px;color:#aaa">نور باسم مجيد — الجامعة المستنصرية 2025-2026</p>' +
        '</body></html>');
      win.document.close();
      win.focus();
      setTimeout(function(){ win.print(); }, 500);
    }
  });
  new MutationObserver(function(ms){
    ms.forEach(function(m){m.addedNodes.forEach(function(n){
      if(n.nodeType===1){
        var al=n.querySelector&&n.querySelector('[data-level]');
        if(al)playAlert(al.getAttribute('data-level'));
      }
    });});
  }).observe(document.body,{childList:true,subtree:true});
});
"""

# ══ app ══
app = dash.Dash(__name__, title="EnviroAI Chatbot 🤖")
server = app.server

app.index_string = (
    "<!DOCTYPE html><html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}"
    "<style>" + CSS + "</style><script>" + JS + "</script>"
    "</head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"
)

C = {
    "red"    :"#c62828","red_bg"    :"#ffebee","red_dk"    :"#b71c1c",
    "blue"   :"#1565c0","blue_bg"   :"#e3f2fd","blue_dk"   :"#0d47a1",
    "green"  :"#43a047","green_bg"  :"#e8f5e9","green_dk"  :"#2e7d32",
    "warn"   :"#f57f17","warn_bg"   :"#fff8e1","warn_dk"   :"#e65100",
    "purple" :"#6a1b9a","purple_bg" :"#f3e5f5","purple_dk" :"#4a148c",
}
LCOLOR={"آمن":C["green"],"متوسط":C["warn"],"تحذير":"#e65100","حرج":C["red"],"طوارئ":C["blue"]}
LBG={"آمن":C["green_bg"],"متوسط":C["warn_bg"],"تحذير":"#fbe9e7","حرج":C["red_bg"],"طوارئ":C["blue_bg"]}
LBADGE={"آمن":"badge-safe","متوسط":"badge-moderate","تحذير":"badge-warning","حرج":"badge-critical","طوارئ":"badge-emergency"}
LICON={"آمن":"✅","متوسط":"🟡","تحذير":"🟠","حرج":"🔴","طوارئ":"🚨"}

# بطاقات الأيقونات — id, عربي, إنجليزي, emoji, color, bg, ar_desc, en_desc, kb_key
CARDS=[
    ("ic-ai",    "الذكاء الاصطناعي","AI Algorithms","🤖",C["purple"],C["purple_bg"],
     "6 خوارزميات: RF·SVM·LOF·OCSVM·ISO·KMeans",
     "6 algorithms: RF·SVM·LOF·OCSVM·ISO·KMeans","ai"),
    ("ic-alert", "الإنذار الفوري",  "Smart Alert",  "🚨",C["red"],   C["red_bg"],
     "تنبيه صوتي فوري عند تجاوز أي حد خطر",
     "Instant sound alert on any threshold breach","co"),
    ("ic-env",   "جودة الهواء",     "Air Quality",  "🌿",C["green"], C["green_bg"],
     "مراقبة مؤشر AQI في الوقت الفعلي",
     "Real-time AQI monitoring","aqi"),
    ("ic-temp",  "الحرارة والرطوبة","Temp & RH",    "🌡️",C["warn"],  C["warn_bg"],
     "DHT22: 20–24°C مثالي · 40–60% رطوبة",
     "DHT22: Ideal 20-24°C · Humidity 40-60%","temp"),
    ("ic-chart", "تحليل البيانات",  "Data Analysis","📊",C["blue"],  C["blue_bg"],
     "رسوم بيانية + خوارزميات تصنيف وتنبؤ",
     "Charts + classification & prediction algorithms","risk"),
    ("ic-sensor","📡 سينرات","Sensors",     "📡",C["green"], C["green_bg"],
     "DHT22 درجة الحرارة · MQ-135 الهواء · MQ-2 الدخان · MQ-7 أول أكسيد الكربون",
     "DHT22 Temp · MQ-135 Air · MQ-2 Smoke · MQ-7 CO · Arduino","sensors"),
]

# شرح الخوارزميات
ALGO_INFO = [
    ("🌲","Random Forest","تصنيف","يبني 200 شجرة قرار مستقلة ويأخذ تصويت الأغلبية لتصنيف مستوى الخطر. الأكثر دقةً — 99%.","Builds 200 independent trees and votes. Most accurate — 99%.","#e8f5e9","#43a047"),
    ("⚡","SVM","تصنيف","يجد الحد الفاصل الأمثل بين مستويات الخطر في فضاء متعدد الأبعاد. دقة 99%.","Finds optimal boundary between risk levels in high-dimensional space. 99% accuracy.","#e3f2fd","#1565c0"),
    ("📡","LOF","كشف قراءات غير طبيعية","يكتشف القراءات الغير طبيعية بمقارنة كثافة كل نقطة بجيرانها. لا يحتاج تسميات مسبقة.","Detects abnormal readings by comparing density to neighbors. No labels needed.","#fff3e0","#ff9800"),
    ("🎯","One-Class SVM","كشف قراءات غير طبيعية","يتدرب على البيانات الطبيعية فقط ويرسم حدوداً حولها. أي قراءة خارجها = قراءة غير طبيعية.","Trains on normal data and draws boundaries. Any reading outside = abnormal reading.","#fce4ec","#c2185b"),
    ("🌳","Isolation Forest","كشف قراءات غير طبيعية","يعزل القراءات الغير طبيعية بأشجار عشوائية — الغير طبيعي يُعزل بخطوات أقل لأنه بعيد.","Isolates abnormal readings with random trees — abnormals need fewer steps.","#f9e8ff","#7b1fa2"),
    ("🗂️","K-Means","تجميع","يجمع القراءات المتشابهة في 5 مجموعات لاكتشاف الأنماط الخفية كذروة التلوث الصباحية.","Groups readings into 5 clusters to find hidden patterns like morning pollution peaks.","#e0f2f1","#00695c"),
]

QUICK_Q=[
    ("🌫️ AQI","ما هي مستويات AQI الآمنة؟","What are safe AQI levels?"),
    ("☠️ CO","ما خطر أول أكسيد الكربون؟","What is CO danger?"),
    ("🔥 دخان","مستويات الدخان الخطرة","Dangerous smoke levels"),
    ("🌡️ حرارة","درجة الحرارة المثلى","Ideal temperature"),
    ("📡 سينرات","ما السينرات (Sensors) المستخدمة؟","What sensors are used?"),
    ("🤖 خوارزميات","شرح خوارزميات الذكاء الاصطناعي","Explain AI algorithms"),
]

def badge(lv):
    return html.Span([LICON.get(lv,"")," ",lv],className=f"badge-base {LBADGE.get(lv,'badge-safe')}")

def stat_card(label,val,color,icon=""):
    return html.Div(className="stat-card",style={
        "background":C[f"{color}_bg"],"border":f"2px solid {C[color]}"},children=[
        html.Div(f"{icon} {label}",style={"fontSize":"10px","fontWeight":"700","opacity":".6","marginBottom":"3px"}),
        html.Div(str(val),style={"fontSize":"20px","fontWeight":"800","color":C[color]}),
    ])

def wave_bars(color):
    return html.Span([html.Span(className="wave-bar",style={"background":color}) for _ in range(5)])

# ══ طالب جامعي ASCII ══
STUDENT = html.Div(style={
    "animation":"studentBob 3s ease-in-out infinite",
    "display":"inline-block","textAlign":"center"
}, children=[
    html.Div(style={"position":"relative","width":"82px","height":"140px","margin":"0 auto"},children=[

        # شعر أسود
        html.Div(style={"position":"absolute","top":"0","left":"6px","right":"6px",
            "height":"38px","background":"linear-gradient(160deg,#1a0800 55%,#2d1200)",
            "borderRadius":"32px 32px 4px 4px","zIndex":"4",
            "boxShadow":"2px 3px 8px rgba(0,0,0,.5)"}),
        # خصلة
        html.Div(style={"position":"absolute","top":"5px","left":"11px",
            "width":"17px","height":"24px","background":"#1a0800",
            "borderRadius":"50% 30% 40% 50%","zIndex":"5","transform":"rotate(-8deg)"}),

        # وجه
        html.Div(style={"position":"absolute","top":"22px","left":"9px","right":"9px",
            "height":"48px","background":"linear-gradient(180deg,#d4956a,#c07840)",
            "borderRadius":"50%","zIndex":"3","boxShadow":"0 4px 12px rgba(0,0,0,.2)"}),

        # حاجبان
        html.Div(style={"position":"absolute","top":"32px","left":"17px","width":"12px",
            "height":"2.5px","background":"#1a0800","borderRadius":"3px","zIndex":"6","transform":"rotate(-4deg)"}),
        html.Div(style={"position":"absolute","top":"32px","right":"17px","width":"12px",
            "height":"2.5px","background":"#1a0800","borderRadius":"3px","zIndex":"6","transform":"rotate(4deg)"}),

        # عيون
        html.Div(style={"position":"absolute","top":"38px","left":"17px","width":"12px","height":"10px",
            "background":"#1a0600","borderRadius":"50%","zIndex":"6",
            "animation":"eyeBlink 4s ease-in-out infinite"},
            children=[html.Div(style={"position":"absolute","top":"1px","right":"2px",
                "width":"3px","height":"3px","background":"white","borderRadius":"50%"})]),
        html.Div(style={"position":"absolute","top":"38px","right":"17px","width":"12px","height":"10px",
            "background":"#1a0600","borderRadius":"50%","zIndex":"6",
            "animation":"eyeBlink 4s ease-in-out infinite"},
            children=[html.Div(style={"position":"absolute","top":"1px","right":"2px",
                "width":"3px","height":"3px","background":"white","borderRadius":"50%"})]),

        # أنف
        html.Div(style={"position":"absolute","top":"48px","left":"50%","transform":"translateX(-50%)",
            "width":"8px","height":"7px","background":"#a86030",
            "borderRadius":"2px 2px 50% 50%","zIndex":"6"}),

        # ابتسامة
        html.Div(style={"position":"absolute","top":"57px","left":"50%","transform":"translateX(-50%)",
            "width":"22px","height":"9px","border":"2.5px solid #7a3810",
            "borderTop":"none","borderRadius":"0 0 14px 14px","zIndex":"6"}),

        # رقبة متصلة بالجسم
        html.Div(style={"position":"absolute","top":"66px","left":"50%","transform":"translateX(-50%)",
            "width":"20px","height":"8px","background":"#c07840","zIndex":"2"}),

        # ذراع أيسر
        html.Div(style={"position":"absolute","top":"76px","left":"0px","width":"13px","height":"36px",
            "background":"linear-gradient(180deg,#43a047,#2e7d32)",
            "borderRadius":"7px","zIndex":"3"}),
        # يد أيسر
        html.Div(style={"position":"absolute","top":"108px","left":"1px","width":"11px","height":"10px",
            "background":"#c07840","borderRadius":"50%","zIndex":"3"}),

        # ذراع أيمن
        html.Div(style={"position":"absolute","top":"76px","right":"0px","width":"13px","height":"36px",
            "background":"linear-gradient(180deg,#43a047,#2e7d32)",
            "borderRadius":"7px","zIndex":"3"}),
        # يد أيمن
        html.Div(style={"position":"absolute","top":"108px","right":"1px","width":"11px","height":"10px",
            "background":"#c07840","borderRadius":"50%","zIndex":"3"}),

        # جسم
        html.Div(style={"position":"absolute","top":"72px","left":"13px","right":"13px","height":"54px",
            "background":"linear-gradient(160deg,#43a047,#2e7d32)",
            "borderRadius":"0 0 10px 10px",
            "boxShadow":"0 6px 16px rgba(46,125,50,.4)","zIndex":"2"},children=[
            # ياقة
            html.Div(style={"position":"absolute","top":"0","left":"50%","transform":"translateX(-50%)",
                "width":"18px","height":"10px","background":"white",
                "clipPath":"polygon(0 0,100% 0,65% 100%,35% 100%)"}),
            # لابتوب
            html.Div(style={"position":"absolute","bottom":"5px","left":"50%",
                "transform":"translateX(-50%)","width":"32px","height":"20px",
                "background":"#1565c0","borderRadius":"3px","border":"2px solid #0d47a1",
                "boxShadow":"0 3px 8px rgba(0,0,0,.3)"},children=[
                html.Div(style={"width":"28px","height":"14px","background":"#bbdefb",
                    "borderRadius":"2px","margin":"1px auto",
                    "fontSize":"7px","textAlign":"center","color":"#1565c0",
                    "fontWeight":"800","lineHeight":"14px"},children=["AI 🤖"]),
            ]),
        ]),
    ]),
])

app.layout = html.Div(
  style={"fontFamily":"Cairo,sans-serif","background":"#f0f4f8","direction":"rtl","minHeight":"100vh"},
  children=[

    # ── رأس ──
    html.Div(style={"background":"#e8f5e9","borderBottom":"3px solid #43a047",
        "padding":"12px 20px","display":"flex","alignItems":"center",
        "justifyContent":"space-between","flexWrap":"wrap","gap":"8px"},children=[
        html.Div([
            html.Span(["EnviroAI ",html.Span("Chatbot",style={"color":C["blue"],"fontSize":"13px"})],
                style={"fontSize":"22px","fontWeight":"800","color":"#c62828"}),
            html.Span(" ",className="live-dot",style={"margin":"0 7px"}),
            html.Div("🌿 نظام المراقبة البيئية الذكي — Smart Environmental Monitoring",
                style={"fontSize":"11px","color":"#888","marginTop":"2px"}),
        ]),
        html.Div([wave_bars(C["green"]),
            html.Span(" ",style={"width":"10px","display":"inline-block"}),
            html.Div(id="clock",style={"fontSize":"20px","fontWeight":"800","color":"#1b5e20",
                "fontFamily":"monospace","letterSpacing":"2px","display":"inline"})],
            style={"display":"flex","alignItems":"center","gap":"6px"}),
    ]),

    # ── شريط إخباري متحرك ──
    html.Div(style={"background":"linear-gradient(90deg,#43a047,#66bb6a,#43a047)","padding":"8px 0","overflow":"hidden","position":"relative"},children=[
        html.Div(style={"display":"inline-block","whiteSpace":"nowrap","animation":"marquee 30s linear infinite",
            "color":"white","fontSize":"13px","fontWeight":"600","paddingRight":"100%"},
            children=[
                "🌿 نظام EnviroAI Pro للمراقبة البيئية  |  "
                "✅ AQI آمن: 0–50 (Safe)  |  "
                "🟡 AQI متوسط: 51–100 (Moderate)  |  "
                "🟠 AQI تحذير: 101–150 (Warning)  |  "
                "🔴 AQI حرج: 151–200 (Critical)  |  "
                "🚨 AQI طوارئ: +200 (Emergency)  |  "
                "☠️ CO آمن: أقل من 9 ppm (WHO)  |  "
                "🔥 Smoke آمن: أقل من 50 ppm  |  "
                "🌡️ حرارة مثلى: 20–24°C  |  "
                "💧 رطوبة مثلى: 40–60% RH  |  "
                "📡 المستشعرات: DHT22 · MQ-135 · MQ-2 · MQ-7 · Arduino UNO  |  "
                "🤖 خوارزميات: Random Forest · SVM · LOF · K-Means · Isolation Forest  |  "
                "📐 RiskIndex = 0.40×AQI + 0.30×CO + 0.30×SMOKE  |  "
                "🎓 الجامعة المستنصرية — قسم علوم الحاسوب 2025-2026  "
            ]),
    ]),

    # ── Chatbot ──
    html.Div(className="card card-3d",style={"margin":"14px"},children=[

        # ── صف الترحيب: طالبة + معلومات + لغة ──
        html.Div(style={"display":"flex","alignItems":"center","gap":"16px",
            "marginBottom":"14px","paddingBottom":"12px","borderBottom":f"1.5px solid #e8f0fe",
            "flexWrap":"wrap"},children=[

            # الطالبة المتحركة
            html.Div(style={"display":"flex","flexDirection":"column","alignItems":"center","gap":"4px","flexShrink":"0"},children=[
                STUDENT,
                html.Div("EnviroBot",style={"fontSize":"14px","fontWeight":"800","color":C["blue"]}),
                
            ]),

            # شرح البوت
            html.Div(style={"flex":"1","minWidth":"200px"},children=[
                html.Div("🤖 عن EnviroBot",style={"fontSize":"13px","fontWeight":"800","color":C["green"],"marginBottom":"6px"}),
                html.Div(style={"background":C["green_bg"],"borderRadius":"10px","padding":"10px 12px","fontSize":"12px","color":"#1a2a3a","lineHeight":"1.8","borderRight":f"4px solid {C['green']}"},children=[
                    html.Div("أنا مساعدك الذكي لنظام مراقبة جودة الهواء البيئي.",style={"fontWeight":"700","marginBottom":"4px"}),
                    html.Div("يمكنني الإجابة عن:"),
                    html.Div("• مستويات AQI الآمنة ومعايير WHO"),
                    html.Div("• خطر CO والدخان على الصحة"),
                    html.Div("• المستشعرات: DHT22 · MQ-135 · MQ-2 · MQ-7"),
                    html.Div("• خوارزميات الذكاء الاصطناعي ودور كل منها"),
                    html.Div("• معادلة مؤشر الخطر RiskIndex"),
                                    ]),
            ]),

            # مفتاح اللغة + بطاقات
            html.Div(style={"flexShrink":"0"},children=[
                html.Div(style={"display":"flex","gap":"5px","marginBottom":"8px","justifyContent":"center"},children=[
                    html.Button("🇸🇦 عربي",id="btn-ar",n_clicks=0,className="lang-btn",
                        style={"borderColor":C["green"],"background":C["green"],"color":"white"}),
                    html.Button("🇬🇧 EN",id="btn-en",n_clicks=0,className="lang-btn",
                        style={"borderColor":C["green"],"background":"white","color":C["green_dk"]}),
                ]),
                html.Div(id="lang-status",style={"fontSize":"10px","color":"#888","textAlign":"center","marginBottom":"6px"}),
                # بطاقات الأيقونات
                html.Div(style={"display":"grid","gridTemplateColumns":"repeat(3,52px)","gap":"5px"},
                    children=[
                        html.Div(id=cid,className="icon-card",
                            style={"background":bg,"border":"3px solid #43a047","padding":"10px 4px"},
                            children=[
                                html.Div(emj,style={"fontSize":"30px","lineHeight":"1","animation":"float 3s ease-in-out infinite","display":"block","textAlign":"center"}),
                                html.Div(ar,style={"color":"#388e3c","fontSize":"10px","fontWeight":"800","textAlign":"center","marginTop":"5px","wordBreak":"break-word","lineHeight":"1.3"}),
                            ])
                        for cid,ar,_,emj,col,bg,*_ in CARDS
                    ]),
                html.Div(id="icon-panel",style={"marginTop":"6px","maxWidth":"170px"}),
            ]),
        ]),

        # ── نافذة الدردشة ──
        html.Div("💬 تحدّث مع EnviroBot",style={"fontSize":"13px","fontWeight":"800","color":C["green"],"marginBottom":"8px"}),
        html.Div(id="chat-win",className="chat-window",
            style={"background":"#f8faff","borderRadius":"10px","border":f"2px solid {C['green']}","minHeight":"300px"},
            children=[html.Div(className="msg-bot",children=[
                html.Div("🤖 EnviroBot",className="msg-label"),
                html.Div("مرحباً! 👋\n\nاكتب سؤالك في الأسفل أو اختر من الأسئلة السريعة.\nاضغط 🔊 بجانب أي رسالة للاستماع بالصوت.",className="bubble"),
            ])]),

        html.Div(style={"display":"flex","gap":"6px","marginTop":"8px"},children=[
            dcc.Input(id="chat-in",type="text",
                placeholder="اكتب سؤالك هنا...",
                style={"flex":"1","padding":"9px 12px","borderRadius":"8px",
                       "border":f"1.5px solid {C['green']}","fontSize":"12px",
                       "fontFamily":"Cairo,sans-serif","outline":"none"},
                debounce=False,n_submit=0),
            html.Button("📤",id="send",n_clicks=0,
                style={"background":C["green"],"color":"white","border":"none","borderRadius":"8px",
                       "padding":"9px 14px","fontSize":"15px","cursor":"pointer","fontWeight":"800"}),
        ]),

        html.Div(style={"marginTop":"7px"},children=[
            html.Div("⚡ أسئلة سريعة:",style={"fontSize":"10px","fontWeight":"700","color":"#999","marginBottom":"4px"}),
            html.Div(style={"display":"flex","gap":"4px","flexWrap":"wrap"},children=[
                html.Button(lbl,id=f"qq-{i}",n_clicks=0,
                    style={"background":C["green_bg"],"color":C["green_dk"],"border":f"1px solid {C['green']}",
                           "borderRadius":"12px","padding":"3px 9px","fontSize":"10px",
                           "fontWeight":"700","cursor":"pointer","fontFamily":"Cairo,sans-serif"})
                for i,(lbl,_,_) in enumerate(QUICK_Q)
            ]),
        ]),
    ]),

    # ── شرح الخوارزميات ──
    html.Div(className="card card-3d",children=[
        html.Div(style={"display":"flex","alignItems":"center","justifyContent":"space-between",
            "marginBottom":"12px","borderBottom":f"2px solid {C['purple']}","paddingBottom":"6px"},children=[
            html.Div("🧠 خوارزميات الذكاء الاصطناعي — دور كل منها في النظام",
                style={"fontSize":"14px","fontWeight":"800","color":C["purple"]}),
            html.Div(style={"display":"flex","flexDirection":"column","gap":"4px"},children=[
                html.Div(style={"textAlign":"center"},children=[
                    dcc.Upload(id="upload",accept=".xlsx",
                        children=html.Button("📂 رفع ملف Excel",
                            style={"background":"#43a047","color":"white","border":"none","borderRadius":"10px",
                                   "padding":"9px 20px","fontSize":"13px","fontWeight":"800",
                                   "cursor":"pointer","fontFamily":"Cairo,sans-serif",
                                   "boxShadow":"0 3px 10px rgba(46,125,50,.3)"})),
                    html.Div("هل تريد تجربة تطبيق الخوارزميات على بياناتك البيئية الحقيقية؟ ارفع ملف Excel يحتوي على قراءات AQI · CO · SMOKE · Temperature · Humidity وسيقوم النظام بتصنيف كل قراءة وكشف القراءات غير طبيعية وعرض النتائج فوراً!",
                        style={"fontSize":"11px","color":"#666","marginTop":"6px"}),
                ]),
            ]),
        ]),
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(260px,1fr))","gap":"10px"},
            children=[
                html.Div(className="algo-card-3d",style={"background":bg,"border":f"2px solid {color}"},children=[
                    html.Div(style={"display":"flex","alignItems":"center","gap":"8px","marginBottom":"6px"},children=[
                        html.Span(emj,style={"fontSize":"22px"}),
                        html.Div([
                            html.Div(name,style={"fontSize":"13px","fontWeight":"800","color":color}),
                            html.Span(role,style={"fontSize":"10px","background":color,"color":"white",
                                "borderRadius":"10px","padding":"1px 8px","fontWeight":"700"}),
                        ]),
                    ]),
                    html.P(ar_desc,style={"fontSize":"11.5px","color":"#444","lineHeight":"1.75","marginBottom":"6px"}),
                    html.P(en_desc,style={"fontSize":"10.5px","color":"#777","lineHeight":"1.65","fontStyle":"italic"}),
                ])
                for emj,name,role,ar_desc,en_desc,bg,color in ALGO_INFO
            ]),
    ]),

    html.Div(id="status-bar",style={"padding":"0 14px 4px"}),

    html.Div(id="alert-zone"),
    html.Div(id="badges-zone",style={"padding":"0 14px 4px"}),
    html.Div(id="dl-zone"),
    html.Div(id="stats-zone"),
    html.Div(id="charts-zone"),
    html.Div(id="sim-zone"),

    dcc.Interval(id="tick",interval=1000,n_intervals=0),
    dcc.Store(id="store"),
    dcc.Store(id="chat-data",data=[]),
    dcc.Store(id="lang",data="ar"),
    dcc.Download(id="dl-excel"),
    dcc.Download(id="dl-csv"),
  ]
)

# ── ساعة ──
@app.callback(Output("clock","children"),Input("tick","n_intervals"))
def clock(_):
    n=datetime.datetime.now(); h=n.hour%12 or 12
    return f"{h:02d}:{n.minute:02d}:{n.second:02d} {'AM' if n.hour<12 else 'PM'}"

# ── اللغة ──
@app.callback(
    Output("lang","data"),Output("btn-ar","style"),Output("btn-en","style"),Output("lang-status","children"),
    Input("btn-ar","n_clicks"),Input("btn-en","n_clicks"),prevent_initial_call=True,
)
def set_lang(*_):
    t=ctx.triggered_id
    if t=="btn-en":
        return("en",{"borderColor":C["blue"],"background":"white","color":C["blue"]},
               {"borderColor":C["blue"],"background":C["blue"],"color":"white"},"Replies: English 🇬🇧")
    return("ar",{"borderColor":C["blue"],"background":C["blue"],"color":"white"},
           {"borderColor":C["blue"],"background":"white","color":C["blue"]},"الردود: عربي 🇸🇦")

# ── بطاقات الأيقونات ──
@app.callback(
    Output("icon-panel","children"),
    [Input(cid,"n_clicks") for cid,*_ in CARDS],
    State("lang","data"),prevent_initial_call=True,
)
def show_icon(*args):
    lang=args[-1]; t=ctx.triggered_id
    card=next((c for c in CARDS if c[0]==t),None)
    if not card: raise dash.exceptions.PreventUpdate
    cid,ar,en,emj,color,bg,ar_desc,en_desc,kb_key=card
    label=ar if lang=="ar" else en
    desc=ar_desc if lang=="ar" else en_desc
    kb_reply=chatbot_reply(kb_key,lang)
    speak_text=kb_reply.replace("•","").replace("🌡️","").replace("💨","").replace("🔥","").replace("☠️","").replace("🔌","")[:180]
    
    # أشكال السينرات لبطاقة ic-sensor
    sensor_shapes = html.Div()
    if cid == "ic-sensor":
        sensor_shapes = html.Div(style={"display":"flex","gap":"6px","justifyContent":"center","marginTop":"8px","flexWrap":"wrap"},children=[
            html.Div(style={"textAlign":"center"},children=[
                html.Div(style={"width":"16px","height":"26px","background":"#e53935","borderRadius":"3px 3px 5px 5px","margin":"0 auto","border":"2px solid #b71c1c","display":"flex","alignItems":"center","justifyContent":"center"},
                    children=[html.Div("T",style={"color":"white","fontSize":"8px","fontWeight":"800"})]),
                html.Div("DHT22",style={"fontSize":"8px","color":"#666","marginTop":"2px","fontWeight":"700"}),
            ]),
            html.Div(style={"textAlign":"center"},children=[
                html.Div(style={"width":"20px","height":"20px","borderRadius":"50%","background":"#ff9800","margin":"0 auto","border":"2px solid #e65100","display":"flex","alignItems":"center","justifyContent":"center"},
                    children=[html.Div("Q",style={"color":"white","fontSize":"8px","fontWeight":"800"})]),
                html.Div("MQ-135",style={"fontSize":"8px","color":"#666","marginTop":"2px","fontWeight":"700"}),
            ]),
            html.Div(style={"textAlign":"center"},children=[
                html.Div(style={"width":"20px","height":"20px","borderRadius":"50%","background":"#1565c0","margin":"0 auto","border":"2px solid #0d47a1","display":"flex","alignItems":"center","justifyContent":"center"},
                    children=[html.Div("Q",style={"color":"white","fontSize":"8px","fontWeight":"800"})]),
                html.Div("MQ-2",style={"fontSize":"8px","color":"#666","marginTop":"2px","fontWeight":"700"}),
            ]),
            html.Div(style={"textAlign":"center"},children=[
                html.Div(style={"width":"20px","height":"20px","borderRadius":"50%","background":"#c62828","margin":"0 auto","border":"2px solid #b71c1c","display":"flex","alignItems":"center","justifyContent":"center"},
                    children=[html.Div("Q",style={"color":"white","fontSize":"8px","fontWeight":"800"})]),
                html.Div("MQ-7",style={"fontSize":"8px","color":"#666","marginTop":"2px","fontWeight":"700"}),
            ]),
            html.Div(style={"textAlign":"center"},children=[
                html.Div(style={"width":"28px","height":"16px","background":"#00979d","borderRadius":"2px","margin":"0 auto","border":"2px solid #006e74","display":"flex","alignItems":"center","justifyContent":"center"},
                    children=[html.Div("UNO",style={"color":"white","fontSize":"7px","fontWeight":"800"})]),
                html.Div("Arduino",style={"fontSize":"8px","color":"#666","marginTop":"2px","fontWeight":"700"}),
            ]),
        ])
    
    return html.Div(style={"background":bg,"border":f"2px solid {color}","borderRadius":"10px","padding":"10px"},children=[
        html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"center","marginBottom":"5px"},children=[
            html.Div([html.Span(emj,style={"fontSize":"16px","marginLeft":"5px"}),
                html.Span(label,style={"fontSize":"12px","fontWeight":"800","color":color})]),
            html.Button("🔊",style={"background":C["green"],"color":"white","border":"none","borderRadius":"6px",
                "padding":"3px 10px","fontSize":"11px","cursor":"pointer","fontFamily":"Cairo,sans-serif"},
                **{"data-speak-text":speak_text,"data-speak-lang":"en"}),
        ]),
        html.Div(desc,style={"fontSize":"11px","color":"#555","lineHeight":"1.7"}),
        sensor_shapes,
    ])

# ── Chatbot ──
@app.callback(
    Output("chat-win","children"),Output("chat-data","data"),Output("chat-in","value"),
    Input("send","n_clicks"),Input("chat-in","n_submit"),
    *[Input(f"qq-{i}","n_clicks") for i in range(len(QUICK_Q))],
    State("chat-in","value"),State("chat-data","data"),State("lang","data"),
    prevent_initial_call=True,
)
def chat(*args):
    nqq=len(QUICK_Q)
    user_text=args[2+nqq]; history=args[3+nqq]; lang=args[4+nqq]
    t=ctx.triggered_id; question=""
    if t in ("send","chat-in"):
        question=user_text or ""
    elif t and t.startswith("qq-"):
        idx=int(t.split("-")[1])
        question=QUICK_Q[idx][2] if lang=="en" else QUICK_Q[idx][1]
    if not question.strip(): raise dash.exceptions.PreventUpdate

    reply=chatbot_reply(question,lang)
    speak=reply.replace("•","").replace("🌡️","").replace("💨","").replace("🔥","").replace("☠️","").replace("🔌","")[:200]
    msgs=list(history or [])
    msgs.append({"role":"user","text":question})
    msgs.append({"role":"bot","text":reply,"speak":speak,"lang":lang})

    bubbles=[html.Div(className="msg-bot",children=[
        html.Div("🤖 EnviroBot",className="msg-label"),
        html.Div("مرحباً! 👋 أنا EnviroBot — المساعد الذكي لنظام مراقبة البيئة.\n\nاسألني عن مستويات AQI، خطر CO، المستشعرات، أو الخوارزميات.\n\nاضغط 🔊 بجانب أي رسالة للاستماع 🎤",className="bubble"),
        html.Button("🔊 استمع",className="speak-btn",
            **{"data-speak-text":'مرحباً أنا EnviroBot المساعد الذكي لنظام مراقبة البيئة',"data-speak-lang":"ar"}),
    ])]
    for msg in msgs:
        if msg["role"]=="user":
            bubbles.append(html.Div(className="msg-user",children=[
                html.Div("أنت 👤",className="msg-label",style={"textAlign":"left"}),
                html.Div(msg["text"],className="bubble")]))
        else:
            sp=msg.get("speak",""); lg=msg.get("lang","ar")
            btn_lbl="🔊 Listen" if lg=="en" else "🔊 استمع"
            bubbles.append(html.Div(className="msg-bot",children=[
                html.Div("🤖 EnviroBot",className="msg-label"),
                html.Div(msg["text"],className="bubble"),
                html.Button("🔊",style={"background":C["green"],"color":"white","border":"none","borderRadius":"6px",
                    "padding":"3px 10px","fontSize":"11px","cursor":"pointer"},
                    **{"data-speak-text":sp,"data-speak-lang":"en"}),
            ]))
    return bubbles,msgs,""

# ── رفع ──
@app.callback(
    Output("status-bar","children"),Output("alert-zone","children"),
    Output("badges-zone","children"),Output("dl-zone","children"),
    Output("stats-zone","children"),Output("charts-zone","children"),
    Output("sim-zone","children"),Output("store","data"),
    Input("upload","contents"),State("upload","filename"),prevent_initial_call=True,
)
def on_upload(contents,filename):
    _,cs=contents.split(",")
    df=pd.read_excel(io.BytesIO(base64.b64decode(cs)))
    df=process_data(df); ms=train_all(df); st=get_stats(df)

    status=html.Div(style={"background":C["green_bg"],"border":f"2px solid {C['green']}",
        "borderRadius":"8px","padding":"10px 14px","textAlign":"center","fontWeight":"700","fontSize":"12px","color":C["green_dk"]},
        children=[f"✅ تم تحليل {len(df):,} قراءة  |  RF: {ms['RF']['acc']*100:.1f}%  |  SVM: {ms['SVM']['acc']*100:.1f}%  |  قراءات غير طبيعية: {ms['LOF']['anom']}"])

    alert_z=html.Div()
    if st["emergency"]>0:
        alert_z=html.Div(className="alert-anim card",style={
            "background":C["red_bg"],"border":f"3px solid {C['red']}","borderRadius":"12px","margin":"0 14px 8px","padding":"16px"},children=[
            html.Span(**{"data-level":"طوارئ"},style={"display":"none"}),
            html.Div(style={"display":"flex","alignItems":"center","gap":"10px","marginBottom":"8px"},children=[
                html.Div([html.Div(className="ping-core"),html.Div(className="ping-ring")],className="ping-wrap"),
                html.Div("🚨 إنذار طوارئ — قراءات خطرة مكتشفة",style={"fontSize":"16px","fontWeight":"800","color":C["red_dk"]}),
            ]),
            html.Div(f"طوارئ: {st['emergency']}  |  أعلى AQI: {st['max_aqi']}  |  أعلى SMOKE: {st['max_smoke']} ppm",
                style={"fontSize":"12px","color":C["red"],"fontWeight":"600"}),
        ])

    badges=html.Div(className="card card-3d",children=[
        html.Div("🏅 مستويات الخطر المكتشفة",style={"fontSize":"11px","fontWeight":"800","color":"#666","marginBottom":"8px"}),
        html.Div(style={"display":"flex","gap":"8px","flexWrap":"wrap","alignItems":"center"},children=[
            html.Div([badge(lv),html.Span(str(st[k]),style={"fontWeight":"800","color":LCOLOR.get(lv,"#333"),"fontSize":"14px","marginRight":"3px"})],
                style={"display":"flex","alignItems":"center","gap":"4px"})
            for lv,k in [("آمن","safe"),("متوسط","moderate"),("تحذير","warning"),("حرج","critical"),("طوارئ","emergency")]
        ]),
    ])

    dl_z=html.Div(className="card card-3d",children=[
        html.Div("📥 تنزيل التقارير",style={"fontSize":"12px","fontWeight":"800","color":C["green"],"marginBottom":"8px"}),
        html.Div(style={"display":"flex","gap":"8px","flexWrap":"wrap"},children=[
            html.Button("📊 Excel",id="btn-excel",n_clicks=0,
                style={"background":C["green"],"color":"white","border":"none","borderRadius":"8px",
                       "padding":"8px 16px","fontSize":"12px","fontWeight":"800","cursor":"pointer","fontFamily":"Cairo,sans-serif"}),
            html.Button("📋 CSV",id="btn-csv",n_clicks=0,
                style={"background":C["blue"],"color":"white","border":"none","borderRadius":"8px",
                       "padding":"8px 16px","fontSize":"12px","fontWeight":"800","cursor":"pointer","fontFamily":"Cairo,sans-serif"}),
            html.Button("🖨️ طباعة",id="btn-print",n_clicks=0,
                style={"background":"#546e7a","color":"white","border":"none","borderRadius":"8px",
                       "padding":"8px 16px","fontSize":"12px","fontWeight":"800","cursor":"pointer","fontFamily":"Cairo,sans-serif"},
                **{"data-action":"print"}),
        ]),
    ])

    stats_z=html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(100px,1fr))",
        "gap":"8px","padding":"0 14px 12px"},children=[
        stat_card("إجمالي",f"{st['total']:,}","blue","📋"),
        stat_card("متوسط AQI",st["avg_aqi"],"red","🌫️"),
        stat_card("متوسط CO",st["avg_co"],"red","💨"),
        stat_card("أعلى SMOKE",st["max_smoke"],"red","🔥"),
        stat_card("آمن",st["safe"],"green","✅"),
        stat_card("تحذير",st["warning"],"warn","⚠️"),
        stat_card("حرج+طوارئ",st["critical"]+st["emergency"],"red","🚨"),
        stat_card("دقة RF",f"{ms['RF']['acc']*100:.1f}%","green","🎯"),
    ])

    charts_z=html.Div(className="card card-3d",children=[
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(280px,1fr))","gap":"12px"},children=[
            html.Div([html.Div("توزيع مستويات الخطر",style={"fontSize":"12px","fontWeight":"700","color":C["red"],"marginBottom":"4px"}),
                dcc.Graph(figure=bar_fig(df),config={"displayModeBar":False})],
                style={"background":"#f8f9fa","borderRadius":"10px","padding":"12px","border":"1.5px solid #e0e0e0"}),
            html.Div([html.Div("مؤشرات التلوث عبر الزمن",style={"fontSize":"12px","fontWeight":"700","color":C["blue"],"marginBottom":"4px"}),
                dcc.Graph(figure=timeline_fig(df),config={"displayModeBar":False})],
                style={"background":"#f8f9fa","borderRadius":"10px","padding":"12px","border":"1.5px solid #e0e0e0"}),
        ]),
    ])

    sim_z=html.Div(className="card card-3d",children=[
        html.Div(style={"display":"flex","alignItems":"center","gap":"8px","marginBottom":"12px"},children=[
            html.Span("🤖",style={"fontSize":"24px","animation":"spin 3s linear infinite","display":"inline-block"}),
            html.Div("⚡ محاكاة قراءة جديدة",style={"fontSize":"13px","fontWeight":"800","color":C["blue"]}),
        ]),
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(140px,1fr))","gap":"8px","marginBottom":"10px"},children=[
            html.Div([html.Label("AQI",style={"fontSize":"11px","color":"#555","fontWeight":"600"}),
                dcc.Slider(0,462,1,value=94,id="s1",marks=None,tooltip={"always_visible":True})]),
            html.Div([html.Label("CO ppm",style={"fontSize":"11px","color":"#555","fontWeight":"600"}),
                dcc.Slider(0,751,1,value=88,id="s2",marks=None,tooltip={"always_visible":True})]),
            html.Div([html.Label("SMOKE ppm",style={"fontSize":"11px","color":"#555","fontWeight":"600"}),
                dcc.Slider(15,1017,1,value=100,id="s3",marks=None,tooltip={"always_visible":True})]),
            html.Div([html.Label("Temp °C",style={"fontSize":"11px","color":"#555","fontWeight":"600"}),
                dcc.Slider(20,38,.5,value=27,id="s4",marks=None,tooltip={"always_visible":True})]),
            html.Div([html.Label("Humidity %",style={"fontSize":"11px","color":"#555","fontWeight":"600"}),
                dcc.Slider(30,100,1,value=52,id="s5",marks=None,tooltip={"always_visible":True})]),
        ]),
        html.Button("⚡ تشغيل الخوارزميات + تنبيه صوتي",id="btn-sim",n_clicks=0,
            style={"width":"100%","padding":"10px","borderRadius":"8px","border":"none","background":C["blue"],
                   "color":"white","fontSize":"13px","fontWeight":"800","cursor":"pointer","fontFamily":"Cairo,sans-serif"}),
        html.Div(id="sim-out",style={"marginTop":"8px"}),
    ])

    # ── ملخص نصي على الواجهة ──
    total_emergency = st["emergency"] + st["critical"]
    if total_emergency > 0:
        summary_color = C["red"]; summary_bg = C["red_bg"]
        summary_icon = "🚨"
        summary_text_ar = f"تحذير: {total_emergency} قراءة خطرة من أصل {st['total']:,} — تحتاج تدخل فوري"
        summary_text_en = f"Alert: {total_emergency} critical readings out of {st['total']:,} — immediate action required"
    elif st["warning"] > 0:
        summary_color = C["warn"]; summary_bg = C["warn_bg"]
        summary_icon = "⚠️"
        summary_text_ar = f"تحذير: {st['warning']} قراءة تحتاج مراقبة من {st['total']:,} قراءة"
        summary_text_en = f"Warning: {st['warning']} readings need monitoring out of {st['total']:,}"
    else:
        summary_color = C["green"]; summary_bg = C["green_bg"]
        summary_icon = "✅"
        summary_text_ar = f"جيد: معظم القراءات {st['safe']} آمنة من {st['total']:,} قراءة"
        summary_text_en = f"Good: most readings {st['safe']} are safe out of {st['total']:,}"

    summary_card = html.Div(style={
        "background":summary_bg,"border":f"2px solid {summary_color}","borderRadius":"12px",
        "padding":"14px 18px","margin":"0 14px 10px","animation":"slideDown .5s ease"
    }, children=[
        html.Div(f"{summary_icon} ملخص نتائج التحليل",
            style={"fontSize":"14px","fontWeight":"800","color":summary_color,"marginBottom":"10px"}),
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(140px,1fr))","gap":"8px","marginBottom":"10px"},
            children=[
                html.Div(style={"background":"white","borderRadius":"8px","padding":"8px 12px","border":f"1px solid {summary_color}","textAlign":"center"},children=[
                    html.Div("إجمالي القراءات",style={"fontSize":"10px","color":"#888"}),
                    html.Div(f"{st['total']:,}",style={"fontSize":"20px","fontWeight":"800","color":"#1b5e20"}),
                ]),
                html.Div(style={"background":"white","borderRadius":"8px","padding":"8px 12px","border":f"1px solid #43a047","textAlign":"center"},children=[
                    html.Div("آمن (Safe)",style={"fontSize":"10px","color":"#888"}),
                    html.Div(str(st["safe"]),style={"fontSize":"20px","fontWeight":"800","color":C["green"]}),
                ]),
                html.Div(style={"background":"white","borderRadius":"8px","padding":"8px 12px","border":f"1px solid #ff9800","textAlign":"center"},children=[
                    html.Div("تحذير (Warning)",style={"fontSize":"10px","color":"#888"}),
                    html.Div(str(st["warning"]),style={"fontSize":"20px","fontWeight":"800","color":C["warn"]}),
                ]),
                html.Div(style={"background":"white","borderRadius":"8px","padding":"8px 12px","border":f"1px solid {C['red']}","textAlign":"center"},children=[
                    html.Div("حرج+طوارئ (Critical)",style={"fontSize":"10px","color":"#888"}),
                    html.Div(str(st["critical"]+st["emergency"]),style={"fontSize":"20px","fontWeight":"800","color":C["red"]}),
                ]),
                html.Div(style={"background":"white","borderRadius":"8px","padding":"8px 12px","border":f"1px solid #43a047","textAlign":"center"},children=[
                    html.Div("دقة النموذج RF",style={"fontSize":"10px","color":"#888"}),
                    html.Div(f"{ms['RF']['acc']*100:.1f}%",style={"fontSize":"20px","fontWeight":"800","color":C["green"]}),
                ]),
                html.Div(style={"background":"white","borderRadius":"8px","padding":"8px 12px","border":f"1px solid {C['blue']}","textAlign":"center"},children=[
                    html.Div("متوسط AQI",style={"fontSize":"10px","color":"#888"}),
                    html.Div(str(st["avg_aqi"]),style={"fontSize":"20px","fontWeight":"800","color":"#1b5e20"}),
                ]),
            ]),
        html.Div(summary_text_ar,style={"fontSize":"13px","fontWeight":"700","color":summary_color,"marginBottom":"4px"}),
        html.Div(summary_text_en,style={"fontSize":"11px","color":"#888","fontStyle":"italic"}),
    ])

    # ── ملخص النتائج ──
    if st["emergency"] > 0:
        sc=C["red"]; sb="#fff5f5"; si="🚨"
        sar=f"نتيجة التحليل: {st['emergency']} قراءة طوارئ تحتاج تدخلاً فورياً من {st['total']:,} قراءة"
        sen=f"Result: {st['emergency']} emergency readings need immediate action out of {st['total']:,}"
    elif st["critical"] > 0:
        sc=C["red"]; sb="#fff8f8"; si="🔴"
        sar=f"نتيجة التحليل: {st['critical']} قراءة حرجة تستوجب المراقبة من {st['total']:,}"
        sen=f"Result: {st['critical']} critical readings require monitoring out of {st['total']:,}"
    elif st["warning"] > 0:
        sc=C["warn"]; sb="#fffbf0"; si="⚠️"
        sar=f"نتيجة التحليل: {st['warning']} قراءة تحذيرية — يُنصح باحتياطات"
        sen=f"Result: {st['warning']} warning readings — precautions advised"
    else:
        sc=C["green"]; sb="#f0faf0"; si="✅"
        sar=f"نتيجة التحليل: البيئة آمنة — {st['safe']} قراءة آمنة"
        sen=f"Result: Environment safe — {st['safe']} safe readings"

    summary_z = html.Div(id="summary-zone",style={
        "background":sb,"border":f"2px solid {sc}","borderRadius":"14px",
        "margin":"0 14px 14px","padding":"16px","animation":"slideDown .5s ease"
    }, children=[
        html.Div(f"{si} ملخص النتائج — Analysis Summary",
            style={"fontSize":"14px","fontWeight":"800","color":sc,
                   "marginBottom":"10px","borderBottom":f"1px solid {sc}","paddingBottom":"6px"}),
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(110px,1fr))","gap":"8px","marginBottom":"10px"},
            children=[
                html.Div(style={"background":"white","borderRadius":"10px","padding":"10px","border":f"2px solid {C['blue']}","textAlign":"center"},children=[
                    html.Div("📋 الإجمالي",style={"fontSize":"10px","color":"#888"}),
                    html.Div(f"{st['total']:,}",style={"fontSize":"20px","fontWeight":"800","color":"#1b5e20"}),
                ]),
                html.Div(style={"background":"white","borderRadius":"10px","padding":"10px","border":"2px solid #66bb6a","textAlign":"center"},children=[
                    html.Div("✅ آمن Safe",style={"fontSize":"10px","color":"#888"}),
                    html.Div(str(st["safe"]),style={"fontSize":"20px","fontWeight":"800","color":"#388e3c"}),
                    html.Div(f"{st['safe']*100//max(st['total'],1)}%",style={"fontSize":"10px","color":"#aaa"}),
                ]),
                html.Div(style={"background":"white","borderRadius":"10px","padding":"10px","border":"2px solid #ff9800","textAlign":"center"},children=[
                    html.Div("⚠️ تحذير",style={"fontSize":"10px","color":"#888"}),
                    html.Div(str(st["warning"]),style={"fontSize":"20px","fontWeight":"800","color":C["warn"]}),
                ]),
                html.Div(style={"background":"white","borderRadius":"10px","padding":"10px","border":f"2px solid {C['red']}","textAlign":"center"},children=[
                    html.Div("🔴 حرج+طوارئ",style={"fontSize":"10px","color":"#888"}),
                    html.Div(str(st["critical"]+st["emergency"]),style={"fontSize":"20px","fontWeight":"800","color":C["red"]}),
                ]),
                html.Div(style={"background":"white","borderRadius":"10px","padding":"10px","border":"2px solid #66bb6a","textAlign":"center"},children=[
                    html.Div("🎯 دقة RF",style={"fontSize":"10px","color":"#888"}),
                    html.Div(f"{ms['RF']['acc']*100:.0f}%",style={"fontSize":"20px","fontWeight":"800","color":"#388e3c"}),
                ]),
                html.Div(style={"background":"white","borderRadius":"10px","padding":"10px","border":f"2px solid {C['blue']}","textAlign":"center"},children=[
                    html.Div("🌫️ متوسط AQI",style={"fontSize":"10px","color":"#888"}),
                    html.Div(str(st["avg_aqi"]),style={"fontSize":"20px","fontWeight":"800","color":"#1b5e20"}),
                ]),
                html.Div(style={"background":"white","borderRadius":"10px","padding":"10px","border":"2px solid #78909c","textAlign":"center"},children=[
                    html.Div("📡 قراءات غير طبيعية",style={"fontSize":"10px","color":"#888"}),
                    html.Div(str(ms["LOF"]["anom"]),style={"fontSize":"20px","fontWeight":"800","color":"#546e7a"}),
                ]),
                html.Div(style={"background":"white","borderRadius":"10px","padding":"10px","border":f"2px solid {C['purple']}","textAlign":"center"},children=[
                    html.Div("🔥 أعلى SMOKE",style={"fontSize":"10px","color":"#888"}),
                    html.Div(str(st["max_smoke"]),style={"fontSize":"20px","fontWeight":"800","color":C["purple"]}),
                ]),
            ]),
        html.Div(sar,style={"fontSize":"13px","fontWeight":"700","color":sc,"marginBottom":"3px"}),
        html.Div(sen,style={"fontSize":"11px","color":"#888","fontStyle":"italic"}),
    ])

    return (status,alert_z,badges,dl_z,summary_z,charts_z,sim_z,
            df.to_json(date_format="iso",orient="split"))

# ── محاكاة ──
@app.callback(
    Output("sim-out","children"),
    Input("btn-sim","n_clicks"),
    State("s1","value"),State("s2","value"),State("s3","value"),
    State("s4","value"),State("s5","value"),
    State("store","data"),State("lang","data"),
    prevent_initial_call=True,
)
def run_sim(n,aqi,co,smoke,temp,hum,store,lang):
    if not store: return html.Div("ارفع ملف Excel أولاً",style={"color":C["red"]})
    df=process_data(pd.read_json(io.StringIO(store),orient="split")); ms=train_all(df)
    Xn=np.array([[aqi,co,smoke,temp,hum]]); Xs=ms["_sc"].transform(Xn)
    ri=round(0.4*aqi+0.3*co+0.3*smoke,1)
    lv="آمن" if ri<50 else "متوسط" if ri<100 else "تحذير" if ri<150 else "حرج" if ri<200 else "طوارئ"
    rf_p=ms["_le"].inverse_transform(ms["RF"]["model"].predict(Xs))[0]
    sv_p=ms["_le"].inverse_transform(ms["SVM"]["model"].predict(Xs))[0]
    lof_p="قراءة غير طبيعية" if ms["LOF"]["model"].predict(Xs)[0]==-1 else "طبيعي"
    col=LCOLOR.get(lv,C["red"]); bg=LBG.get(lv,C["red_bg"])
    sp=f"مستوى الخطر {lv} مؤشر الخطر {ri}" if lang=="ar" else f"Risk level {lv} index {ri}"
    return html.Div(style={"background":bg,"border":f"2px solid {col}","borderRadius":"10px","padding":"12px"},children=[
        html.Span(**{"data-level":lv},style={"display":"none"}),
        html.Div(style={"display":"flex","alignItems":"center","gap":"8px","marginBottom":"8px"},children=[
            badge(lv),
            html.Span(f"RiskIndex = {ri}",style={"fontSize":"14px","fontWeight":"800","color":col}),
            html.Button("🔊",className="speak-btn",**{"data-speak-text":sp,"data-speak-lang":lang}),
        ]),
        html.Div(style={"display":"grid","gridTemplateColumns":"repeat(auto-fit,minmax(120px,1fr))","gap":"6px"},children=[
            html.Div([html.Span("🌲 RF: ",style={"fontWeight":"700","fontSize":"11px"}),html.Span(rf_p)],
                style={"background":"white","border":f"1px solid {C['red']}","borderRadius":"6px","padding":"5px 9px","fontSize":"12px"}),
            html.Div([html.Span("⚡ SVM: ",style={"fontWeight":"700","fontSize":"11px"}),html.Span(sv_p)],
                style={"background":"white","border":f"1px solid {C['red']}","borderRadius":"6px","padding":"5px 9px","fontSize":"12px"}),
            html.Div([html.Span("📡 LOF: ",style={"fontWeight":"700","fontSize":"11px"}),html.Span(lof_p)],
                style={"background":"white","border":f"1px solid {C['blue']}","borderRadius":"6px","padding":"5px 9px","fontSize":"12px"}),
        ]),
    ])

@app.callback(Output("dl-excel","data"),Input("btn-excel","n_clicks"),State("store","data"),prevent_initial_call=True)
def dl_excel(n,store):
    if not store: return None
    df=process_data(pd.read_json(io.StringIO(store),orient="split")); ms=train_all(df); st=get_stats(df)
    return {"base64":True,"content":export_excel(df,st,ms),
            "filename":f"EnviroAI_{datetime.date.today()}.xlsx",
            "type":"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}

@app.callback(Output("dl-csv","data"),Input("btn-csv","n_clicks"),State("store","data"),prevent_initial_call=True)
def dl_csv(n,store):
    if not store: return None
    df=process_data(pd.read_json(io.StringIO(store),orient="split"))
    return {"base64":True,"content":base64.b64encode(df.to_csv(index=False,encoding="utf-8-sig").encode("utf-8-sig")).decode(),
            "filename":f"EnviroAI_{datetime.date.today()}.csv","type":"text/csv"}

if __name__=="__main__":
    port=int(os.environ.get("PORT",8050))
    print("\n"+"="*56)
    print("  EnviroAI Chatbot v3 — طالب جامعي + خوارزميات")
    print(f"  افتح: http://localhost:{port}")
    print("="*56+"\n")
    app.run(debug=False,host="0.0.0.0",port=port)
