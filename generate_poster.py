"""
EnviroAI Poster Generator -- Iraq
python generate_poster.py
"""

import os, sys, json, threading, subprocess, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

OUTPUT_FILE = "enviro_poster.html"
SERVER_PORT = int(os.environ.get("PORT", 5500))  # Railway injects PORT

PROGRAMS = [
    {"file":"enviroai_iraq", "name":"EnviroAI Portal", "name_ar":"\u0627\u0644\u0645\u0648\u0642\u0639 \u0627\u0644\u0631\u0633\u0645\u064a", "desc":"Railway \u2014 \u0627\u0644\u0645\u0648\u0642\u0639 \u0627\u0644\u0631\u0633\u0645\u064a \u0639\u0644\u0649 \u0627\u0644\u0625\u0646\u062a\u0631\u0646\u062a", "color":"#0d5c48","bg":"#d4f0e6","border":"#6fcfaf","icon":"ti-world"},
    {"file":"enviro_hub.py",      "name":"EnviroAI Hub",    "name_ar":"\u0627\u0644\u0645\u0648\u0642\u0639 \u0627\u0644\u062c\u0627\u0645\u0639",   "desc":"\u064a\u062c\u0645\u0639 \u0643\u0644 \u0627\u0644\u0628\u0631\u0627\u0645\u062c",     "color":"#0F6E56","bg":"#E1F5EE","border":"#9FE1CB","icon":"ti-layout-dashboard"},
    {"file":"enviro_ai_v3",       "name":"EnviroAI Pro v3", "name_ar":"\u0627\u0644\u062a\u0646\u0628\u0624 \u0628\u0627\u0644\u0645\u062e\u0627\u0637\u0631",  "desc":"\u0646\u0638\u0627\u0645 \u0645\u0631\u0627\u0642\u0628\u0629 \u0630\u0643\u064a \u2014 RF + SVM + LOF + KMeans",   "color":"#185FA5","bg":"#E6F1FB","border":"#B5D4F4","icon":"ti-chart-bar"},
    {"file":"enviro_chatbot",     "name":"EnviroAI Chatbot","name_ar":"\u0627\u0644\u0634\u0627\u062a \u0628\u0648\u062a",      "desc":"\u0634\u0631\u062d \u0627\u0644\u062e\u0648\u0627\u0631\u0632\u0645\u064a\u0627\u062a \u0648\u0627\u0644\u0645\u0633\u062a\u0634\u0639\u0631\u0627\u062a",  "color":"#7F77DD","bg":"#EEEDFE","border":"#AFA9EC","icon":"ti-message-chatbot"},
    {"file":"enviro_ai_full.py",  "name":"EnviroAI Full",   "name_ar":"XGBoost + SHAP",    "desc":"\u062e\u0648\u0627\u0631\u0632\u0645\u064a\u0629 \u0645\u062a\u0642\u062f\u0645\u0629 + \u062a\u0641\u0633\u064a\u0631 \u0627\u0644\u0642\u0631\u0627\u0631\u0627\u062a",  "color":"#854F0B","bg":"#FAEEDA","border":"#FAC775","icon":"ti-brain"},
    {"file":"enviro_xgboost",     "name":"EnviroAI XGBoost","name_ar":"\u0646\u0638\u0627\u0645 XGBoost \u0645\u062a\u0642\u062f\u0645", "desc":"Excel \u2190 \u062a\u062d\u0644\u064a\u0644 \u0641\u0648\u0631\u064a \u2190 \u0631\u0633\u0648\u0645 3D \u2190 XGBoost", "color":"#C0392B","bg":"#FDEDEC","border":"#F5B7B1","icon":"ti-chart-dots-3"},
    {"file":"smart_eco_monitor",  "name":"Smart Eco-Monitor","name_ar":"\u0627\u0644\u0645\u0631\u0627\u0642\u0628 \u0627\u0644\u0628\u064a\u0626\u064a \u0627\u0644\u0630\u0643\u064a","desc":"\u0645\u0631\u0627\u0642\u0628\u0629 \u062d\u064a\u0629 \u0645\u0646 \u0627\u0644\u0645\u0633\u062a\u0634\u0639\u0631\u0627\u062a \u0641\u064a \u0627\u0644\u0648\u0642\u062a \u0627\u0644\u0641\u0639\u0644\u064a","color":"#1A8C6F","bg":"#D8F5EC","border":"#6FD5B0","icon":"ti-wave-sine"},
]


def build_launcher_cards():
    cards = ""
    for p in PROGRAMS:
        if "." in p["file"] and not os.path.exists(p["file"]):
            continue
        f=p["file"]; n=p["name"]; na=p["name_ar"]; d=p["desc"]
        c=p["color"]; bg=p["bg"]; br=p["border"]; ic=p["icon"]
        cards += (
            f'<div class="app-card" style="background:{bg};border:1.5px solid {br};">'
            f'<div class="app-icon" style="background:{c};"><i class="ti {ic}"></i></div>'
            f'<div class="app-info">'
            f'<div class="app-name" style="color:{c};">{n}</div>'
            f'<div class="app-name-ar">{na}</div>'
            f'<div class="app-desc">{d}</div></div>'
            f'<button class="run-btn" style="background:{c};" onclick="runApp(\'{f}\',\'{n}\',this)">'
            f'<i class="ti ti-player-play"></i> \u062a\u0634\u063a\u064a\u0644</button></div>'
        )
    return cards

def build_html():
    launcher = build_launcher_cards()

    css = (
        "*{box-sizing:border-box;margin:0;padding:0;}"
        "body{font-family:'Cairo',sans-serif;direction:rtl;background:#e8f5f0;min-height:100vh;padding:58px 16px 24px;}"
        ".poster{max-width:960px;margin:0 auto;background:#fff;border-radius:20px;overflow:hidden;box-shadow:0 8px 40px rgba(15,110,86,0.13);}"
        ".hdr{background:linear-gradient(135deg,#0F6E56 0%,#1D9E75 45%,#22c995 100%);padding:52px 24px 22px;position:relative;overflow:hidden;}"
        ".hdr::before{content:'';position:absolute;top:-40px;left:-40px;width:200px;height:200px;border-radius:50%;background:rgba(255,255,255,0.06);}"
        ".hdr::after{content:'';position:absolute;bottom:-60px;right:-30px;width:240px;height:240px;border-radius:50%;background:rgba(255,255,255,0.05);}"
        ".hdr h1{font-size:16px;font-weight:700;color:white;line-height:1.6;text-align:center;position:relative;z-index:1;}"
        ".hdr h1 em{display:block;font-size:12px;color:rgba(255,255,255,0.72);font-style:normal;margin-top:3px;font-weight:400;}"
        ".hdr-desc{font-size:11px;color:rgba(255,255,255,0.8);line-height:1.85;max-width:640px;margin:9px auto 0;text-align:center;position:relative;z-index:1;}"
        ".pill-row{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-top:13px;position:relative;z-index:1;}"
        ".pill{border-radius:20px;padding:4px 16px;font-size:11px;font-weight:500;}"
        ".pill.p1{background:rgba(255,255,255,0.22);color:white;border:1px solid rgba(255,255,255,0.3);}"
        ".pill.p2{background:#fff;color:#0F6E56;font-weight:700;letter-spacing:1px;}"
        ".pill.p3{background:rgba(255,255,255,0.15);color:rgba(255,255,255,0.9);border:1px solid rgba(255,255,255,0.25);}"
        ".body{padding:18px;display:flex;flex-direction:column;gap:14px;background:#f7fdf9;}"
        ".sec{background:#fff;border:1px solid #d4ede5;border-radius:14px;padding:15px 17px;}"
        ".sec-title{font-size:12px;font-weight:600;color:#085041;border-bottom:1px solid #d4ede5;padding-bottom:8px;margin-bottom:13px;display:flex;align-items:center;gap:7px;}"
        ".sec-title i{font-size:16px;color:#1D9E75;}"
        ".apps-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;}"
        ".app-card{border-radius:12px;padding:12px 14px;display:flex;align-items:center;gap:11px;transition:transform .15s;}"
        ".app-card:hover{transform:translateY(-2px);}"
        ".app-icon{width:42px;height:42px;border-radius:10px;display:flex;align-items:center;justify-content:center;flex-shrink:0;}"
        ".app-icon i{font-size:20px;color:white;}"
        ".app-info{flex:1;min-width:0;}"
        ".app-name{font-size:12px;font-weight:700;}"
        ".app-name-ar{font-size:10px;color:#666;margin-top:1px;}"
        ".app-desc{font-size:10px;color:#888;margin-top:3px;line-height:1.4;}"
        ".run-btn{flex-shrink:0;border:none;border-radius:8px;padding:7px 13px;color:white;font-family:\'Cairo\',sans-serif;font-size:11px;font-weight:700;cursor:pointer;display:flex;align-items:center;gap:5px;transition:opacity .15s,transform .1s;white-space:nowrap;}"
        ".run-btn:hover{opacity:.85;}.run-btn:active{transform:scale(.96);}.run-btn i{font-size:13px;}"
        ".run-btn.running{opacity:.6;cursor:not-allowed;}.run-btn.done{background:#1D9E75 !important;}"
        ".info-bar{margin-top:11px;background:#f7fdf9;border:1px solid #d4ede5;border-radius:8px;padding:9px 14px;font-size:11px;color:#5F9E8A;display:flex;align-items:center;gap:7px;}"
        ".info-bar i{font-size:15px;color:#1D9E75;flex-shrink:0;}"
        ".stats-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(90px,1fr));gap:8px;}"
        ".stat{background:#fff;border:1px solid #d4ede5;border-radius:10px;padding:11px;text-align:center;}"
        ".stat .sv{font-size:23px;font-weight:700;}.stat .sl{font-size:10px;color:#5F9E8A;margin-top:2px;}"
        ".s1 .sv{color:#1D9E75;}.s2 .sv{color:#378ADD;}.s3 .sv{color:#639922;}.s4 .sv{color:#BA7517;}.s5 .sv{color:#7F77DD;}"
        ".sensors-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:9px;}"
        ".sc{border-radius:10px;padding:11px 13px;display:flex;flex-direction:column;gap:3px;}"
        ".sc.s1{background:#E1F5EE;border:1px solid #9FE1CB;}.sc.s2{background:#E6F1FB;border:1px solid #B5D4F4;}"
        ".sc.s3{background:#FAEEDA;border:1px solid #FAC775;}.sc.s4{background:#FCEBEB;border:1px solid #F7C1C1;}"
        ".sc i{font-size:21px;}"
        ".s1 i{color:#0F6E56;}.s2 i{color:#185FA5;}.s3 i{color:#854F0B;}.s4 i{color:#A32D2D;}"
        ".sc .sn{font-size:12px;font-weight:600;}.s1 .sn{color:#085041;}.s2 .sn{color:#0C447C;}.s3 .sn{color:#633806;}.s4 .sn{color:#791F1F;}"
        ".sc .sm{font-size:10px;font-family:monospace;letter-spacing:.5px;}"
        ".s1 .sm{color:#0F6E56;}.s2 .sm{color:#185FA5;}.s3 .sm{color:#854F0B;}.s4 .sm{color:#A32D2D;}"
        ".sc .sr{font-size:10px;color:#888;}"
        ".flow-wrap{display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:7px;}"
        ".fb{border-radius:9px;padding:9px 13px;font-size:11px;font-weight:600;text-align:center;min-width:88px;}"
        ".fb.ft{background:#E1F5EE;border:1px solid #9FE1CB;color:#085041;}"
        ".fb.fb2{background:#E6F1FB;border:1px solid #B5D4F4;color:#0C447C;}"
        ".fb.fg{background:#EAF3DE;border:1px solid #C0DD97;color:#27500A;}"
        ".fb .fb-sub{font-size:9px;font-weight:400;opacity:.75;display:block;margin-top:1px;}"
        ".arr{color:#9FE1CB;font-size:14px;}"
        ".two-col{display:grid;grid-template-columns:1fr 1fr;gap:12px;}"
        "@media(max-width:540px){.two-col{grid-template-columns:1fr;}}"
        ".algo-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(126px,1fr));gap:8px;}"
        ".ac{border-radius:10px;padding:10px 12px;}"
        ".ac.at{background:#E1F5EE;border:1px solid #9FE1CB;}.ac.ab{background:#E6F1FB;border:1px solid #B5D4F4;}.ac.ag{background:#EAF3DE;border:1px solid #C0DD97;}"
        ".ac .al{font-size:9px;font-weight:600;color:#888;margin-bottom:2px;}.ac .an{font-size:12px;font-weight:700;}"
        ".at .an{color:#085041;}.ab .an{color:#0C447C;}.ag .an{color:#27500A;}"
        ".ac .ad{font-size:10px;color:#666;margin-top:2px;line-height:1.5;}"
        ".ac .aa{font-size:18px;font-weight:700;margin-top:5px;}"
        ".at .aa{color:#1D9E75;}.ab .aa{color:#378ADD;}.ag .aa{color:#639922;}"
        ".lv-row{display:flex;gap:5px;flex-wrap:wrap;}"
        ".lv{flex:1;min-width:68px;border-radius:8px;padding:7px 5px;text-align:center;}"
        ".lv .ln{font-size:12px;font-weight:600;}.lv .lr{font-size:9px;margin-top:1px;}"
        ".lv-s{background:#EAF3DE;border:1px solid #C0DD97;}.lv-s .ln{color:#27500A;}.lv-s .lr{color:#3B6D11;}"
        ".lv-m{background:#FAEEDA;border:1px solid #FAC775;}.lv-m .ln{color:#633806;}.lv-m .lr{color:#854F0B;}"
        ".lv-w{background:#FFF3E0;border:1px solid #FFB74D;}.lv-w .ln{color:#7B4F00;}.lv-w .lr{color:#B56A00;}"
        ".lv-c{background:#FCEBEB;border:1px solid #F7C1C1;}.lv-c .ln{color:#791F1F;}.lv-c .lr{color:#A32D2D;}"
        ".lv-e{background:#3B0000;border:1px solid #A32D2D;}.lv-e .ln{color:#FFAAAA;}.lv-e .lr{color:#FF8888;}"
        ".formula{background:#f7fdf9;border:1px solid #d4ede5;border-radius:10px;padding:13px 15px;text-align:center;}"
        ".feq{font-size:13px;font-weight:600;color:#0d2e26;font-family:monospace;letter-spacing:.3px;}"
        ".fsub{font-size:10px;color:#5F9E8A;margin-top:5px;}"
        ".wpills{display:flex;gap:6px;justify-content:center;margin-top:8px;flex-wrap:wrap;}"
        ".wp{border-radius:14px;padding:3px 11px;font-size:10px;font-weight:600;}"
        ".wp.r{background:#FCEBEB;color:#791F1F;border:1px solid #F7C1C1;}"
        ".wp.b{background:#E6F1FB;color:#0C447C;border:1px solid #B5D4F4;}"
        ".wp.a{background:#FAEEDA;color:#633806;border:1px solid #FAC775;}"
        ".hw-row{display:flex;gap:7px;flex-wrap:wrap;}"
        ".hw{background:#f7fdf9;border:1px solid #d4ede5;border-radius:8px;padding:5px 12px;font-size:11px;color:#085041;display:flex;align-items:center;gap:5px;font-weight:500;}"
        ".hw i{font-size:14px;color:#1D9E75;}"
        ".ftr{background:linear-gradient(135deg,#085041 0%,#0F6E56 50%,#1D9E75 100%);padding:16px 22px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;}"
        ".ftr-info{font-size:11px;color:rgba(255,255,255,0.7);line-height:1.9;}"
        ".ftr-info span{color:white;font-weight:600;}"
        ".ftr-code{background:rgba(255,255,255,0.18);border:1px solid rgba(255,255,255,0.35);border-radius:10px;padding:8px 20px;font-size:15px;color:white;font-weight:700;letter-spacing:2px;}"
        ".toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%) translateY(80px);background:#0F6E56;color:white;border-radius:12px;padding:11px 20px;font-family:\'Cairo\',sans-serif;font-size:13px;font-weight:600;transition:transform .3s;z-index:9999;display:flex;align-items:center;gap:8px;white-space:nowrap;}"
        ".toast.show{transform:translateX(-50%) translateY(0);}.toast.err{background:#A32D2D;}"
        ".ar{display:block;}.en{display:none;}"
        "[lang=en] .ar{display:none;}[lang=en] .en{display:block;}"
        "[lang=en] body{direction:ltr;}"
        "@media print{body{background:white;padding:0;}.poster{box-shadow:none;border-radius:0;max-width:100%;}.run-btn,.toast{display:none;}}"
    )

    # بناء HTML — كلية يمين (COL أول في RTL) ، جامعة يسار (UNI آخر)
    html = (
        '<!DOCTYPE html>\n<html lang="ar" dir="rtl">\n<head>\n'
        '<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        '<title>EnviroAI — Intelligent IoT &amp; ML Environmental System</title>\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;700;900&family=Inter:wght@400;600;700&display=swap">\n'
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css">\n'
        '<style>' + css + '</style>\n</head>\n<body>\n'
        '<div id="toast" class="toast"><i class="ti ti-rocket"></i><span id="tmsg"></span></div>\n'
        '<div class="poster">\n'

        # ── LANG SWITCHER + HEADER ──
        '<div id="langBar" style="position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:9999;display:flex;gap:0;border-radius:24px;overflow:hidden;box-shadow:0 4px 18px rgba(0,0,0,.22);border:2px solid #ccc;background:#fff;">'
        '<button id="btnAr" onclick="setLang(\'ar\')" style="padding:9px 28px;font-size:13px;font-weight:700;cursor:pointer;border:none;background:#0F6E56;color:white;font-family:Cairo,sans-serif;">\u0627\u0644\u0639\u0631\u0628\u064a\u0629</button>'
        '<button id="btnEn" onclick="setLang(\'en\')" style="padding:9px 28px;font-size:13px;font-weight:700;cursor:pointer;border:none;background:rgba(255,255,255,.92);color:#0F6E56;font-family:Inter,sans-serif;">English</button>'
        '</div>\n'
        '<div class="hdr">\n'
        '<div class="hdr-inner" style="position:relative;z-index:1;text-align:center;">\n'
        '<h1 style="font-size:17px;font-weight:900;color:white;line-height:1.65;max-width:700px;margin:0 auto;">'
        '<span class="ar">\u0646\u0638\u0627\u0645 \u0630\u0643\u064a \u0644\u0645\u0631\u0627\u0642\u0628\u0629 \u062c\u0648\u062f\u0629 \u0627\u0644\u0628\u064a\u0626\u0629 \u0648\u062a\u062d\u0644\u064a\u0644 \u0627\u0644\u0627\u0633\u062a\u062f\u0627\u0645\u0629 \u0628\u0627\u0633\u062a\u062e\u062f\u0627\u0645 \u0625\u0646\u062a\u0631\u0646\u062a \u0627\u0644\u0623\u0634\u064a\u0627\u0621 \u0648\u062a\u0639\u0644\u0645 \u0627\u0644\u0622\u0644\u0629</span>'
        '<span class="en">An Intelligent IoT and Machine-Learning-Based System for Environmental Air-Quality Monitoring and Sustainability Analysis</span>'
        '</h1>\n'
        '<div class="hdr-desc ar">\u0646\u0638\u0627\u0645 \u064a\u0639\u062a\u0645\u062f \u0645\u0633\u062a\u0634\u0639\u0631\u0627\u062a \u0645\u062a\u062e\u0635\u0635\u0629 \u0644\u0642\u064a\u0627\u0633 \u062c\u0648\u062f\u0629 \u0627\u0644\u0647\u0648\u0627\u0621 \u0641\u064a \u0627\u0644\u0648\u0642\u062a \u0627\u0644\u0641\u0639\u0644\u064a \u0645\u0639 \u062a\u062d\u0644\u064a\u0644 \u0628\u062e\u0648\u0627\u0631\u0632\u0645\u064a\u0627\u062a \u0627\u0644\u0630\u0643\u0627\u0621 \u0627\u0644\u0627\u0635\u0637\u0646\u0627\u0639\u064a</div>\n'
        '<div class="hdr-desc en">A real-time system using IoT sensors &amp; Machine Learning to monitor air quality and analyze sustainability.</div>\n'
        '<div class="pill-row">'
        '<div class="pill p1"><span class="ar">\u0627\u0644\u0645\u0633\u0627\u0631 \u0627\u0644\u0628\u064a\u0626\u064a</span><span class="en">Environmental Track</span></div>'
        '<div class="pill p2">EnviroAI</div>'
        '<div class="pill p3"><span class="ar">\u0630\u0643\u0627\u0621 \u0627\u0635\u0637\u0646\u0627\u0639\u064a &middot; \u0625\u0646\u062a\u0631\u0646\u062a \u0627\u0644\u0623\u0634\u064a\u0627\u0621</span><span class="en">AI &middot; IoT</span></div>'
        '</div>\n'
        '</div>\n'
        '</div>\n'  # end hdr

        # ── BODY ──
        '<div class="body">\n'

        # LAUNCHER
        '<div class="sec">\n'
        '<div class="sec-title"><i class="ti ti-rocket"></i>\u062a\u0634\u063a\u064a\u0644 \u0627\u0644\u0628\u0631\u0627\u0645\u062c \u2014 Launch Applications</div>\n'
        '<div class="apps-grid">' + launcher + '</div>\n'
        '<div class="info-bar" style="justify-content:center;background:linear-gradient(135deg,#EEF6FF 0%,#F0FFF8 50%,#FFFBEA 100%);border:1.5px solid #B5D4F4;">'
        '<a href="https://github.com/docscisi-create?tab=repositories" target="_blank" style="display:inline-flex;align-items:center;gap:0;text-decoration:none;border-radius:12px;overflow:hidden;box-shadow:0 3px 12px rgba(59,130,246,0.18);font-family:monospace;font-size:12px;font-weight:700;">'
        '<!-- Python blue segment -->'
        '<span style="background:#3776AB;color:white;padding:8px 14px;display:flex;align-items:center;gap:6px;">'
        '<svg width="15" height="15" viewBox="0 0 16 16" fill="white" xmlns="http://www.w3.org/2000/svg"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>'
        'GitHub</span>'
        '<!-- Python middle segment -->'
        '<span style="background:#FF80AB;color:white;padding:8px 12px;display:flex;align-items:center;gap:5px;font-weight:800;">'
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg"><path d="M9.86 2C7.73 2 6 3.6 6 5.57v1.65h4.29v.55H4.14C2.11 7.77.5 9.46.5 11.43c0 1.96 1.35 3.62 3.21 4.04l-.01.1h2.87v-1.9c0-1.46 1.25-2.66 2.79-2.66h4.28c1.21 0 2.19-.92 2.19-2.07V5.57C15.83 3.6 14.1 2 11.97 2H9.86zm-1.28 1.9c.55 0 1 .44 1 .97s-.45.97-1 .97c-.56 0-1-.44-1-.97s.44-.97 1-.97zm5.56 5.87h-2.86v1.9c0 1.46-1.25 2.66-2.79 2.66H4.21c-1.21 0-2.21.93-2.21 2.07v3.43C2 21.8 3.73 23 5.86 23h4.28c2.13 0 3.86-1.6 3.86-3.57v-1.65H9.71v-.55h6.15c2.03 0 3.64-1.69 3.64-3.66 0-1.97-1.35-3.62-3.22-4.04v-.1h-2.2zm1.14 7.1c.56 0 1 .44 1 .97s-.44.97-1 .97-1-.44-1-.97.44-.97 1-.97z"/></svg>'
        '\u0644\u0644\u062d\u0635\u0648\u0644 \u0639\u0644\u0649 \u0627\u0644\u0628\u0631\u0627\u0645\u062c</span>'
        '<!-- code tag segment -->'
        '<span style="background:#1D9E75;color:white;padding:8px 13px;font-size:13px;font-weight:900;letter-spacing:1px;">&lt;/&gt;</span>'
        '</a>'
        '</div>\n</div>\n'

        # STATS
        '<div class="stats-row">'
        '<div class="stat s1"><div class="sv">6</div><div class="sl">\u062e\u0648\u0627\u0631\u0632\u0645\u064a\u0627\u062a AI</div></div>'
        '<div class="stat s2"><div class="sv">4</div><div class="sl">\u0645\u0633\u062a\u0634\u0639\u0631\u0627\u062a IoT</div></div>'
        '<div class="stat s3"><div class="sv">99%</div><div class="sl">\u062f\u0642\u0629 \u0627\u0644\u062a\u0635\u0646\u064a\u0641</div></div>'
        '<div class="stat s4"><div class="sv">5</div><div class="sl">\u0645\u0633\u062a\u0648\u064a\u0627\u062a \u062e\u0637\u0631</div></div>'
        '<div class="stat s5"><div class="sv">RT</div><div class="sl">\u0645\u0631\u0627\u0642\u0628\u0629 \u0641\u0648\u0631\u064a\u0629</div></div>'
        '</div>\n'

        # SENSORS
        '<div class="sec"><div class="sec-title"><i class="ti ti-cpu"></i>\u0627\u0644\u0645\u0633\u062a\u0634\u0639\u0631\u0627\u062a \u2014 Sensors</div>'
        '<div class="sensors-grid">'
        '<div class="sc s1"><i class="ti ti-wind"></i><div class="sn">\u062c\u0648\u062f\u0629 \u0627\u0644\u0647\u0648\u0627\u0621</div><div class="sm">MQ-135</div><div class="sr">Air Quality Index</div></div>'
        '<div class="sc s2"><i class="ti ti-thermometer"></i><div class="sn">\u0627\u0644\u062d\u0631\u0627\u0631\u0629 \u0648\u0627\u0644\u0631\u0637\u0648\u0628\u0629</div><div class="sm">DHT22</div><div class="sr">Temperature &amp; Humidity</div></div>'
        '<div class="sc s3"><i class="ti ti-flame"></i><div class="sn">\u0627\u0644\u062f\u062e\u0627\u0646 \u0648\u0627\u0644\u063a\u0627\u0632</div><div class="sm">MQ-2</div><div class="sr">Smoke &amp; Gas</div></div>'
        '<div class="sc s4"><i class="ti ti-skull"></i><div class="sn">\u0623\u0648\u0644 \u0623\u0643\u0633\u064a\u062f \u0627\u0644\u0643\u0631\u0628\u0648\u0646</div><div class="sm">MQ-7</div><div class="sr">Carbon Monoxide</div></div>'
        '</div></div>\n'

        # PIPELINE
        '<div class="sec"><div class="sec-title"><i class="ti ti-arrow-right"></i>\u0645\u0633\u0627\u0631 \u062a\u062f\u0641\u0642 \u0627\u0644\u0628\u064a\u0627\u0646\u0627\u062a</div>'
        '<div class="flow-wrap">'
        '<div class="fb ft">\u0645\u0633\u062a\u0634\u0639\u0631\u0627\u062a<span class="fb-sub">Sensors</span></div>'
        '<div class="arr"><i class="ti ti-arrow-left"></i></div>'
        '<div class="fb fb2">Arduino UNO<span class="fb-sub">Microcontroller</span></div>'
        '<div class="arr"><i class="ti ti-arrow-left"></i></div>'
        '<div class="fb ft">\u0645\u0639\u0627\u0644\u062c\u0629 \u0627\u0644\u0628\u064a\u0627\u0646\u0627\u062a<span class="fb-sub">Preprocessing</span></div>'
        '<div class="arr"><i class="ti ti-arrow-left"></i></div>'
        '<div class="fb fb2">\u062e\u0648\u0627\u0631\u0632\u0645\u064a\u0627\u062a AI<span class="fb-sub">ML Models</span></div>'
        '<div class="arr"><i class="ti ti-arrow-left"></i></div>'
        '<div class="fb fg">\u062a\u0646\u0628\u064a\u0647 \u0641\u0648\u0631\u064a<span class="fb-sub">Alert</span></div>'
        '</div></div>\n'

        # TWO COL
        '<div class="two-col">'
        '<div class="sec"><div class="sec-title"><i class="ti ti-brain"></i>\u062e\u0648\u0627\u0631\u0632\u0645\u064a\u0627\u062a \u0627\u0644\u0630\u0643\u0627\u0621 \u0627\u0644\u0627\u0635\u0637\u0646\u0627\u0639\u064a</div>'
        '<div class="algo-grid">'
        '<div class="ac at"><div class="al">\u062a\u0635\u0646\u064a\u0641</div><div class="an">Random Forest</div><div class="ad">200 Decision Tree</div><div class="aa">99%</div></div>'
        '<div class="ac at"><div class="al">\u062a\u0635\u0646\u064a\u0641</div><div class="an">SVM</div><div class="ad">Hyperplane</div><div class="aa">99%</div></div>'
        '<div class="ac ab"><div class="al">\u0634\u0630\u0648\u0630\u0627\u062a</div><div class="an">LOF</div><div class="ad">\u0643\u062b\u0627\u0641\u0629 \u0627\u0644\u062c\u064a\u0631\u0627\u0646</div></div>'
        '<div class="ac ab"><div class="al">\u0634\u0630\u0648\u0630\u0627\u062a</div><div class="an">Isolation Forest</div><div class="ad">\u0639\u0632\u0644 \u0627\u0644\u0634\u0627\u0630</div></div>'
        '<div class="ac ab"><div class="al">\u0634\u0630\u0648\u0630\u0627\u062a</div><div class="an">One-Class SVM</div><div class="ad">\u062d\u062f\u0648\u062f \u0637\u0628\u064a\u0639\u064a\u0629</div></div>'
        '<div class="ac ag"><div class="al">\u062a\u062c\u0645\u064a\u0639</div><div class="an">K-Means</div><div class="ad">5 \u0645\u062c\u0645\u0648\u0639\u0627\u062a</div></div>'
        '</div></div>'

        '<div style="display:flex;flex-direction:column;gap:12px;">'
        '<div class="sec"><div class="sec-title"><i class="ti ti-alert-triangle"></i>\u0645\u0633\u062a\u0648\u064a\u0627\u062a \u0627\u0644\u062e\u0637\u0631</div>'
        '<div class="lv-row">'
        '<div class="lv lv-s"><div class="ln">\u0622\u0645\u0646</div><div class="lr">0\u201350</div></div>'
        '<div class="lv lv-m"><div class="ln">\u0645\u062a\u0648\u0633\u0637</div><div class="lr">51\u2013100</div></div>'
        '<div class="lv lv-w"><div class="ln">\u062a\u062d\u0630\u064a\u0631</div><div class="lr">101\u2013150</div></div>'
        '<div class="lv lv-c"><div class="ln">\u062d\u0631\u062c</div><div class="lr">151\u2013200</div></div>'
        '<div class="lv lv-e"><div class="ln">\u0637\u0648\u0627\u0631\u0626</div><div class="lr">&gt;200</div></div>'
        '</div></div>'
        '<div class="sec"><div class="sec-title"><i class="ti ti-math-function"></i>\u0645\u0639\u0627\u062f\u0644\u0629 \u0645\u0624\u0634\u0631 \u0627\u0644\u062e\u0637\u0631</div>'
        '<div class="formula">'
        '<div class="feq">RiskIndex = 0.40&times;AQI + 0.30&times;CO + 0.30&times;SMOKE</div>'
        '<div class="wpills"><div class="wp r">AQI \u2014 40%</div><div class="wp b">CO \u2014 30%</div><div class="wp a">SMOKE \u2014 30%</div></div>'
        '<div class="fsub">AQI=100, CO=80, SMOKE=60 &rarr; RiskIndex=82 (\u0645\u062a\u0648\u0633\u0637)</div>'
        '</div></div>'
        '<div class="sec"><div class="sec-title"><i class="ti ti-device-laptop"></i>\u0627\u0644\u0623\u062f\u0648\u0627\u062a \u0648\u0627\u0644\u062a\u0642\u0646\u064a\u0627\u062a</div>'
        '<div class="hw-row">'
        '<div class="hw"><i class="ti ti-circuit-board"></i>Arduino UNO</div>'
        '<div class="hw"><i class="ti ti-brand-python"></i>Python</div>'
        '<div class="hw"><i class="ti ti-chart-bar"></i>Dash + Plotly</div>'
        '<div class="hw"><i class="ti ti-brain"></i>Scikit-learn</div>'
        '<div class="hw"><i class="ti ti-file-spreadsheet"></i>Excel / CSV</div>'
        '</div></div>'
        '</div>'
        '</div>\n'  # end two-col

        '</div>\n'  # end body

        # FOOTER
        '<div class="ftr" style="justify-content:center;">'
        '<div class="ftr-code">EnviroAI</div>'
        '</div>\n'  # end poster

        '<script>\n'
        'function showToast(m,e){var t=document.getElementById("toast");document.getElementById("tmsg").textContent=m;t.className="toast"+(e?" err":"");setTimeout(function(){t.classList.add("show");},10);setTimeout(function(){t.classList.remove("show");},3800);}\n'
        'var RAILWAY_URLS={"enviroai_iraq":"https://enviroaimustansiriyah-production.up.railway.app/","enviro_chatbot.py":"https://enviroai-chatbot-production.up.railway.app/","enviro_chatbot":"https://enviroai-chatbot-production.up.railway.app/","enviro_ai_v3":"https://enviroai-v3-production.up.railway.app/","enviro_ai_v3.py":"https://enviroai-v3-production.up.railway.app/","enviro_xgboost":"https://enviroai-production.up.railway.app/","smart_eco_monitor":"https://smartecomonitor-production.up.railway.app/"};\nfunction runApp(f,n,b){if(b.classList.contains("running"))return;\nif(RAILWAY_URLS[f]){b.classList.add("done");b.innerHTML=\'<i class="ti ti-check"></i> \u064a\u0639\u0645\u0644\';showToast("\u2705 "+n+" \u064a\u0639\u0645\u0644 \u0639\u0644\u0649 Railway",false);setTimeout(function(){window.open(RAILWAY_URLS[f],"_blank");},400);return;}\nb.classList.add("running");b.innerHTML=\'<i class="ti ti-loader-2"></i> \u062c\u0627\u0631\u064a...\';fetch("/run?file="+encodeURIComponent(f)).then(function(r){return r.json();}).then(function(d){b.classList.remove("running");if(d.ok){b.classList.add("done");b.innerHTML=\'<i class="ti ti-check"></i> \u064a\u0639\u0645\u0644\';showToast("\u2705 "+n+" \u064a\u0639\u0645\u0644 \u0639\u0644\u0649 localhost:8050",false);setTimeout(function(){window.open("http://localhost:8050","_blank");},900);}else{b.innerHTML=\'<i class="ti ti-player-play"></i> \u062a\u0634\u063a\u064a\u0644\';showToast("\u274c "+(d.error||"\u062e\u0637\u0623"),true);}}).catch(function(){b.classList.remove("running");b.innerHTML=\'<i class="ti ti-player-play"></i> \u062a\u0634\u063a\u064a\u0644\';showToast("\u274c \u0634\u063a\u0651\u0644: python generate_poster.py \u0623\u0648\u0644\u0627\u064b",true);});}\n'
        'function setLang(l){document.documentElement.setAttribute(\'lang\',l);document.documentElement.setAttribute(\'dir\',l===\'ar\'?\'rtl\':\'ltr\');document.getElementById(\'btnAr\').style.background=l===\'ar\'?\'#0F6E56\':\'rgba(255,255,255,.92)\';document.getElementById(\'btnAr\').style.color=l===\'ar\'?\'white\':\'#0F6E56\';document.getElementById(\'btnEn\').style.background=l===\'en\'?\'#0F6E56\':\'rgba(255,255,255,.92)\';document.getElementById(\'btnEn\').style.color=l===\'en\'?\'white\':\'#0F6E56\';}\n''</script>\n'
        '</body></html>'
    )
    return html


running_proc = None

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/run":
            global running_proc
            f = parse_qs(p.query).get("file", [""])[0]
            result = {"ok": False, "error": "\u0627\u0644\u0645\u0644\u0641 \u063a\u064a\u0631 \u0645\u0648\u062c\u0648\u062f"}
            if f and os.path.exists(f):
                try:
                    if running_proc and running_proc.poll() is None:
                        running_proc.terminate(); time.sleep(0.4)
                    running_proc = subprocess.Popen(
                        [sys.executable, f],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    result = {"ok": True}
                    print(f"  Running: {f}  PID={running_proc.pid}")
                except Exception as e:
                    result = {"ok": False, "error": str(e)}
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        if p.path in ("/", "/" + OUTPUT_FILE):
            try:
                with open(OUTPUT_FILE, "rb") as fh: data = fh.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers(); self.wfile.write(data)
            except FileNotFoundError:
                self.send_response(404); self.end_headers()
            return
        self.send_response(404); self.end_headers()


def main():
    print("\n" + "="*56)
    print("  EnviroAI Poster Generator -- Iraq")
    print("="*56)
    html = build_html()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("  OK  " + os.path.abspath(OUTPUT_FILE))
    print("\n  Programs:")
    for prog in PROGRAMS:
        status = "  found  " if os.path.exists(prog["file"]) else "  missing"
        print('  ' + status + '  ' + prog['file'])

    server = HTTPServer(("0.0.0.0", SERVER_PORT), Handler)  # 0.0.0.0 required
    print(f"\n  Server running on port {SERVER_PORT}")
    print("="*56 + "\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
        if running_proc and running_proc.poll() is None:
            running_proc.terminate()
        server.shutdown()


if __name__ == "__main__":
    main()
