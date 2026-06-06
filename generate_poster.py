"""
EnviroAI Poster Generator -- MUST-012
python generate_poster.py
"""

import os, sys, json, threading, subprocess, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

OUTPUT_FILE = "enviro_poster.html"
# تم تعديل المنفذ الافتراضي إلى 8050 ليتطابق مع طلبك
SERVER_PORT = int(os.environ.get("PORT", 8050))  

UNI_LOGO = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIbGNtcwIQAABtbnRyUkdCIFhZWiAH4gADABQACQAOAB1hY3NwTVNGVAAAAABzYXdzY3RybAAAAAAAAAAAAAAAAAAA9tYAAQAAAADTLWhhbmSdkQA9QICwPUB0LIGepSKOAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAAF9jcHJ0AAABDAAAAAx3dHB0AAABGAAAABRyWFlaAAABLAAAABRnWFlaAAABQAAAABRiWFlaAAABVAAAABRyVFJDAAABaAAAAGBnVFJDAAABaAAAAGBiVFJDAAABaAAAAGBkZXNjAAAAAAAAAAV1UkdCAAAAAAAAAAAAAAAAdGV4dAAAAABDQzAAWFlaIAAAAAAAAPNUAAEAAAABFslYWVogAAAAAAAAb6AAADjyAAADj1hZWiAAAAAAAABilgAAt4kAABjaWFlaIAAAAAAAACSgAAAPhQAAtsRjdXJ2AAAAAAAAACoAA..."

# Simulated database of past configurations
CONFIG_FILE = "poster_configs.json"
try:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        PAST_POSTERS = json.load(f)
except Exception:
    PAST_POSTERS = [
        {
            "id": 1,
            "title": "Air Quality Assessment & Prediction Using EnviroAI",
            "authors": "Dr. Ali J. + AI Assistant",
            "department": "Atmospheric Sciences Dept., Mustansiriyah University",
            "abstract": "This study demonstrates how machine learning models trained on IoT sensor arrays can accurately forecast PM2.5 levels up to 24 hours in advance.",
            "intro": "Urban air pollution remains a critical health hazard. Standard monitoring stations are accurate but sparse, leaving significant resolution gaps.",
            "methods": "We deployed 15 low-cost particulate matter sensors across the campus, feeding continuous metrics into a cloud-hosted LSTM neural network via EnviroAI.",
            "results": "The predictive model achieved an R² score of 0.89 for next-hour forecasting, dropping to 0.74 for a 12-hour look-ahead window.",
            "conclusion": "Real-time AI integration enables proactive campus health alerts and smarter ventilation schedules for university infrastructure.",
            "references": "1. Mustansiriyah Climate Review (2025)\n2. AI & Environmental Tech Journal, Vol. 4."
        }
    ]

PROGRAMS = [
    {"name": "Python 3", "file": sys.executable},
]

def build_html(active_config=None):
    if not active_config:
        active_config = PAST_POSTERS[-1]

    def clean(k):
        return active_config.get(k, "").replace("\n", "<br>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EnviroAI Poster Generator</title>
    <style>
        :root {{
            --bg-gradient-start: #eef5f0;
            --bg-gradient-end: #dcece2;
            --primary-color: #1e4620; 
            --secondary-color: #2e6f40; 
            --accent-color: #8fa89b; 
            --text-dark: #1b2e24;
            --card-bg: #ffffff;
            
            /* ألوان درجات الخطورة البيئية المعتمَدة */
            --severity-low: #2ecc71;      /* أخضر - آمن */
            --severity-moderate: #f1c40f; /* أصفر - متوسط */
            --severity-high: #e67e22;     /* برتقالي - خطر مرتفع */
            --severity-critical: #e74c3c; /* أحمر - خطر حرج */
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, var(--bg-gradient-start), var(--bg-gradient-end));
            color: var(--text-dark);
            min-height: 100vh;
            padding: 20px;
        }}

        header {{
            max-width: 1200px;
            margin: 0 auto 25px auto;
            background: var(--card-bg);
            padding: 20px 30px;
            border-radius: 16px;
            box-shadow: 0 8px 24px rgba(30, 70, 32, 0.08);
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-left: 6px solid var(--primary-color);
        }}

        .header-logo-title {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}

        .logo {{
            height: 70px;
            width: auto;
            object-fit: contain;
            border-radius: 8px;
        }}

        .header-text h1 {{
            font-size: 24px;
            color: var(--primary-color);
            font-weight: 700;
        }}

        .header-text p {{
            font-size: 14px;
            color: #556b5c;
            margin-top: 4px;
        }}

        .visit-btn {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            text-decoration: none;
            padding: 12px 24px;
            border-radius: 30px;
            font-weight: 600;
            font-size: 14px;
            box-shadow: 0 4px 12px rgba(46, 111, 64, 0.2);
            transition: all 0.3s ease;
            white-space: nowrap;
        }}

        .visit-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(46, 111, 64, 0.3);
            opacity: 0.95;
        }}

        .main-container {{
            max-width: 1200px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 25px;
        }}

        @media (max-width: 900px) {{
            .main-container {{
                grid-template-columns: 1fr;
            }}
            header {{
                flex-direction: column;
                text-align: center;
                gap: 15px;
            }}
            .header-logo-title {{
                flex-direction: column;
            }}
        }}

        .panel {{
            background: var(--card-bg);
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 8px 24px rgba(30, 70, 32, 0.06);
            height: fit-content;
        }}

        .panel h2 {{
            color: var(--primary-color);
            font-size: 18px;
            margin-bottom: 20px;
            padding-bottom: 8px;
            border-bottom: 2px solid #e1ebe5;
        }}

        .form-group {{
            margin-bottom: 16px;
        }}

        .form-group label {{
            display: block;
            font-size: 13px;
            font-weight: 600;
            color: #44584c;
            margin-bottom: 6px;
        }}

        .form-group input, .form-group textarea {{
            width: 100%;
            padding: 10px 14px;
            border: 1.5px solid #cedcd3;
            border-radius: 8px;
            font-family: inherit;
            font-size: 14px;
            color: var(--text-dark);
            background-color: #fafdfb;
            transition: border-color 0.2s;
        }}

        .form-group input:focus, .form-group textarea:focus {{
            outline: none;
            border-color: var(--secondary-color);
            background-color: #fff;
        }}

        .submit-btn {{
            width: 100%;
            background: var(--primary-color);
            color: white;
            border: none;
            padding: 12px;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
            margin-top: 10px;
        }}

        .submit-btn:hover {{
            background: var(--secondary-color);
        }}

        .poster-preview {{
            background: white;
            border-radius: 16px;
            padding: 40px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            border: 1px solid #e2ece5;
            position: relative;
        }}

        .poster-header {{
            text-align: center;
            border-bottom: 3px solid var(--primary-color);
            padding-bottom: 20px;
            margin-bottom: 25px;
        }}

        .poster-title {{
            font-size: 28px;
            color: var(--primary-color);
            font-weight: 800;
            line-height: 1.3;
            margin-bottom: 10px;
        }}

        .poster-authors {{
            font-size: 16px;
            color: var(--secondary-color);
            font-weight: 600;
            margin-bottom: 4px;
        }}

        .poster-dept {{
            font-size: 13px;
            color: #667e6f;
            font-style: italic;
        }}

        .severity-legend {{
            background: #f4f9f6;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 25px;
            border: 1px solid #e1ebe5;
        }}

        .severity-legend-title {{
            font-size: 13px;
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            text-align: center;
        }}

        .severity-scales {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            text-align: center;
        }}

        .scale-box {{
            padding: 8px 4px;
            border-radius: 6px;
            color: white;
            font-size: 12px;
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .scale-low {{ background-color: var(--severity-low); }}
        .scale-mod {{ background-color: var(--severity-moderate); color: #333; }}
        .scale-high {{ background-color: var(--severity-high); }}
        .scale-crit {{ background-color: var(--severity-critical); }}

        .poster-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}

        @media (max-width: 600px) {{
            .poster-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .poster-section {{
            background: #fdfefe;
            border: 1px solid #ebf2ee;
            border-radius: 8px;
            padding: 18px;
            transition: transform 0.2s;
        }}
        
        .poster-section:hover {{
            transform: translateY(-2px);
            border-color: var(--accent-color);
        }}

        .poster-section h3 {{
            color: var(--primary-color);
            font-size: 15px;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-left: 3px solid var(--secondary-color);
            padding-left: 8px;
        }}

        .poster-section p {{
            font-size: 13.5px;
            line-height: 1.6;
            color: #334439;
            text-align: justify;
        }}

        .full-width {{
            grid-column: span 2;
        }}
        @media (max-width: 600px) {{
            .full-width {{
                grid-column: span 1;
            }}
        }}

        .poster-footer {{
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid #e1ebe5;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 11px;
            color: #8fa89b;
        }}
    </style>
</head>
<body>

    <header>
        <div class="header-logo-title">
            <img src="{UNI_LOGO}" alt="University Logo" class="logo">
            <div class="header-text">
                <h1>EnviroAI Poster System</h1>
                <p>Mustansiriyah University — Environmental AI Dashboard</p>
            </div>
        </div>
        <a href="https://enviroaimustansiriyah-production.up.railway.app/" target="_blank" class="visit-btn">
            Visit Live Dashboard →
        </a>
    </header>

    <div class="main-container">
        
        <div class="panel">
            <h2>Poster Content Editor</h2>
            <form action="/generate" method="POST">
                <div class="form-group">
                    <label>Poster Title</label>
                    <input type="text" name="title" value="{active_config.get('title', '')}" required>
                </div>
                <div class="form-group">
                    <label>Authors</label>
                    <input type="text" name="authors" value="{active_config.get('authors', '')}" required>
                </div>
                <div class="form-group">
                    <label>Department / Institution</label>
                    <input type="text" name="department" value="{active_config.get('department', '')}">
                </div>
                <div class="form-group">
                    <label>Abstract</label>
                    <textarea name="abstract" rows="3">{active_config.get('abstract', '')}</textarea>
                </div>
                <div class="form-group">
                    <label>Introduction</label>
                    <textarea name="intro" rows="3">{active_config.get('intro', '')}</textarea>
                </div>
                <div class="form-group">
                    <label>Methodology</label>
                    <textarea name="methods" rows="3">{active_config.get('methods', '')}</textarea>
                </div>
                <div class="form-group">
                    <label>Results & Analysis</label>
                    <textarea name="results" rows="3">{active_config.get('results', '')}</textarea>
                </div>
                <div class="form-group">
                    <label>Conclusion</label>
                    <textarea name="conclusion" rows="2">{active_config.get('conclusion', '')}</textarea>
                </div>
                <div class="form-group">
                    <label>References</label>
                    <textarea name="references" rows="2">{active_config.get('references', '')}</textarea>
                </div>
                <button type="submit" class="submit-btn">Update & Render Poster</button>
            </form>
        </div>

        <div class="poster-preview">
            
            <div class="poster-header">
                <div class="poster-title">{clean('title')}</div>
                <div class="poster-authors">{clean('authors')}</div>
                <div class="poster-dept">{clean('department')}</div>
            </div>

            <div class="severity-legend">
                <div class="severity-legend-title">Environmental Risk & Severity Indicator</div>
                <div class="scales row severity-scales">
                    <div class="scale-box scale-low">Low Risk (Safe)</div>
                    <div class="scale-box scale-mod">Moderate</div>
                    <div class="scale-box scale-high">High Risk</div>
                    <div class="scale-box scale-crit">Critical Alert</div>
                </div>
            </div>

            <div class="poster-grid">
                <div class="poster-section full-width">
                    <h3>Abstract</h3>
                    <p>{clean('abstract')}</p>
                </div>
                
                <div class="poster-section">
                    <h3>Introduction</h3>
                    <p>{clean('intro')}</p>
                </div>

                <div class="poster-section">
                    <h3>Methodology</h3>
                    <p>{clean('methods')}</p>
                </div>

                <div class="poster-section full-width">
                    <h3>Results & Analysis</h3>
                    <p>{clean('results')}</p>
                </div>

                <div class="poster-section">
                    <h3>Conclusion</h3>
                    <p>{clean('conclusion')}</p>
                </div>

                <div class="poster-section">
                    <h3>References</h3>
                    <p>{clean('references')}</p>
                </div>
            </div>

            <div class="poster-footer">
                <span>Powered by EnviroAI Platform</span>
                <span>Mustansiriyah University &copy; 2026</span>
            </div>
        </div>

    </div>

</body>
</html>
"""
    return html


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/favicon.ico":
            self.send_response(404); self.end_headers(); return

        if p.path in ("/", "/" + OUTPUT_FILE):
            try:
                with open(OUTPUT_FILE, "rb") as fh:
                    data = fh.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except FileNotFoundError:
                self.send_response(404); self.end_headers()
            return
        self.send_response(404); self.end_headers()

    def do_POST(self):
        p = urlparse(self.path)
        if p.path == "/generate":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            qs = parse_qs(post_data)
            
            config = {
                "id": len(PAST_POSTERS) + 1,
                "title": qs.get("title", [""])[0],
                "authors": qs.get("authors", [""])[0],
                "department": qs.get("department", [""])[0],
                "abstract": qs.get("abstract", [""])[0],
                "intro": qs.get("intro", [""])[0],
                "methods": qs.get("methods", [""])[0],
                "results": qs.get("results", [""])[0],
                "conclusion": qs.get("conclusion", [""])[0],
                "references": qs.get("references", [""])[0],
            }
            
            PAST_POSTERS.append(config)
            try:
                with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                    json.dump(PAST_POSTERS, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print("Error saving config:", e)

            updated_html = build_html(config)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
                fh.write(updated_html)

            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()
            return
        
        self.send_response(404); self.end_headers()


def main():
    print("\n" + "="*56)
    print("  EnviroAI Poster Generator -- MUST-012")
    print("="*56)
    html = build_html()
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("  OK  " + os.path.abspath(OUTPUT_FILE))
    print("\n  Programs:")
    for prog in PROGRAMS:
        status = "  found  " if os.path.exists(prog["file"]) else "  missing"
        print('  ' + status + '  ' + prog['file'])

    server = HTTPServer(("0.0.0.0", SERVER_PORT), Handler)
    print(f"\n  Server running on port {SERVER_PORT}...")
    print(f"  👉 Open browser at: http://localhost:{SERVER_PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped manually.")


if __name__ == "__main__":
    main()
