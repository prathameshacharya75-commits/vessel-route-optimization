"""
Vessel Route Optimization – Streamlit App
No Flask. No file dependencies beyond models/ and aco_optimizer.py.
"""

import sys, pickle, json, math
from pathlib import Path

# Resolve base directory robustly — os.path.dirname(__file__) returns "" on Streamlit Cloud
BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from aco_optimizer import ACORoutePlanner

st.set_page_config(page_title="Vessel Route Optimizer", page_icon="🚢",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    iframe { border: none !important; }
</style>
""", unsafe_allow_html=True)

MODELS = BASE / "models"
WEATHER_SEVERITY = {"Clear":0,"Cloudy":1,"Overcast":2,"Fog":3,"Rain":4,"Storm":5}

# ── Artifacts ────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        with open(MODELS/"model.pkl",    "rb") as f: model    = pickle.load(f)
        with open(MODELS/"encoders.pkl", "rb") as f: encoders = pickle.load(f)
        with open(MODELS/"features.pkl", "rb") as f: features = pickle.load(f)
        with open(MODELS/"meta.pkl",     "rb") as f: meta     = pickle.load(f)
        with open(MODELS/"metrics.pkl",  "rb") as f: metrics  = pickle.load(f)
        return model, encoders, features, meta, metrics, None
    except Exception as e:
        return None, None, None, None, None, str(e)

MODEL, ENCODERS, FEATURES, META, METRICS, LOAD_ERROR = load_artifacts()

# ── ML helpers ───────────────────────────────────────────
def haversine_km(lat1,lon1,lat2,lon2):
    R=6371; la1,lo1,la2,lo2=map(math.radians,[lat1,lon1,lat2,lon2])
    dlat=la2-la1; dlon=lo2-lo1
    a=math.sin(dlat/2)**2+math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))

def predict_travel_time(turbine_id,port,vessel,weather,wave_height,wind_speed,vessel_speed,hour=12,dow=0,month=6):
    enc=ENCODERS
    try:
        vessel_enc  = enc["vessel"].transform([vessel])[0]
        port_enc    = enc["port"].transform([port])[0]
        turbine_enc = enc["turbine"].transform([turbine_id])[0]
        weather_enc = enc["weather"].transform([weather])[0]
        tech_enc    = enc["tech"].transform(["T001"])[0]
    except Exception as e:
        return None, f"Encoding error: {e}"
    pc     = META["port_coords"].get(port,{"lat":57.69,"lon":-2.51})
    tc_lat = META["turbine_coords"]["latitude"].get(turbine_id,58.2)
    tc_lon = META["turbine_coords"]["longitude"].get(turbine_id,-2.9)
    dist_km=haversine_km(pc["lat"],pc["lon"],tc_lat,tc_lon)
    ws=float(WEATHER_SEVERITY.get(weather,2))
    row={"distance_km":dist_km,"vessel_speed":vessel_speed,"wave_height":wave_height,
         "wind_speed":wind_speed,"vessel_enc":vessel_enc,"port_enc":port_enc,
         "turbine_enc":turbine_enc,"weather_enc":weather_enc,"tech_enc":tech_enc,
         "weather_severity":ws,"effective_speed":vessel_speed/(1+wave_height*0.1),
         "dist_speed_ratio":dist_km/(vessel_speed+1e-5),
         "wind_penalty":wind_speed/(vessel_speed+1e-5),
         "hour":hour,"day_of_week":dow,"month":month}
    X=pd.DataFrame([row])[FEATURES]
    return max(0.1,round(float(MODEL.predict(X)[0]),2)), None

def run_optimise(port,turbine_list,vessel,weather,wave_height,wind_speed):
    vessel_speed=float(META["vessel_speeds"].get(vessel,15.0))
    predictions={}
    for tid in turbine_list:
        t,err=predict_travel_time(tid,port,vessel,weather,wave_height,wind_speed,vessel_speed)
        if err: return {"error":err}
        predictions[tid]={"time":t,"lat":META["turbine_coords"]["latitude"][tid],
                          "lon":META["turbine_coords"]["longitude"][tid]}
    pc=META["port_coords"][port]
    locs=[{"id":f"PORT:{port}","lat":pc["lat"],"lon":pc["lon"],"pred_time":0.0}]
    for tid,info in predictions.items():
        locs.append({"id":tid,"lat":info["lat"],"lon":info["lon"],"pred_time":info["time"]})
    aco=ACORoutePlanner(n_ants=50,n_iterations=150)
    result=aco.optimise(locs)
    route_details=[]; total_dist=0.0; prev_lat,prev_lon=pc["lat"],pc["lon"]
    for rid in result["route"]:
        if rid.startswith("PORT:"): lat,lon,name,ptime=pc["lat"],pc["lon"],port,0.0
        else:
            lat=META["turbine_coords"]["latitude"][rid]; lon=META["turbine_coords"]["longitude"][rid]
            name=rid; ptime=predictions[rid]["time"]
        seg=haversine_km(prev_lat,prev_lon,lat,lon); total_dist+=seg
        route_details.append({"id":rid,"name":name,"lat":lat,"lon":lon,
                               "pred_time_hrs":ptime,"segment_dist_km":round(seg,2)})
        prev_lat,prev_lon=lat,lon
    return {"route":result["route"],"route_details":route_details,
            "total_time_hrs":result["total_time_hrs"],"total_dist_km":round(total_dist,2),
            "total_cost_usd":round(result["total_time_hrs"]*(5000/24),0),
            "predictions":{k:v["time"] for k,v in predictions.items()},
            "convergence":result["convergence"],"port":port,"port_coords":pc,
            "vessel":vessel,"weather":weather}

def build_meta_json():
    tc={k:{"lat":META["turbine_coords"]["latitude"][k],"lon":META["turbine_coords"]["longitude"][k]}
        for k in META["turbines"]}
    return json.dumps({"vessels":META["vessels"],"ports":META["ports"],"turbines":META["turbines"],
                       "weathers":META["weathers"],"turbine_coords":tc,"port_coords":META["port_coords"],
                       "metrics":{"mae":float(METRICS["mae"])if METRICS else 0,
                                  "r2":float(METRICS["r2"])if METRICS else 0}})

# ── Handle optimise query param ──────────────────────────
optimise_result_json = "null"
if "optimise" in st.query_params:
    try:
        payload = json.loads(st.query_params["optimise"])
        result  = run_optimise(port=payload["port"],turbine_list=payload["turbines"],
                               vessel=payload["vessel"],weather=payload["weather"],
                               wave_height=float(payload["wave_height"]),
                               wind_speed=float(payload["wind_speed"]))
        optimise_result_json = json.dumps(result)
    except Exception as e:
        optimise_result_json = json.dumps({"error":str(e)})
    st.query_params.clear()

meta_json  = build_meta_json() if MODEL else "null"
load_error = json.dumps(LOAD_ERROR or "")

# ── HTML (inlined — no file dependency) ─────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vessel Route Optimizer</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root{--bg:#0b1120;--panel:#111827;--card:#1a2436;--border:#1e3a5f;--accent:#00d4ff;--accent2:#7c3aed;--green:#10b981;--yellow:#f59e0b;--red:#ef4444;--text:#e2e8f0;--muted:#64748b;--glow:0 0 20px rgba(0,212,255,.2)}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}
  header{background:linear-gradient(90deg,#0b1120,#111827);border-bottom:1px solid var(--border);padding:14px 28px;display:flex;align-items:center;justify-content:space-between}
  .logo{display:flex;align-items:center;gap:12px}
  .logo-icon{font-size:28px}
  .logo-text h1{font-size:1.3rem;font-weight:700;color:var(--accent);letter-spacing:.5px}
  .logo-text p{font-size:.75rem;color:var(--muted);margin-top:1px}
  .header-badges{display:flex;gap:10px}
  .badge{padding:4px 12px;border-radius:20px;font-size:.72rem;font-weight:600;display:flex;align-items:center;gap:5px}
  .badge-ml{background:rgba(124,58,237,.2);border:1px solid var(--accent2);color:#a78bfa}
  .badge-aco{background:rgba(0,212,255,.1);border:1px solid var(--accent);color:var(--accent)}
  .dot{width:7px;height:7px;border-radius:50%;animation:pulse 1.5s infinite}
  .dot-green{background:var(--green)}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
  .layout{display:grid;grid-template-columns:360px 1fr;height:calc(100vh - 61px)}
  .sidebar{background:var(--panel);border-right:1px solid var(--border);overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:18px}
  .sidebar::-webkit-scrollbar{width:5px}
  .sidebar::-webkit-scrollbar-track{background:var(--bg)}
  .sidebar::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
  .section-title{font-size:.7rem;font-weight:700;letter-spacing:1.2px;color:var(--muted);text-transform:uppercase;margin-bottom:10px;display:flex;align-items:center;gap:7px}
  .section-title::after{content:'';flex:1;height:1px;background:var(--border)}
  .form-group{margin-bottom:14px}
  label{display:block;font-size:.78rem;color:var(--muted);margin-bottom:5px;font-weight:500}
  select,input[type=number],input[type=range]{width:100%;background:var(--card);border:1px solid var(--border);color:var(--text);padding:9px 12px;border-radius:8px;font-size:.85rem;outline:none;transition:border .2s}
  select:focus,input:focus{border-color:var(--accent);box-shadow:var(--glow)}
  option{background:var(--card)}
  .turbine-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .turbine-chip{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:8px 10px;cursor:pointer;transition:all .2s;display:flex;align-items:center;gap:7px;font-size:.8rem}
  .turbine-chip:hover{border-color:var(--accent);background:rgba(0,212,255,.06)}
  .turbine-chip.selected{border-color:var(--accent);background:rgba(0,212,255,.12);color:var(--accent)}
  .turbine-chip input{width:14px;height:14px;accent-color:var(--accent);cursor:pointer}
  .slider-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:4px}
  .slider-val{font-size:.82rem;color:var(--accent);font-weight:600}
  input[type=range]{padding:4px 0;accent-color:var(--accent)}
  .optimise-btn{width:100%;padding:13px;border:none;border-radius:10px;background:linear-gradient(135deg,#0ea5e9,#7c3aed);color:#fff;font-size:.95rem;font-weight:700;cursor:pointer;letter-spacing:.5px;transition:opacity .2s,transform .1s;display:flex;align-items:center;justify-content:center;gap:8px}
  .optimise-btn:hover{opacity:.9}
  .optimise-btn:active{transform:scale(.98)}
  .optimise-btn:disabled{opacity:.5;cursor:not-allowed}
  .ml-info-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px;font-size:.78rem;color:var(--muted);line-height:1.7}
  .ml-info-card .metric-row{display:flex;justify-content:space-between;margin-top:6px}
  .ml-info-card .metric-val{color:var(--green);font-weight:600;font-size:.82rem}
  .main-panel{display:flex;flex-direction:column;overflow:hidden}
  .kpi-bar{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--border);border-bottom:1px solid var(--border)}
  .kpi{background:var(--panel);padding:14px 20px;display:flex;flex-direction:column;gap:4px}
  .kpi-label{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.8px}
  .kpi-value{font-size:1.5rem;font-weight:700;color:var(--accent);line-height:1}
  .kpi-unit{font-size:.72rem;color:var(--muted)}
  .content-area{display:grid;grid-template-rows:1fr auto;flex:1;overflow:hidden}
  #map{width:100%;height:100%;min-height:380px}
  .bottom-panel{background:var(--panel);border-top:1px solid var(--border);display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;background-color:var(--border);max-height:240px}
  .bottom-card{background:var(--panel);padding:16px;overflow:auto}
  .bottom-card h3{font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px}
  .route-step{display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid rgba(30,58,95,.4);font-size:.78rem}
  .step-num{width:20px;height:20px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.65rem;font-weight:700;flex-shrink:0}
  .step-port{background:var(--accent);color:#000}
  .step-turbine{background:var(--accent2);color:#fff}
  .step-info{flex:1}
  .step-name{font-weight:600;color:var(--text)}
  .step-sub{color:var(--muted);font-size:.7rem}
  .step-time{color:var(--green);font-weight:600;white-space:nowrap}
  .chart-wrap{position:relative;height:170px}
  table{width:100%;border-collapse:collapse;font-size:.78rem}
  th{color:var(--muted);font-weight:600;text-align:left;padding:4px 6px;border-bottom:1px solid var(--border);font-size:.7rem}
  td{padding:5px 6px;border-bottom:1px solid rgba(30,58,95,.3)}
  tr:last-child td{border:none}
  .td-good{color:var(--green)}.td-warn{color:var(--yellow)}.td-bad{color:var(--red)}
  #toast{position:fixed;bottom:24px;right:24px;z-index:9999;padding:12px 20px;border-radius:10px;font-size:.85rem;font-weight:500;display:none;animation:slideIn .3s ease}
  .toast-error{background:rgba(239,68,68,.9);color:#fff;border:1px solid var(--red)}
  .toast-success{background:rgba(16,185,129,.9);color:#fff;border:1px solid var(--green)}
  @keyframes slideIn{from{transform:translateY(20px);opacity:0}to{transform:translateY(0);opacity:1}}
  .loading-overlay{display:none;position:absolute;inset:0;background:rgba(11,17,32,.7);z-index:500;align-items:center;justify-content:center;flex-direction:column;gap:12px}
  .loading-overlay.show{display:flex}
  .spinner{width:44px;height:44px;border:3px solid var(--border);border-top-color:var(--accent);border-radius:50%;animation:spin .7s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  .loading-text{color:var(--accent);font-size:.85rem;letter-spacing:.5px}
  .empty-state{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;color:var(--muted);gap:10px;font-size:.85rem}
  .empty-icon{font-size:2.5rem;opacity:.4}
  @media(max-width:900px){.layout{grid-template-columns:1fr}.kpi-bar{grid-template-columns:1fr 1fr}.bottom-panel{grid-template-columns:1fr;max-height:none}}
</style>
</head>
<body>
<header>
  <div class="logo">
    <span class="logo-icon">⚓</span>
    <div class="logo-text">
      <h1>Vessel Route Optimizer</h1>
      <p>ML-Powered · ACO Optimisation · Wind Farm Logistics</p>
    </div>
  </div>
  <div class="header-badges">
    <div class="badge badge-ml"><span>GBM Model</span></div>
    <div class="badge badge-aco"><div class="dot dot-green"></div><span>ACO Ready</span></div>
  </div>
</header>
<div class="layout">
  <aside class="sidebar">
    <div>
      <div class="section-title">⚙️ Configuration Input</div>
      <div class="form-group"><label>🚢 Departure Port</label><select id="sel-port"><option value="">Loading...</option></select></div>
      <div class="form-group"><label>🛥️ Vessel</label><select id="sel-vessel"><option value="">Loading...</option></select></div>
      <div class="form-group"><label>🌤️ Weather Condition</label><select id="sel-weather"><option value="">Loading...</option></select></div>
    </div>
    <div>
      <div class="section-title">🌊 Sea Conditions</div>
      <div class="form-group">
        <div class="slider-row"><label>Wave Height (m)</label><span class="slider-val" id="wave-val">1.0 m</span></div>
        <input type="range" id="sl-wave" min="0.1" max="3.5" step="0.1" value="1.0" oninput="document.getElementById('wave-val').textContent=parseFloat(this.value).toFixed(1)+' m'">
      </div>
      <div class="form-group">
        <div class="slider-row"><label>Wind Speed (knots)</label><span class="slider-val" id="wind-val">10 kts</span></div>
        <input type="range" id="sl-wind" min="0" max="40" step="1" value="10" oninput="document.getElementById('wind-val').textContent=this.value+' kts'">
      </div>
    </div>
    <div>
      <div class="section-title">🏭 Select Turbines</div>
      <div class="turbine-grid" id="turbine-grid"><div style="color:var(--muted);font-size:.8rem">Loading turbines...</div></div>
      <div style="margin-top:8px;display:flex;gap:8px">
        <button onclick="selectAllTurbines(true)"  style="flex:1;padding:5px;background:var(--card);border:1px solid var(--border);color:var(--text);border-radius:6px;cursor:pointer;font-size:.75rem">Select All</button>
        <button onclick="selectAllTurbines(false)" style="flex:1;padding:5px;background:var(--card);border:1px solid var(--border);color:var(--text);border-radius:6px;cursor:pointer;font-size:.75rem">Clear</button>
      </div>
    </div>
    <div>
      <div class="section-title">📦 Manifest ID</div>
      <div class="form-group"><input type="text" id="inp-manifest" placeholder="e.g. BOWL02763" value="BOWL02763" style="width:100%;background:var(--card);border:1px solid var(--border);color:var(--text);padding:9px 12px;border-radius:8px;font-size:.85rem;outline:none"></div>
    </div>
    <button class="optimise-btn" id="btn-optimise" onclick="runOptimisation()">
      <span>🔍</span><span>Run ML + ACO Optimisation</span>
    </button>
    <div class="ml-info-card">
      <strong style="color:var(--text);font-size:.8rem">🤖 Model Performance</strong>
      <div class="metric-row"><span>Algorithm</span><span class="metric-val">Gradient Boosting</span></div>
      <div class="metric-row"><span>MAE</span><span class="metric-val" id="meta-mae">—</span></div>
      <div class="metric-row"><span>R²</span><span class="metric-val" id="meta-r2">—</span></div>
      <div class="metric-row"><span>Training Rows</span><span class="metric-val">3,000</span></div>
      <div class="metric-row"><span>Optimizer</span><span class="metric-val">ACO (50 ants, 150 iter)</span></div>
    </div>
  </aside>
  <main class="main-panel">
    <div class="kpi-bar">
      <div class="kpi"><span class="kpi-label">Best Route</span><span class="kpi-value" id="kpi-stops">—</span><span class="kpi-unit">stops</span></div>
      <div class="kpi"><span class="kpi-label">Total Travel Time</span><span class="kpi-value" id="kpi-time">—</span><span class="kpi-unit">hours</span></div>
      <div class="kpi"><span class="kpi-label">Total Distance</span><span class="kpi-value" id="kpi-dist">—</span><span class="kpi-unit">km</span></div>
      <div class="kpi"><span class="kpi-label">Est. Trip Cost</span><span class="kpi-value" id="kpi-cost">—</span><span class="kpi-unit">USD</span></div>
    </div>
    <div class="content-area" style="position:relative;flex:1">
      <div id="map"></div>
      <div class="loading-overlay" id="loading"><div class="spinner"></div><div class="loading-text">Running ML + ACO optimisation…</div></div>
    </div>
    <div class="bottom-panel">
      <div class="bottom-card"><h3>📍 Optimised Route</h3><div id="route-steps"><div class="empty-state" style="height:120px"><span class="empty-icon">🗺️</span><span>Run optimisation to see route</span></div></div></div>
      <div class="bottom-card"><h3>⏱ ML Travel Predictions</h3><div id="pred-table"><div class="empty-state" style="height:120px"><span class="empty-icon">🤖</span><span>Predictions appear here</span></div></div></div>
      <div class="bottom-card"><h3>📈 ACO Convergence</h3><div class="chart-wrap"><canvas id="convergence-chart"></canvas></div></div>
    </div>
  </main>
</div>
<div id="toast"></div>
<script>
let META={},map,routeLayer,turbineMarkers=[],portMarker,convergenceChart;
document.addEventListener("DOMContentLoaded",()=>{initMap();fetchMeta();initChart();});
function initMap(){
  map=L.map("map",{zoomControl:true}).setView([58.2,-2.92],10);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",{attribution:"© CartoDB",maxZoom:18}).addTo(map);
}
function initChart(){
  const ctx=document.getElementById("convergence-chart").getContext("2d");
  convergenceChart=new Chart(ctx,{type:"line",data:{labels:[],datasets:[{label:"Best Cost",data:[],borderColor:"#00d4ff",backgroundColor:"rgba(0,212,255,.08)",borderWidth:2,pointRadius:0,tension:0.4}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{ticks:{color:"#64748b",maxTicksLimit:6},grid:{color:"rgba(30,58,95,.4)"}},y:{ticks:{color:"#64748b"},grid:{color:"rgba(30,58,95,.4)"}}}}});
}
async function fetchMeta(){
  try{
    const r=await fetch("/api/meta");
    if(!r.ok)throw new Error("Model not trained");
    META=await r.json();
    const wIcons={Clear:"☀️",Cloudy:"⛅",Fog:"🌫️",Overcast:"☁️",Rain:"🌧️",Storm:"⛈️"};
    document.getElementById("sel-port").innerHTML=META.ports.map(p=>`<option value="${p}">${p}</option>`).join("");
    document.getElementById("sel-vessel").innerHTML=META.vessels.map(v=>`<option value="${v}">${v}</option>`).join("");
    document.getElementById("sel-weather").innerHTML=META.weathers.map(w=>`<option value="${w}">${wIcons[w]||""} ${w}</option>`).join("");
    document.getElementById("turbine-grid").innerHTML=META.turbines.map(t=>`
      <div class="turbine-chip" id="chip-${t}" onclick="toggleTurbine('${t}')">
        <input type="checkbox" id="chk-${t}" value="${t}">
        <div><div style="font-weight:600">${t}</div><div style="font-size:.68rem;color:var(--muted)">${META.turbine_coords[t].lat.toFixed(4)}, ${META.turbine_coords[t].lon.toFixed(4)}</div></div>
      </div>`).join("");
    if(META.metrics){document.getElementById("meta-mae").textContent=(META.metrics.mae||0).toFixed(3)+" hrs";document.getElementById("meta-r2").textContent=(META.metrics.r2||0).toFixed(4);}
    plotTurbineMarkers();plotPortMarker(META.ports[0]);
  }catch(e){showToast("⚠ "+e.message+" – check model files","error");}
}
function turbineIcon(sel){return L.divIcon({html:`<div style="background:${sel?"#7c3aed":"#1e3a5f"};border:2px solid ${sel?"#a78bfa":"#00d4ff"};width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:14px;box-shadow:0 0 8px rgba(0,212,255,.4)">⚡</div>`,className:"",iconAnchor:[14,14]});}
function portIcon(){return L.divIcon({html:`<div style="background:#0ea5e9;border:2px solid #fff;width:34px;height:34px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:18px;box-shadow:0 0 12px rgba(14,165,233,.5)">⚓</div>`,className:"",iconAnchor:[17,17]});}
function vesselIcon(){return L.divIcon({html:`<div style="background:#f59e0b;border:2px solid #fff;width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:16px;box-shadow:0 0 10px rgba(245,158,11,.5)">🚢</div>`,className:"",iconAnchor:[15,15]});}
function plotTurbineMarkers(){turbineMarkers.forEach(m=>map.removeLayer(m));turbineMarkers=[];if(!META.turbine_coords)return;Object.entries(META.turbine_coords).forEach(([id,c])=>{const m=L.marker([c.lat,c.lon],{icon:turbineIcon(false)}).addTo(map).bindPopup(`<b>🌀 Turbine ${id}</b><br>Lat: ${c.lat.toFixed(5)}<br>Lon: ${c.lon.toFixed(5)}`);m._turbineId=id;turbineMarkers.push(m);});}
function plotPortMarker(n){if(portMarker)map.removeLayer(portMarker);const c=META.port_coords?.[n];if(!c)return;portMarker=L.marker([c.lat,c.lon],{icon:portIcon()}).addTo(map).bindPopup(`<b>⚓ ${n}</b>`);}
function toggleTurbine(id){const chk=document.getElementById("chk-"+id),chip=document.getElementById("chip-"+id);chk.checked=!chk.checked;chip.classList.toggle("selected",chk.checked);}
function selectAllTurbines(state){META.turbines?.forEach(t=>{document.getElementById("chk-"+t).checked=state;document.getElementById("chip-"+t).classList.toggle("selected",state);});}
document.addEventListener("change",e=>{if(e.target.id==="sel-port")plotPortMarker(e.target.value);});
async function runOptimisation(){
  const port=document.getElementById("sel-port").value,vessel=document.getElementById("sel-vessel").value,weather=document.getElementById("sel-weather").value,wave=parseFloat(document.getElementById("sl-wave").value),wind=parseFloat(document.getElementById("sl-wind").value),turbines=[...document.querySelectorAll("#turbine-grid input:checked")].map(c=>c.value);
  if(!port||!vessel||turbines.length===0){showToast("Please select port, vessel, and at least one turbine","error");return;}
  document.getElementById("loading").classList.add("show");document.getElementById("btn-optimise").disabled=true;
  try{
    const res=await fetch("/api/optimise",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({port,vessel,weather,wave_height:wave,wind_speed:wind,turbines})});
    const data=await res.json();
    if(data._pending){window._streamlitOptimise(data._params);return;}
    if(!res.ok||data.error)throw new Error(data.error||"Optimisation failed");
    renderResults(data);showToast("✓ Optimisation complete – best route found!","success");
  }catch(e){showToast("Error: "+e.message,"error");}
  finally{document.getElementById("loading").classList.remove("show");document.getElementById("btn-optimise").disabled=false;}
}
function renderResults(data){
  document.getElementById("kpi-stops").textContent=data.route_details.length;
  document.getElementById("kpi-time").textContent=data.total_time_hrs.toFixed(2);
  document.getElementById("kpi-dist").textContent=data.total_dist_km.toFixed(1);
  document.getElementById("kpi-cost").textContent="$"+data.total_cost_usd.toLocaleString();
  drawRoute(data);
  document.getElementById("route-steps").innerHTML=data.route_details.map((s,i)=>{const isPort=s.id.startsWith("PORT:");return`<div class="route-step"><div class="step-num ${isPort?"step-port":"step-turbine"}">${i+1}</div><div class="step-info"><div class="step-name">${isPort?"⚓ ":"⚡ "}${s.name}</div><div class="step-sub">${s.segment_dist_km} km · ${s.lat.toFixed(4)}, ${s.lon.toFixed(4)}</div></div><div class="step-time">${isPort?"START":s.pred_time_hrs+"h"}</div></div>`;}).join("");
  document.getElementById("pred-table").innerHTML=`<table><thead><tr><th>Turbine</th><th>Pred. Time</th><th>Weather</th></tr></thead><tbody>${Object.entries(data.predictions).map(([tid,t])=>`<tr><td style="font-weight:600">⚡ ${tid}</td><td class="${t<2?"td-good":t<4?"td-warn":"td-bad"}">${t} hrs</td><td style="color:var(--muted)">${data.weather}</td></tr>`).join("")}</tbody></table>`;
  convergenceChart.data.labels=data.convergence.map((_,i)=>i+1);convergenceChart.data.datasets[0].data=data.convergence;convergenceChart.update();
}
function drawRoute(data){
  if(routeLayer)map.removeLayer(routeLayer);
  const ll=data.route_details.map(s=>[s.lat,s.lon]);
  routeLayer=L.layerGroup();
  L.polyline(ll,{color:"#00d4ff",weight:3,opacity:.85,dashArray:"8 5"}).addTo(routeLayer);
  for(let i=0;i<ll.length-1;i++){const mid=[(ll[i][0]+ll[i+1][0])/2,(ll[i][1]+ll[i+1][1])/2];L.circleMarker(mid,{radius:3,color:"#00d4ff",fillColor:"#00d4ff",fillOpacity:1,weight:0}).addTo(routeLayer);}
  L.marker(ll[0],{icon:vesselIcon()}).bindPopup(`<b>🚢 ${data.vessel}</b>`).addTo(routeLayer);
  routeLayer.addTo(map);map.fitBounds(L.latLngBounds(ll).pad(.15));
  turbineMarkers.forEach(m=>{m.setIcon(turbineIcon(data.route.includes(m._turbineId)));});
}
function showToast(msg,type="success"){const t=document.getElementById("toast");t.textContent=msg;t.className=`toast-${type}`;t.style.display="block";clearTimeout(t._timer);t._timer=setTimeout(()=>t.style.display="none",4000);}
</script>
</body>
</html>"""

# ── Inject Streamlit shim + pre-computed data ─────────────
SHIM = f"""
<script>
const _META   = {meta_json};
const _RESULT = {optimise_result_json};
const _ERR    = {load_error};

window.fetch=(function(orig){{
  return async function(url,opts){{
    if(typeof url==="string"&&url.includes("/api/meta")){{
      if(!_META)return new Response(JSON.stringify({{error:_ERR||"Model not loaded"}}),{{status:503,headers:{{"Content-Type":"application/json"}}}});
      return new Response(JSON.stringify(_META),{{status:200,headers:{{"Content-Type":"application/json"}}}});
    }}
    if(typeof url==="string"&&url.includes("/api/optimise")){{
      if(!_META)return new Response(JSON.stringify({{error:"Model not loaded"}}),{{status:503,headers:{{"Content-Type":"application/json"}}}});
      const body=JSON.parse(opts.body);
      if(_RESULT!==null)return new Response(JSON.stringify(_RESULT),{{status:_RESULT.error?400:200,headers:{{"Content-Type":"application/json"}}}});
      return new Response(JSON.stringify({{_pending:true,_params:body}}),{{status:200,headers:{{"Content-Type":"application/json"}}}});
    }}
    return orig(url,opts);
  }};
}})(window.fetch);

window._streamlitOptimise=function(p){{
  const base=window.parent.location.href.split("?")[0];
  window.parent.location.href=base+"?optimise="+encodeURIComponent(JSON.stringify(p));
}};
</script>
"""

AUTO_RENDER = ""
if optimise_result_json != "null":
    AUTO_RENDER = f"""
<script>
window.addEventListener("load",function(){{
  const r={optimise_result_json};
  if(r&&!r.error){{renderResults(r);showToast("✓ Optimisation complete – best route found!","success");}}
  else if(r&&r.error){{showToast("Error: "+r.error,"error");}}
}});
</script>
"""

html_out = HTML.replace("</head>", SHIM + "</head>", 1)
html_out = html_out.replace("</body>", AUTO_RENDER + "</body>", 1)

components.html(html_out, height=900, scrolling=False)
