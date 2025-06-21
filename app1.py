# app1.py

import os
import sys
import threading
import zipfile
import shutil
import traceback
import webbrowser
import tkinter as tk
from PIL import Image, ImageDraw
import pystray
import requests
from flask import Flask, render_template, request, jsonify

# ─── Helper to locate resources in PyInstaller bundles ───────────────────────
def resource_path(rel):
    try:
        base = sys._MEIPASS
    except AttributeError:
        base = os.path.abspath(".")
    return os.path.join(base, rel)

# ─── Splash Screen ────────────────────────────────────────────────────────────
def show_splash():
    root = tk.Tk()
    root.overrideredirect(True)
    w, h = 400, 120
    x = (root.winfo_screenwidth() - w) // 2
    y = (root.winfo_screenheight() - h) // 3
    root.geometry(f"{w}x{h}+{x}+{y}")
    frame = tk.Frame(root, bg="#111")
    frame.pack(fill="both", expand=True)
    tk.Label(frame,
             text="🧠 Launching CheetChat…\nPlease wait up to 3 minutes",
             fg="#fff", bg="#111",
             font=("Segoe UI", 12),
             pady=10).pack()
    from tkinter.ttk import Progressbar, Style
    style = Style()
    style.theme_use("clam")
    style.configure("S.Horizontal.TProgressbar",
                    troughcolor="#111",
                    background="#e74c3c",
                    bordercolor="#111")
    pb = Progressbar(frame, style="S.Horizontal.TProgressbar", mode="indeterminate")
    pb.pack(fill="x", padx=20, pady=10)
    pb.start(15)
    # auto‑close splash after 8s or when flask is ready
    root.after(8000, root.destroy)
    root.mainloop()

# fire off splash in background
threading.Thread(target=show_splash, daemon=True).start()

# ─── Flask Setup ──────────────────────────────────────────────────────────────
UPLOAD_DIR = resource_path("uploaded_models")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(
    __name__,
    template_folder=resource_path("templates"),
    static_folder=resource_path("static")
)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
chatbot = None

def allowed_file(fn):
    return fn.lower().endswith(".zip")

@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception:
        traceback.print_exc()
        return "❌ Failed to load index.html", 500

@app.route("/upload", methods=["POST"])
def upload():
    if "model_zip" not in request.files:
        return "No file part", 400
    f = request.files["model_zip"]
    if not f or not allowed_file(f.filename):
        return "Invalid file", 400

    model_dir = os.path.join(UPLOAD_DIR, "model")
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    tmp = os.path.join(UPLOAD_DIR, "tmp.zip")
    f.save(tmp)
    with zipfile.ZipFile(tmp, "r") as z:
        z.extractall(model_dir)
    os.remove(tmp)

    required = {"config.json","pytorch_model.bin",
                "tokenizer_config.json","merges.txt","vocab.json"}
    if not required.issubset(os.listdir(model_dir)):
        return f"ZIP must contain: {', '.join(required)}", 400

    try:
        from infer1 import TorchChatbot
        global chatbot
        chatbot = TorchChatbot(model_dir)
        return "OK", 200
    except Exception as e:
        traceback.print_exc()
        return f"Load failed: {e}", 500

@app.route("/chat")
def chat_page():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat_api():
    if chatbot is None:
        return jsonify({"response":"Upload a model first"}), 400
    msg = request.json.get("message","").strip()
    if not msg:
        return jsonify({"response":"Please type something"}), 400
    try:
        out = chatbot.generate(msg)
    except Exception as e:
        traceback.print_exc()
        out = f"Error: {e}"
    return jsonify({"response": out})

@app.route("/shutdown", methods=["POST"])
def shutdown():
    func = request.environ.get("werkzeug.server.shutdown")
    if not func:
        return "Not running with Werkzeug", 500
    func()
    return "Shutting down…"

# ─── System Tray Icon ─────────────────────────────────────────────────────────
def make_tray_icon():
    # create a red cross on transparent background
    img = Image.new("RGBA", (64,64), (0,0,0,0))
    d  = ImageDraw.Draw(img)
    # draw two thick red lines
    d.line((16,16,48,48), fill="#e74c3c", width=8)
    d.line((48,16,16,48), fill="#e74c3c", width=8)
    return img

def on_quit(icon, item):
    try:
        requests.post("http://127.0.0.1:8080/shutdown", timeout=1)
    except:
        pass
    icon.stop()
    sys.exit(0)

def start_tray():
    icon = pystray.Icon(
        "cheetchat",
        icon=make_tray_icon(),
        title="CheetChat",
        menu=pystray.Menu(
            pystray.MenuItem("Quit", on_quit, default=True)
        )
    )
    icon.run()

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    # 1) Start Flask in background
    threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=8080, debug=False),
        daemon=True
    ).start()
    # 2) Open browser after brief delay
    threading.Timer(2.0, lambda: webbrowser.open("http://127.0.0.1:8080")).start()
    # 3) Show system tray menu
    start_tray()
