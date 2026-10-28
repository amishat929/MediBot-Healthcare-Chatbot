"""
Web server for the AI Healthcare Chatbot (MediBot).
Run:  python app.py        then open  http://127.0.0.1:5000
"""
import os
import uuid
from flask import Flask, jsonify, render_template, request
from chatbot import HealthBot

app = Flask(__name__)
bot = HealthBot()          # trains the intent classifier once at start-up (about a second)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", ""))[:1000]
    sid = str(data.get("session_id") or uuid.uuid4())
    result = bot.respond(sid, message)
    result["session_id"] = sid
    return jsonify(result)


@app.post("/api/reset")
def reset():
    data = request.get_json(silent=True) or {}
    sid = str(data.get("session_id") or "")
    if sid:
        bot.reset(sid)
    return jsonify({"ok": True})


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "conditions": len(__import__("knowledge_base").DISEASES)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
