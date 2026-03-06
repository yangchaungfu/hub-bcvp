# -*- coding: utf-8 -*-
"""
任务型多轮对话 - 网页服务（ChatGPT 风格）
运行: python web_app.py  然后访问 http://127.0.0.1:8000
"""
import os
import uuid
from pathlib import Path
from typing import Dict
from flask import Flask, jsonify, request

# 确保工作目录为当前脚本所在目录，以便加载 scenario 和 slot 文件
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

from dialog_service import DialogueSystem


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False

# 全局对话系统实例（无状态，仅 memory 按会话维护）
_dialogue_system: DialogueSystem | None = None

# session_id -> 用户记忆（多轮状态）
SESSIONS: Dict[str, dict] = {}

INITIAL_MEMORY = {
    "available_nodes": ["scenario-买衣服_node1", "scenario-看电影_node1"],
}


def get_dialogue_system() -> DialogueSystem:
    global _dialogue_system
    if _dialogue_system is None:
        _dialogue_system = DialogueSystem()
    return _dialogue_system


@app.post("/api/session/start")
def start_session():
    """开始新会话，返回 session_id 和欢迎语。"""
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = dict(INITIAL_MEMORY)
    welcome = "请直接输入你的需求，例如：我想买衣服、我想看电影。"
    return jsonify({"session_id": session_id, "welcome": welcome})


@app.post("/api/chat")
def chat():
    """发送用户消息，调用对话系统，返回回复。"""
    payload = request.get_json(silent=True) or {}  # 把用户请求的 body 按 JSON 解析成一个字典。
    session_id = str(payload.get("session_id", "")).strip()
    message = str(payload.get("message", "")).strip()

    if not session_id or not message:
        return jsonify({"error": "缺少 session_id 或 message"}), 400

    memory = SESSIONS.get(session_id)
    if not memory:
        return jsonify({"error": "会话不存在或已过期，请点击「开始新会话」。"}), 404
        
    ds = get_dialogue_system()
    memory = ds.run(message, memory)
    response = memory.get("response", "")

    # 流程结束，返回一级节点
    if memory.get("end_process", False):
        memory.clear()
        memory["available_nodes"] = ds.lv1_node_info
        memory["end_process"] = False

    SESSIONS[session_id] = memory
    return jsonify({"reply": response})


@app.get("/")
def index():
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>任务型对话</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg-main: #212121;
      --bg-side: #171717;
      --bg-msg-user: #2f2f2f;
      --bg-msg-bot: #2d2d2d;
      --border: #3f3f3f;
      --text: #ececec;
      --text-muted: #8e8e8e;
      --accent: #10a37f;
      --accent-hover: #0d8c6d;
      --input-bg: #40414f;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: "Inter", "PingFang SC", "Microsoft YaHei", sans-serif;
      background: var(--bg-main);
      color: var(--text);
      height: 100vh;
      overflow: hidden;
    }
    .layout {
      display: flex;
      height: 100%;
    }
    .sidebar {
      width: 260px;
      min-width: 260px;
      background: var(--bg-side);
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
    }
    .sidebar-header {
      padding: 12px;
      border-bottom: 1px solid var(--border);
    }
    .sidebar-header .new-chat {
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: transparent;
      color: var(--text);
      font-size: 14px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .sidebar-header .new-chat:hover {
      background: var(--bg-msg-user);
    }
    .main {
      flex: 1;
      display: flex;
      flex-direction: column;
      min-width: 0;
    }
    .chat-header {
      padding: 12px 20px;
      border-bottom: 1px solid var(--border);
      font-size: 14px;
      color: var(--text-muted);
    }
    .chat-header strong { color: var(--text); }
    .messages {
      flex: 1;
      overflow-y: auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }
    .message {
      display: flex;
      gap: 16px;
      max-width: 768px;
      margin: 0 auto;
      width: 100%;
    }
    .message.user { flex-direction: row-reverse; }
    .message .avatar {
      width: 36px;
      height: 36px;
      border-radius: 8px;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 16px;
    }
    .message.user .avatar { background: var(--accent); }
    .message.assistant .avatar { background: var(--bg-msg-bot); border: 1px solid var(--border); }
    .message .content {
      padding: 12px 16px;
      border-radius: 12px;
      line-height: 1.6;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .message.user .content {
      background: var(--bg-msg-user);
      border: 1px solid var(--border);
    }
    .message.assistant .content {
      background: var(--bg-msg-bot);
      border: 1px solid var(--border);
    }
    .composer-wrap {
      padding: 16px 24px 24px;
      border-top: 1px solid var(--border);
    }
    .composer {
      max-width: 768px;
      margin: 0 auto;
      border: 1px solid var(--border);
      border-radius: 12px;
      background: var(--input-bg);
      padding: 12px 16px;
      display: flex;
      gap: 12px;
      align-items: flex-end;
    }
    .composer:focus-within { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
    .composer textarea {
      flex: 1;
      background: transparent;
      border: none;
      color: var(--text);
      font-size: 15px;
      font-family: inherit;
      resize: none;
      min-height: 24px;
      max-height: 200px;
    }
    .composer textarea::placeholder { color: var(--text-muted); }
    .composer textarea:focus { outline: none; }
    .composer button {
      width: 36px;
      height: 36px;
      border-radius: 8px;
      border: none;
      background: var(--accent);
      color: #fff;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    .composer button:hover:not(:disabled) { background: var(--accent-hover); }
    .composer button:disabled { opacity: 0.5; cursor: not-allowed; }
    .empty-state {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      color: var(--text-muted);
      padding: 24px;
      text-align: center;
    }
    .empty-state h2 { font-size: 24px; font-weight: 600; color: var(--text); margin-bottom: 8px; }
    .empty-state p { margin-bottom: 24px; max-width: 400px; }
    .empty-state .hint { font-size: 13px; }
    .status { font-size: 12px; color: var(--text-muted); margin-top: 8px; }
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <div class="sidebar-header">
        <button type="button" class="new-chat" id="newChatBtn">
          <span>+</span> 开始新会话
        </button>
      </div>
    </aside>
    <main class="main">
      <div class="chat-header" id="chatHeader">
        <strong>任务型多轮对话</strong>
        <span id="headerStatus"></span>
      </div>
      <div class="messages" id="messages">
        <div class="empty-state" id="emptyState">
          <h2>任务型对话</h2>
          <p>点击左侧「开始新会话」，然后输入你的需求，系统会按流程引导完成任务。</p>
          <p class="hint">支持买衣服、看电影等场景的多轮填槽与意图识别。</p>
        </div>
      </div>
      <div class="composer-wrap">
        <div class="composer">
          <textarea id="inputBox" placeholder="输入消息… (Enter 发送)" rows="1"></textarea>
          <button type="button" id="sendBtn" title="发送">&#8629;</button>
        </div>
        <div class="status" id="status"></div>
      </div>
    </main>
  </div>
  <script>
    const messagesEl = document.getElementById("messages");
    const emptyState = document.getElementById("emptyState");
    const newChatBtn = document.getElementById("newChatBtn");
    const inputBox = document.getElementById("inputBox");
    const sendBtn = document.getElementById("sendBtn");
    const headerStatus = document.getElementById("headerStatus");
    const statusEl = document.getElementById("status");

    let sessionId = "";

    function appendMessage(role, text) {
      if (emptyState) emptyState.style.display = "none";
      const wrap = document.createElement("div");
      wrap.className = "message " + role;
      const avatar = document.createElement("div");
      avatar.className = "avatar";
      avatar.textContent = role === "user" ? "你" : "助";
      const content = document.createElement("div");
      content.className = "content";
      content.textContent = text;
      wrap.appendChild(avatar);
      wrap.appendChild(content);
      messagesEl.appendChild(wrap);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    async function startSession() {
      const resp = await fetch("/api/session/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      const data = await resp.json();
      if (!resp.ok) {
        appendMessage("assistant", "启动失败：" + (data.error || "未知错误"));
        return;
      }
      sessionId = data.session_id;
      messagesEl.innerHTML = "";
      emptyState.style.display = "none";
      appendMessage("assistant", data.welcome || "会话已开始。");
      headerStatus.textContent = "会话进行中";
      statusEl.textContent = "可直接输入需求";
      inputBox.focus();
    }

    newChatBtn.addEventListener("click", () => startSession());

    async function sendMessage() {
      const message = inputBox.value.trim();
      if (!message) return;
      if (!sessionId) {
        appendMessage("assistant", "请先点击「开始新会话」。");
        return;
      }
      appendMessage("user", message);
      inputBox.value = "";
      inputBox.style.height = "auto";
      sendBtn.disabled = true;
      try {
        const resp = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId, message }),
        });
        const data = await resp.json();
        if (!resp.ok) {
          appendMessage("assistant", "请求失败：" + (data.error || "未知错误"));
          return;
        }
        appendMessage("assistant", data.reply || "");
      } finally {
        sendBtn.disabled = false;
        inputBox.focus();
      }
    }

    sendBtn.addEventListener("click", sendMessage);
    inputBox.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
    inputBox.addEventListener("input", function () {
      this.style.height = "auto";
      this.style.height = Math.min(this.scrollHeight, 200) + "px";
    });
  </script>
</body>
</html>"""


if __name__ == "__main__":
    try:
        app.run(host="127.0.0.1", port=8000, debug=True)
    except Exception as e:
        raise RuntimeError("请先安装依赖: pip install flask pandas openpyxl") from e
