# 在本机调用 Gemini 3 Pro 模型指南

## 1. 启动本地代理

每次使用前需先启动代理（如已运行可跳过）：

```bash
source /newcpfs/user/qixuan1/310/miniforge3/use_shared_loongflow_ml.sh
cd /newcpfs/user/qixuan1/310/LoongFlow

export UPSTREAM_API_KEY="YOUR_API_KEY"
export UPSTREAM_URL="https://runway.devops.rednote.life/openai/google/v1:generateContent"
export ALLOWED_MODELS="gemini/gemini-3-flash-preview,gemini-3-flash-preview,gemini3_pro,gemini/gemini3_pro"
export REQUEST_TIMEOUT=300

nohup python -m uvicorn local_proxy.gemini_gateway_proxy:app \
  --host 127.0.0.1 \
  --port 8010 \
  > /newcpfs/user/qixuan1/310/LoongFlow/local_proxy/gemini_proxy_8010.log 2>&1 &

sleep 4
curl http://127.0.0.1:8010/healthz
```

healthz 返回正常即代理就绪。

---

## 2. 关键信息

| 项目 | 值 |
|------|-----|
| 代理地址 | `http://127.0.0.1:8010` |
| API Key | `YOUR_API_KEY` |
| 可用模型 | `gemini3_pro` / `gemini-3-flash-preview` |
| URL 格式 | `/models/<model_name>:generateContent` |

**注意**：路径必须是 `/models/<model_name>:generateContent`，不能直接用 `/v1:generateContent`。

---

## 3. Python 调用示例（推荐）

```python
import urllib.request
import json

url = "http://127.0.0.1:8010/models/gemini3_pro:generateContent"
api_key = "YOUR_API_KEY"

payload = {
    "contents": [
        {
            "role": "user",
            "parts": [{"text": "你的问题"}]
        }
    ],
    "generationConfig": {
        "temperature": 1,
        "maxOutputTokens": 65535,
        "topP": 0.95,
        "thinkingConfig": {
            "thinkingLevel": "HIGH",   # 或 "LOW"
            "includeThoughts": True    # False 则不返回思考过程
        }
    }
}

data = json.dumps(payload).encode()
req = urllib.request.Request(url, data=data, headers={
    "api-key": api_key,
    "Content-Type": "application/json"
})

with urllib.request.urlopen(req) as resp:
    result = json.loads(resp.read().decode())
    print(json.dumps(result, ensure_ascii=False, indent=2))
```

测试脚本已保存在 `/tmp/test_gemini.py`，可直接运行：

```bash
python3 /tmp/test_gemini.py
```

---

## 4. 多轮对话示例

```python
import urllib.request
import json

url = "http://127.0.0.1:8010/models/gemini3_pro:generateContent"
api_key = "YOUR_API_KEY"

def chat(history: list, user_msg: str) -> str:
    history.append({"role": "user", "parts": [{"text": user_msg}]})
    payload = {
        "contents": history,
        "generationConfig": {"temperature": 1, "maxOutputTokens": 65535, "topP": 0.95}
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={
        "api-key": api_key,
        "Content-Type": "application/json"
    })
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode())
    reply = result["candidates"][0]["content"]["parts"][-1]["text"]
    history.append({"role": "model", "parts": [{"text": reply}]})
    return reply

history = []
print(chat(history, "你好，请介绍一下自己"))
print(chat(history, "你擅长哪些领域？"))
```

---

## 5. 带 System Prompt 示例

```python
payload = {
    "contents": [
        {"role": "user", "parts": [{"text": "组织一场游学"}]}
    ],
    "systemInstruction": {
        "parts": [{"text": "你是一名经验丰富的老师"}]
    },
    "generationConfig": {
        "temperature": 1,
        "maxOutputTokens": 65535,
        "topP": 0.95
    }
}
```

---

## 6. 常见错误

| 错误信息 | 原因 | 解决 |
|----------|------|------|
| `unsupported path` | URL 路径格式错误 | 改为 `/models/gemini3_pro:generateContent` |
| `curl: option -d: requires parameter` | 命令行换行导致参数截断 | 改用 Python 脚本调用 |
| `bash: --header: command not found` | 反斜杠 `\` 后有空格 | 用 Python 或 `-d @file` 方式 |
| 连接拒绝 | 代理未启动 | 重新执行第 1 步启动代理 |
