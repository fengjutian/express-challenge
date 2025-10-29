import base64
import io
import json
import torch
from PIL import Image
from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

app = FastAPI()

# 加载模型 - 从本地文件夹导入
extractor = AutoFeatureExtractor.from_pretrained("./vit-face-expression")
model = AutoModelForImageClassification.from_pretrained("./vit-face-expression").eval()
id2label = model.config.id2label

@app.websocket("/ws/emotion")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket 已连接")
    while True:
        try:
            # 接收前端发送的人脸ROI（Base64）
            data = await websocket.receive_text()

            # Guard against malformed JSON or missing fields
            try:
                obj = json.loads(data)
                img_b64 = obj.get("image")
                if not img_b64:
                    # Bad payload, skip
                    await websocket.send_json({"error": "missing image field"})
                    continue
            except json.JSONDecodeError:
                await websocket.send_json({"error": "invalid json"})
                continue

            try:
                img_data = base64.b64decode(img_b64)
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
            except Exception as decode_err:
                await websocket.send_json({"error": "invalid image data"})
                print("⚠️ invalid image data:", decode_err)
                continue

            # 模型推理
            inputs = extractor(images=image, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            pred = int(probs.argmax())
            label = id2label[pred]
            conf = float(probs[pred])

            # 发送结果
            await websocket.send_json({
                "emotion": label,
                "confidence": conf
            })
        except WebSocketDisconnect as e:
            # Client closed the connection (code available on some websockets implementations)
            print("❌ WebSocket 客户端断开:", getattr(e, 'code', repr(e)))
            break
        except Exception as e:
            # Other errors (log and continue loop or break as appropriate)
            print("❌ 连接处理出错:", e)
            # For safety, break to avoid tight exception loops. Client can reconnect.
            break
