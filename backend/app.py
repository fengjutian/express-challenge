import base64
import io
import json
import torch
from PIL import Image
from fastapi import FastAPI, WebSocket
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
            obj = json.loads(data)
            img_data = base64.b64decode(obj["image"])
            image = Image.open(io.BytesIO(img_data)).convert("RGB")

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
        except Exception as e:
            print("❌ 连接关闭:", e)
            break
