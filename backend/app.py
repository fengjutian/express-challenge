"""人脸情感识别后端服务

该模块使用FastAPI构建WebSocket服务，接收前端发送的人脸图像，并使用预训练的
Vision Transformer模型进行情感分类。主要功能包括：
1. 建立WebSocket连接接收人脸图像数据
2. 预处理图像并进行模型推理
3. 返回情感分类结果和置信度
"""

import base64
import io
import json
import torch
from PIL import Image
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketDisconnect
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# 初始化FastAPI应用实例
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables for model and processor
extractor = None
model = None
id2label = None

@app.on_event("startup")
async def startup_event():
    global extractor, model, id2label
    try:
        print("🔄 Loading model from ./vit-face-expression...")
        extractor = AutoFeatureExtractor.from_pretrained("./vit-face-expression")
        model = AutoModelForImageClassification.from_pretrained("./vit-face-expression").eval()
        id2label = model.config.id2label
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise e  # Re-raise to prevent app from starting with broken model

@app.websocket("/ws/emotion")
async def websocket_endpoint(websocket: WebSocket):
    if not all([extractor, model, id2label]):
        print("❌ Model not initialized")
        return

    await websocket.accept()
    print(f"✅ WebSocket connected from {websocket.client.host}:{websocket.client.port}")
    while True:
        try:
            # 接收前端发送的人脸ROI（Base64）
            print("👂 Waiting for data from frontend...")
            data = await websocket.receive_text()
            print("... Received data.")

            # Guard against malformed JSON or missing fields
            try:
                obj = json.loads(data)
                img_b64 = obj.get("image")
                if not img_b64:
                    # Bad payload, skip
                    print("⚠️ Missing 'image' field in JSON payload.")
                    await websocket.send_json({"error": "missing image field"})
                    continue
            except json.JSONDecodeError:
                print("⚠️ Invalid JSON received.")
                await websocket.send_json({"error": "invalid json"})
                continue

            try:
                print("🖼️ Decoding base64 image...")
                img_data = base64.b64decode(img_b64)
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
                print("... Image decoded successfully.")
            except Exception as decode_err:
                await websocket.send_json({"error": "invalid image data"})
                print("⚠️ invalid image data:", decode_err)
                continue

            # 模型推理
            print("🧠 Performing model inference...")
            inputs = extractor(images=image, return_tensors="pt")
            # 禁用梯度计算，提高推理速度并减少内存使用
            with torch.no_grad():
                # 前向传播获取模型输出的原始分数
                logits = model(**inputs).logits
                # 应用softmax函数获取概率分布
                probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            # 获取最高概率对应的类别索引
            pred = int(probs.argmax())
            # 获取对应的情感标签
            label = id2label[pred]
            # 获取预测的置信度
            conf = float(probs[pred])
            print(f"... Inference complete. Detected: {label} (Confidence: {conf:.2f})")


            # 发送结果
            print("📤 Sending results to frontend...")
            await websocket.send_json({
                "emotion": label,    # 识别出的情感类别
                "confidence": conf   # 预测置信度
            })
            print("... Results sent.")
        except WebSocketDisconnect as e:
            # Client closed the connection (code available on some websockets implementations)
            print("❌ WebSocket 客户端断开:", getattr(e, 'code', repr(e)))
            break
        except Exception as e:
            # Other errors (log and continue loop or break as appropriate)
            print("❌ 连接处理出错:", e)
            # For safety, break to avoid tight exception loops. Client can reconnect.
            break
