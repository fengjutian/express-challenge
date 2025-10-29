import base64
import io
import json
import torch
from PIL import Image
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketDisconnect
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

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
