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

# 配置CORS中间件，允许跨域请求
# 这对于前后端分离架构非常重要，特别是当前端运行在不同的端口或域名时
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（生产环境中应该指定具体的域名）
    allow_credentials=True,  # 允许携带凭证（如cookies）
    allow_methods=["*"],  # 允许所有HTTP方法（GET, POST, PUT等）
    allow_headers=["*"],  # 允许所有HTTP头
)

# 全局模型和处理器变量
# 使用None初始化，在应用启动时加载，避免在导入模块时阻塞
# 这种方式可以提高应用的启动速度和错误处理能力
extractor = None  # 特征提取器：用于图像预处理
model = None     # 分类模型：用于情感识别
id2label = None  # 标签映射：用于将预测索引转换为情感标签

@app.on_event("startup")
async def startup_event():
    """应用启动事件处理函数
    
    在FastAPI应用启动时自动执行，负责加载预训练模型并初始化必要的组件。
    使用异步方式确保不会阻塞应用启动流程，同时提供详细的日志输出。
    """
    global extractor, model, id2label
    try:
        print("🔄 Loading model from ./vit-face-expression...")
        # 从本地预训练模型文件夹加载特征提取器
        extractor = AutoFeatureExtractor.from_pretrained("./vit-face-expression")
        # 加载分类模型并设置为评估模式（禁用dropout等训练时特有的层）
        model = AutoModelForImageClassification.from_pretrained("./vit-face-expression").eval()
        # 获取标签映射表，用于将预测索引转换为情感类别标签
        id2label = model.config.id2label
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        # 重新抛出异常，防止应用在模型加载失败的情况下启动
        # 这样可以确保应用只有在所有必要组件都准备就绪时才会正常运行
        raise e

@app.websocket("/ws/emotion")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点函数，处理人脸情感识别的实时通信
    
    Args:
        websocket: WebSocket连接对象，用于与前端进行双向通信
    
    实现的功能：
    1. 验证模型是否已正确初始化
    2. 接受WebSocket连接请求
    3. 接收并解析前端发送的人脸图像数据
    4. 进行模型推理，识别情感
    5. 将识别结果发送回前端
    6. 处理各种异常情况，确保服务稳定性
    """
    # 检查模型组件是否已正确初始化
    # 如果模型未初始化，拒绝连接并记录错误
    if not all([extractor, model, id2label]):
        print("❌ Model not initialized")
        return

    # 接受WebSocket连接请求
    await websocket.accept()
    # 记录连接信息，便于调试和监控
    print(f"✅ WebSocket connected from {websocket.client.host}:{websocket.client.port}")
    
    # 持续监听并处理前端消息
    while True:
        try:
            # 接收前端发送的人脸ROI（Base64编码的图像数据）
            print("👂 Waiting for data from frontend...")
            data = await websocket.receive_text()
            print("... Received data.")

            # 数据验证和错误处理 - JSON解析
            try:
                obj = json.loads(data)
                img_b64 = obj.get("image")
                # 检查必要的字段是否存在
                if not img_b64:
                    print("⚠️ Missing 'image' field in JSON payload.")
                    # 向客户端发送错误信息
                    await websocket.send_json({"error": "missing image field"})
                    # 跳过当前循环，等待下一条有效消息
                    continue
            except json.JSONDecodeError:
                print("⚠️ Invalid JSON received.")
                # 向客户端发送错误信息
                await websocket.send_json({"error": "invalid json"})
                continue

            # 图像解码和预处理
            try:
                print("🖼️ Decoding base64 image...")
                # 将Base64编码的字符串解码为二进制数据
                img_data = base64.b64decode(img_b64)
                # 将二进制数据转换为PIL图像对象，并确保为RGB格式
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
                print("... Image decoded successfully.")
            except Exception as decode_err:
                # 向客户端发送错误信息
                await websocket.send_json({"error": "invalid image data"})
                print("⚠️ invalid image data:", decode_err)
                continue

            # 模型推理过程
            print("🧠 Performing model inference...")
            # 使用特征提取器处理图像，准备模型输入
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
            # 记录推理结果
            print(f"... Inference complete. Detected: {label} (Confidence: {conf:.2f})")

            # 发送结果
            print("📤 Sending results to frontend...")
            await websocket.send_json({
                "emotion": label,    # 识别出的情感类别
                "confidence": conf   # 预测置信度
            })
            print("... Results sent.")
        except WebSocketDisconnect as e:
            # 客户端主动断开连接的情况
            # 使用getattr获取可能不存在的code属性，提供更好的兼容性
            print("❌ WebSocket 客户端断开:", getattr(e, 'code', repr(e)))
            break
        except Exception as e:
            # 捕获其他所有异常，确保服务不会崩溃
            print("❌ 连接处理出错:", e)
            # 为安全起见，断开连接以避免无限异常循环
            # 客户端可以根据需要重新连接
            break
