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
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# 初始化FastAPI应用实例
app = FastAPI()

# 加载模型组件 - 从本地文件夹导入预训练模型
# 特征提取器：用于图像预处理，将输入图像转换为模型可接受的格式
extractor = AutoFeatureExtractor.from_pretrained("./vit-face-expression")
# 加载分类模型并设置为评估模式
model = AutoModelForImageClassification.from_pretrained("./vit-face-expression").eval()
# 获取标签映射表，用于将预测索引转换为情感标签
id2label = model.config.id2label

@app.websocket("/ws/emotion")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点函数，处理人脸情感识别的实时通信
    
    Args:
        websocket: WebSocket连接对象，用于与前端进行双向通信
    """
    # 接受WebSocket连接请求
    await websocket.accept()
    print("✅ WebSocket 已连接")
    
    # 持续监听并处理前端消息
    while True:
        try:
            # 接收前端发送的人脸ROI（Base64编码的图像数据）
            data = await websocket.receive_text()
            obj = json.loads(data)
            img_data = base64.b64decode(obj["image"])
            # 将Base64解码后的数据转换为PIL图像对象，并确保为RGB格式
            image = Image.open(io.BytesIO(img_data)).convert("RGB")

            # 模型推理过程
            # 使用特征提取器处理图像
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

            # 将预测结果以JSON格式发送给前端
            await websocket.send_json({
                "emotion": label,    # 识别出的情感类别
                "confidence": conf   # 预测置信度
            })
        except Exception as e:
            # 捕获并记录异常，关闭连接
            print("❌ 连接关闭:", e)
            break
