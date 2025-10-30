#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
情绪识别模型模块

该模块提供了基于Vision Transformer的人脸表情识别功能，使用预训练的模型对人脸图像进行情绪分类。
支持的情绪类别取决于预训练模型的配置。
"""

import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import numpy as np
from PIL import Image


class EmotionModel:
    """
    情绪识别模型类
    
    使用预训练的Vision Transformer模型进行人脸情绪识别。该类负责模型的加载、初始化和推理。
    支持GPU加速（如果可用）。
    """
    
    def __init__(self, model_name="./vit-face-expression"):
        """
        初始化情绪识别模型
        
        Args:
            model_name (str): 模型名称或本地路径，默认为./vit-face-expression（本地模型）
        """
        # 设置计算设备（优先使用GPU，否则使用CPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载特征提取器，用于图像预处理
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        
        # 加载预训练模型并移至指定设备
        self.model = AutoModelForImageClassification.from_pretrained(model_name).to(self.device)
        
        # 设置模型为评估模式（禁用dropout等训练特有操作）
        self.model.eval()
        
        # 获取类别映射字典，用于将预测ID转换为可读标签
        self.id2label = self.model.config.id2label

    def predict_face(self, pil_face: Image.Image):
        """
        对人脸图像进行情绪预测
        
        Args:
            pil_face (PIL.Image): 单个人脸裁剪后的图像对象
            
        Returns:
            tuple: (label, confidence) - 预测的情绪标签和对应的置信度
                - label (str): 预测的情绪类别
                - confidence (float): 预测的置信度，范围[0, 1]
        """
        # 使用特征提取器预处理图像并转换为PyTorch张量
        inputs = self.extractor(images=pil_face, return_tensors="pt").to(self.device)
        
        # 进行推理，禁用梯度计算以提高性能
        with torch.no_grad():
            # 前向传播获取模型输出
            outputs = self.model(**inputs)
            
            # 获取原始预测分数
            logits = outputs.logits
            
            # 应用softmax函数转换为概率分布
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        # 获取最高概率的类别ID
        pred_id = int(torch.argmax(probs))
        
        # 将类别ID转换为可读标签
        label = self.id2label[pred_id]
        
        # 获取对应类别的置信度并转换为NumPy浮点数
        conf = float(probs[pred_id].cpu().numpy())
        
        return label, conf
