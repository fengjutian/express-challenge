#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
人脸检测模块

该模块提供了基于MTCNN (Multi-Task Cascaded Convolutional Networks) 的人脸检测功能，
能够在图像中定位人脸并返回边界框坐标和置信度。支持GPU加速以提高检测性能。
"""

from facenet_pytorch import MTCNN
import torch


class FaceDetector:
    """
    人脸检测器类
    
    使用MTCNN算法实现高效的人脸检测。该类负责MTCNN模型的初始化和人脸检测功能。
    可以检测图像中的所有人脸，并返回它们的边界框和检测置信度。
    """
    
    def __init__(self, device=None, keep_all=True):
        """
        初始化人脸检测器
        
        Args:
            device (str, optional): 计算设备，如"cuda"或"cpu"。如果为None，将自动选择可用设备
            keep_all (bool): 是否检测并返回图像中所有检测到的人脸，默认为True
        """
        # 设置计算设备，如果未指定则自动选择（优先使用GPU）
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化MTCNN人脸检测器，设置参数
        # keep_all=True表示检测所有人脸，而不仅仅是最可能的人脸
        self.mtcnn = MTCNN(keep_all=keep_all, device=self.device)

    def detect(self, pil_image):
        """
        在图像中检测人脸
        
        Args:
            pil_image (PIL.Image): 输入的RGB图像对象
            
        Returns:
            tuple: (boxes, probs)
                - boxes (ndarray or None): 人脸边界框坐标数组，形状为(N,4)，
                  每一行包含[x1, y1, x2, y2]格式的坐标（浮点像素坐标）
                - probs (ndarray or None): 对应边界框的检测置信度数组，形状为(N,)
                
                如果未检测到人脸，两个返回值都为None
        """
        # 使用MTCNN进行人脸检测
        # 返回检测到的人脸边界框和对应的置信度
        boxes, probs = self.mtcnn.detect(pil_image)
        
        return boxes, probs
