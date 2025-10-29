# face_detector.py
from facenet_pytorch import MTCNN
import torch

class FaceDetector:
    def __init__(self, device=None, keep_all=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(keep_all=keep_all, device=self.device)

    def detect(self, pil_image):
        """
        输入: PIL.Image (RGB)
        返回: boxes ndarray (N,4) or None
        格式: [x1, y1, x2, y2]（浮点像素坐标）
        """
        boxes, probs = self.mtcnn.detect(pil_image)
        return boxes, probs
