import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import numpy as np
from PIL import Image

class EmotionModel:
    def __init__(self, model_name="trpakov/vit-face-expression"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def predict_face(self, pil_face: Image.Image):
        """
        输入：PIL.Image（单个人脸裁剪后的图像）
        返回：label(str), confidence(float)
        """
        inputs = self.extractor(images=pil_face, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs))
        label = self.id2label[pred_id]
        conf = float(probs[pred_id].cpu().numpy())
        return label, conf
