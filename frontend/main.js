const video = document.getElementById('video');
const canvas = document.getElementById('output');
const ctx = canvas.getContext('2d');
const resultDiv = document.getElementById('result');

// 连接WebSocket
const socket = new WebSocket("ws://localhost:8000/ws/emotion");

socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  resultDiv.innerText = `Emotion: ${data.emotion} (${(data.confidence * 100).toFixed(1)}%)`;
};

// 初始化 MediaPipe
const faceDetection = new FaceDetection.FaceDetection({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`,
});
faceDetection.setOptions({
  model: 'short',   // 快速模型
  minDetectionConfidence: 0.5
});

// 检测到人脸时
faceDetection.onResults((results) => {
  ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
  if (results.detections.length > 0) {
    const box = results.detections[0].boundingBox;
    const x = box.xMin * canvas.width;
    const y = box.yMin * canvas.height;
    const w = box.width * canvas.width;
    const h = box.height * canvas.height;

    // 截取人脸 ROI
    const face = ctx.getImageData(x, y, w, h);
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = w;
    tmpCanvas.height = h;
    tmpCanvas.getContext('2d').putImageData(face, 0, 0);
    const base64 = tmpCanvas.toDataURL('image/jpeg').split(',')[1];

    // 发送给后端
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ image: base64 }));
    }
  }
});

// 启动摄像头
const camera = new Camera(video, {
  onFrame: async () => {
    await faceDetection.send({ image: video });
  },
  width: 640,
  height: 480
});
camera.start();
