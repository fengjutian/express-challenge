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

// 初始化 MediaPipe - 使用正确的全局命名空间和配置
let faceDetector;

function initializeFaceDetection() {
  try {
    // 使用正确的命名空间 - 检查全局对象
    if (window.FaceDetection) {
      faceDetector = new window.FaceDetection({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`,
      });
      
      faceDetector.setOptions({
        model: 'short',   // 快速模型
        minDetectionConfidence: 0.5
      });
      
      console.log('✅ FaceDetection 初始化成功');
      return true;
    } else if (window.MP_FaceDetection) {
      // 备用命名空间
      faceDetector = new window.MP_FaceDetection.FaceDetection({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`,
      });
      
      faceDetector.setOptions({
        model: 'short',   // 快速模型
        minDetectionConfidence: 0.5
      });
      
      console.log('✅ FaceDetection 初始化成功（备用命名空间）');
      return true;
    } else {
      console.error('❌ FaceDetection 未找到');
      return false;
    }
  } catch (error) {
    console.error('❌ FaceDetection 初始化失败:', error);
    return false;
  }
}

// 检测到人脸时
faceDetector.onResults((results) => {
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

// 启动摄像头或提供备用方案
async function initializeCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ 
      video: { width: 640, height: 480 },
      audio: false 
    });
    
    // 设置视频源
    video.srcObject = stream;
    
    // 当视频准备好后开始检测
    video.onloadedmetadata = () => {
      // 使用requestAnimationFrame代替Camera类以避免依赖问题
      const processVideo = async () => {
        if (!video.paused && !video.ended) {
          await faceDetector.send({ image: video });
        }
        requestAnimationFrame(processVideo);
      };
      processVideo();
    };
    
    console.log('✅ 摄像头访问成功');
  } catch (error) {
    console.error('❌ 摄像头访问失败:', error);
    resultDiv.innerHTML = `
      <div style="color: red;">无法访问摄像头: ${error.message}</div>
      <div style="margin-top: 10px;">请确保摄像头已连接且未被其他应用占用</div>
    `;
  }
}

// 初始化应用
initializeCamera();
