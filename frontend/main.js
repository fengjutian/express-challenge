const video = document.getElementById('video');
const canvas = document.getElementById('output');
const ctx = canvas.getContext('2d');
const resultDiv = document.getElementById('result');

// WebSocket connection management
let socket = null;
let reconnectAttempt = 0;
const maxReconnectAttempts = 5;
const baseReconnectDelay = 1000; // Start with 1 second

// Rate-limiting configuration (ms) — at most one send per sendInterval
let lastSendTime = 0;
const sendInterval = 200; // 200 ms -> ~5 FPS
let lastEmotion = '';

function connectWebSocket() {
  if (socket && (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN)) {
    return; // Already connecting or connected
  }

  try {
    socket = new WebSocket("ws://localhost:8000/ws/emotion");

    socket.onopen = () => {
      console.log('✅ WebSocket 已连接 (client)');
      reconnectAttempt = 0; // Reset attempt counter on successful connection
      resultDiv.style.color = ''; // Reset any error styling
    };

    socket.onclose = (ev) => {
      console.warn('❌ WebSocket closed (client):', ev.code, ev.reason);
      
      if (reconnectAttempt < maxReconnectAttempts) {
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempt), 10000); // Exponential backoff, max 10s
        console.log(`Attempting to reconnect in ${delay/1000}s... (attempt ${reconnectAttempt + 1}/${maxReconnectAttempts})`);
        resultDiv.innerHTML = `<div style="color: orange;">正在重新连接服务器... (${reconnectAttempt + 1}/${maxReconnectAttempts})</div>`;
        setTimeout(connectWebSocket, delay);
        reconnectAttempt++;
      } else {
        console.error('Max reconnection attempts reached');
        resultDiv.innerHTML = `
          <div style="color: red;">无法连接到服务器</div>
          <div style="margin-top: 10px;">请确保后端服务器正在运行 (localhost:8000)</div>
        `;
      }
    };

    socket.onerror = (err) => {
      console.error('WebSocket error (client):', err);
      resultDiv.style.color = 'red';
    };

    socket.onmessage = (event) => {
      console.log("Message from server: ", event.data);
      try {
        const data = JSON.parse(event.data);
        console.log("Parsed data:", data);
        if (data.error) {
          resultDiv.innerHTML = `<div style="color: orange;">处理错误: ${data.error}</div>`;
        } else {
          lastEmotion = `${data.emotion} (${(data.confidence * 100).toFixed(1)}%)`;
          resultDiv.innerHTML = `Emotion: ${lastEmotion}`;
          console.log(123, lastEmotion);
        }
      } catch (err) {
        console.warn('Error parsing server message:', err);
      }
    };
  } catch (err) {
    console.error('Error creating WebSocket:', err);
    resultDiv.innerHTML = `<div style="color: red;">连接错误: ${err.message}</div>`;
  }
}

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
        minDetectionConfidence: 0.3
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
function setupFaceDetectionCallback() {
  if (!faceDetector) {
    console.error('❌ 无法设置人脸检测回调: faceDetector 未初始化');
    return;
  }
  
  faceDetector.onResults((results) => {
  // Draw the frame onto canvas
  ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

  if (lastEmotion) {
    ctx.fillStyle = 'white';
    ctx.font = '30px Arial';
    ctx.fillText(lastEmotion, 10, 40);
  }

  if (results.detections && results.detections.length > 0) {
    console.log("Face detected!", results.detections);
    const box = results.detections[0].boundingBox;

    // Compute raw coordinates
    const rawX = box.width * canvas.width;
    const rawY = box.height * canvas.height;
    const rawW = box.width * canvas.width;
    const rawH = box.height * canvas.height;

    // Ensure values are finite numbers
    console.log('Raw coordinates:', rawX, rawY, rawW, rawH);
    if (![rawX, rawY, rawW, rawH].every(Number.isFinite)) {
      // invalid values, skip this frame
      return;
    }

    // Round and clamp to canvas bounds
    const x = Math.max(0, Math.floor(rawX));
    const y = Math.max(0, Math.floor(rawY));
    // ensure width/height at least 1 and don't overflow canvas
    const w = Math.max(1, Math.min(canvas.width - x, Math.floor(rawW)));
    const h = Math.max(1, Math.min(canvas.height - y, Math.floor(rawH)));

    // If ROI is degenerate, skip
    console.log("Raw coordinates:");
    if (w <= 0 || h <= 0) return;

    try {
      // 截取人脸 ROI — getImageData requires integer 'long' parameters
      const face = ctx.getImageData(x, y, w, h);
      const tmpCanvas = document.createElement('canvas');
      tmpCanvas.width = w;
      tmpCanvas.height = h;
      tmpCanvas.getContext('2d').putImageData(face, 0, 0);
      const base64 = tmpCanvas.toDataURL('image/jpeg').split(',')[1];

      console.log("Sending face data to backend...");

      // 发送给后端 (限速，避免过多并发消息导致连接不稳定)
      if (socket && socket.readyState === WebSocket.OPEN) {
        const now = Date.now();
        // lastSendTime & sendInterval are defined at module scope
        if ((now - lastSendTime) >= sendInterval) {
          try {
            console.log("Sending face data to backend...");
            socket.send(JSON.stringify({ image: base64 }));
            lastSendTime = now;
          } catch (err) {
            console.warn('Error sending frame:', err);
            // Force reconnect on send error
            socket.close();
            connectWebSocket();
          }
        }
      } else {
        console.log("WebSocket not open. Ready state: " + (socket ? socket.readyState : 'null'));
        if (!socket || socket.readyState === WebSocket.CLOSED) {
          // Try to reconnect if socket is null or closed
          connectWebSocket();
        }
      }
    } catch (err) {
      // If getImageData still throws (e.g., cross-origin or invalid region), skip gracefully
      console.warn('Skipping frame due to getImageData error:', err);
    }
  } else {
    console.log("No face detected.");
  }

});
}

// 启动摄像头或提供备用方案
async function initializeCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ 
      video: { width: 640, height: 480 },
      audio: false 
    });
    
    // Hide error container on success
    document.getElementById('error-container').style.display = 'none';

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
    const errorContainer = document.getElementById('error-container');
    const errorTitle = document.getElementById('error-title');
    const errorMessage = document.getElementById('error-message');

    errorTitle.textContent = '无法访问摄像头';
    errorMessage.textContent = '请确保摄像头已连接，未被其他应用（如视频会议软件）占用，并已授予浏览器摄像头权限。';
    errorContainer.style.display = 'block';
  }
}

// 初始化应用
function initializeApp() {
  console.log('🔄 开始初始化应用...');
  
  // 1. 首先初始化人脸检测
  const faceDetectionReady = initializeFaceDetection();
  
  if (faceDetectionReady) {
    // 2. 设置人脸检测回调
    setupFaceDetectionCallback();
    
    // 3. 连接WebSocket
    connectWebSocket();
    
    // 4. 初始化摄像头
    initializeCamera();
  } else {
    resultDiv.innerHTML = `
      <div style="color: red;">人脸检测组件初始化失败</div>
      <div style="margin-top: 10px;">请检查 MediaPipe 脚本是否正确加载</div>
    `;
  }
}

// 当页面加载完成后初始化应用
window.addEventListener('DOMContentLoaded', initializeApp);
