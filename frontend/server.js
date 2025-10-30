const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 8080;

// 启用CORS，允许前端访问后端API
app.use(cors({
  origin: '*', // 在生产环境中应该设置为特定的域名
  methods: ['GET', 'POST', 'OPTIONS']
}));

// 托管静态文件
app.use(express.static(__dirname));

// 配置路由，所有请求都返回index.html（支持SPA路由）
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// 启动服务器
app.listen(PORT, () => {
  console.log(`前端服务器已启动，运行在 http://localhost:${PORT}`);
});