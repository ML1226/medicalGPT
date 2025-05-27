
#### 安装依赖
```shell
// 新建虚拟环境
conda create -n deepmed python=3.10

// 激活虚拟环境
conda activate deepmed

// 安装依赖
pip install flask flask_cors


// 进入项目目录
cd ChatMate
```

#### 启动与终止
```shell
// 启动服务
python web_server.py
python rag_server.py
```

成功运行后可在终端看到提示信息`程序已运行: http://127.0.0.1:8080`，表示前端服务已启动成功。

> 注意：
> 1. 默认将前端运行在 `localhost:8080`，后端运行在 `localhost:8000`，可以在 `web_server.py` 和 `rag_server.py` 中分别修改前、后端的运行端口。
> 2. 终止运行项目一定要释放端口占用。
> 3. 如果修改后端的运行端口，需要在 `static/scripts.js` 文件的第 3 行同步修改 `SERVER_PORT` 变量的值。

### 使用方式
1. **API 设置**：点击页面侧边栏第一个按钮，输入 API 地址、 API 密钥和模型名称，点击保存按钮即可完成设置。
2. **开始对话**：和 Deepseek 怎么聊天的就和 ChatMate 聊天，输入消息，点击发送按钮或回车即可。

### 运行示例 & 功能说明
1. **API 设置**：点击页面侧边栏第一个按钮，输入 API 密钥和模型名称，点击保存按钮即可完成设置。默认使用 OpenAI API 调用模型，如果运行本地模型，请在 `API 地址` 栏输入 对应的地址，并忽略 `API 密钥`。
    <img src="./media/api-setting.png" alt="API 设置" width="720">
2. **会话切换**：点击页面侧边栏第二个按钮，可切换当前会话或开启新的会话。
   <img src="./media/session-switch.png" alt="API 设置" width="720">
3. **清空对话上下文**：点击页面侧边栏第三个按钮，可清空当前会话中的对话历史。
   <img src="./media/clear.png" alt="API 设置" width="720">
4. **流式输出**：白色气泡中为最终响应，采用流式输出。
