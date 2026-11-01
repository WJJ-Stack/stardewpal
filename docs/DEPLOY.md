# 部署到魔搭创空间（Docker 类型）

魔搭（ModelScope）创空间提供免费 CPU 实例（2 核 16G），
支持 Docker 部署，且**国内访问稳定**——这一点很关键，
实测 Vercel / Render / Koyeb / Fly.io 的域名在此网络环境下均无法访问。

## 前置条件

- [x] 魔搭账号，已绑定阿里云并完成实名认证（Docker 类型强制要求）
- [ ] 一个模型 API Key（智谱 / 通义 / 硅基流动，任选；不填也能跑但回答质量下降）

## 关键约束

| 约束 | 说明 |
|---|---|
| 端口 | **必须监听 `0.0.0.0:7860`**，不能用 8080（平台占用） |
| 分支 | 默认分支必须是 `master` |
| 强制推送 | 禁止 `git push -f` |
| 单文件大小 | 超过 100MB 需 Git LFS（本项目的索引 19.5MB + 模型 22.9MB 均未超） |
| 持久化目录 | `/mnt/workspace`，重启后其他路径的数据会丢失 |
| 请求头 | 禁止使用 `Authorization`、`X-modelscope-*`、`X-studio-*` |

## 步骤

### 1. 创建创空间

1. 打开 https://modelscope.cn/studios
2. 点击「新建创空间」
3. 填写：
   - 空间名称：`stardewpal`
   - 许可证：选 `CC-BY-NC-SA-4.0`（语料来自 BWiki，需同协议）
   - SDK 类型：**Docker**
   - 硬件：**免费 CPU**（2vCPU / 16GB）
   - 可见性：公开
4. 创建后得到仓库地址，形如：
   ```
   https://www.modelscope.cn/studios/<你的用户名>/stardewpal.git
   ```

### 2. 推送代码

魔搭的仓库是独立的，与 GitHub 那个不冲突。两种方式：

**方式 A：直接推送到魔搭**

```bash
cd E:\rag
git remote add modelscope https://www.modelscope.cn/studios/<用户名>/stardewpal.git
git push modelscope master
```

首次推送会要求输入用户名和访问令牌（在魔搭「个人设置 → 访问令牌」生成）。

**方式 B：从 GitHub 导入**

创空间创建页支持「关联 GitHub 仓库」，
但它要求仓库是 public（本项目是）。不过魔搭对 GitHub 的拉取可能较慢，
**推荐方式 A**。

### 3. 配置环境变量（密钥）

创空间页面 →「设置」→「环境变量」，添加：

```
LLM_PROVIDER=zhipu
LLM_API_KEY=<你的智谱 key>
```

⚠️ **不要把 key 写进代码或 `.env` 提交**。`.env` 已在 `.gitignore` 中，
容器读取的是平台注入的环境变量。

### 4. 触发构建

推送后会自动开始构建。首次约 3~5 分钟。

构建日志在创空间页面的「日志」标签页。

### 5. 验证

构建完成后，访问创空间主页，应看到聊天界面。

健康检查：
```
https://www.modelscope.cn/studios/<用户名>/stardewpal/health
```

或直接在页面里问「防风草卖多少钱」测试。

## 常见问题

### 构建失败：端口错误

```
Error: container is not listening on 0.0.0.0:7860
```

检查 `Dockerfile` 最后一行的 `--port 7860`，以及 `serve.py` 的默认端口。

### 构建失败：pip 安装超时

魔搭构建机在国内，访问 PyPI 官方源可能慢。`Dockerfile` 里可以加：

```dockerfile
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 运行时崩溃：找不到图标

`icons.zip` 解包失败。检查 `Dockerfile` 里的这行是否执行成功：

```dockerfile
RUN python -m backend.pipeline.bundle_assets --unpack
```

### 免费实例休眠

免费 CPU 实例一段时间无访问后会休眠，首次访问需要唤醒（约 20~30 秒）。
**面试演示前先访问一次预热**。

### 数据丢失

除 `/mnt/workspace` 外，容器重启后数据都会重置。
本项目索引是只读的，不会丢数据；但如果将来加了用户上传功能，
需要把写入目录指向 `/mnt/workspace`。

## 资源占用参考

| 项 | 值 |
|---|---|
| 镜像体积 | 约 600MB（Python 3.12 slim + onnxruntime + numpy/scipy） |
| 常驻内存 | 约 300MB（模型 23MB + 索引 20MB + Python 运行时） |
| 冷启动 | 约 25 秒 |
| 单次问答 | 3~12 秒（取决于模型服务响应） |

免费 CPU（2 核 16G）完全够用，甚至有过剩。

## 本地验证（可选）

如果本地装了 Docker Desktop：

```bash
docker build -t stardewpal .
docker run --rm -p 7860:7860 \
  -e LLM_PROVIDER=zhipu \
  -e LLM_API_KEY=<你的 key> \
  stardewpal
```

打开 http://127.0.0.1:7860。
