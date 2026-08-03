# 阿里云部署

项目部署在新加坡 ECS 上，业务推理由阿里云百炼 `qwen3.7-plus` 提供，
通过业务空间专属 OpenAI 兼容端点调用。ECS 不运行本地 vLLM。

## 安全边界

- API Key 仅保存在 `secrets/dashscope_api_key`，通过 Docker secret 挂载。
- `docker/aliyun/.env`、`docker/milvus/.env` 和 `secrets/` 不进入 Git 或镜像。
- FastAPI 默认只监听 ECS 的 `127.0.0.1:8000`，不直接暴露未认证接口。
- Milvus 和健康检查端口同样只监听回环地址。

需要从本机访问 API 时建立 SSH 隧道：

```bash
ssh -L 8000:127.0.0.1:8000 root@<ecs-public-ip>
curl http://127.0.0.1:8000/health
```

## ECS 启动

在项目根目录准备三个本地文件：

- `secrets/dashscope_api_key`
- `docker/aliyun/.env`
- `docker/milvus/.env`

然后运行：

```bash
bash scripts/deploy_aliyun.sh
```

部署脚本先校验两个 Compose 配置，再启动 Milvus、etcd、MinIO 和 API，
最后等待 API 健康检查通过。百炼请求失败时云端配置禁止静默回退，避免把
确定性本地结果误报为真实模型结果。

## 模型配置

当前配置为：

- model: `qwen3.7-plus`
- region: `ap-southeast-1`
- protocol: OpenAI-compatible chat completions
- thinking: disabled

关闭 thinking 是为了保持现有结构化输出链路稳定。业务空间地址和 API Key
必须来自同一地域，不能跨地域混用。
