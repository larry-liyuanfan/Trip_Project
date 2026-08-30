# 阿里云 GPU ECS：Qwen3-VL 部署

本文仅适用于 `trip-gpu-a10` 的单卡 Qwen3-VL 推理环境。业务 FastAPI、Milvus、etcd、MinIO 和百炼 `qwen3.7-plus` 继续由原业务服务器负责，不在本机启动或迁移。

## 已验证基线

- Ubuntu 24.04；NVIDIA A10 24 GiB。
- 数据盘使用 ext4 并通过 UUID 挂载到 `/data`。
- Docker 已注册 NVIDIA runtime，官方 CUDA 12.8 容器可访问 GPU。
- Qwen3-VL 官方要求 vLLM 0.11.0 或以上；本配置固定 `vllm/vllm-openai:v0.11.0`，不复用历史 v0.8.5/Qwen2-VL 配置。

## 目录与配置

模型和运行产物必须保存在数据盘：

```text
/data/Trip_Project
/data/huggingface
/data/models
/data/experiments
/data/outputs
```

首次启动前创建本机环境文件：

```bash
cd /data/Trip_Project
cp docker/gpu/.env.example docker/gpu/.env
docker compose --env-file docker/gpu/.env \
  -f docker/gpu/docker-compose.yml config
```

`.env` 不得提交或传回本地。初始配置只允许 `Qwen/Qwen3-VL-2B-Instruct`，缓存目录为 `/data/huggingface`，宿主机端口只绑定 `127.0.0.1:8001`。Compose 显式禁用 `hf_transfer` 和 Xet，使用标准 Hugging Face 下载器，以便网络中断后复用数据盘缓存。

## 启动与验证

```bash
cd /data/Trip_Project
bash scripts/deploy_gpu_vllm.sh up
bash scripts/deploy_gpu_vllm.sh status
curl --fail http://127.0.0.1:8001/health
curl --fail http://127.0.0.1:8001/v1/models
```

下载模型期间可查看日志：

```bash
bash scripts/deploy_gpu_vllm.sh logs
```

启动脚本优先使用当前用户的 Docker socket；若 ECS 镜像只允许管理员访问，则在 `sudo -n docker info` 可用时自动使用免密 sudo，不修改 Docker 用户组。

本地访问必须使用 SSH 隧道，不得开放公网 8001：

```powershell
ssh -i .\secrets\trip-project-key.pem `
  -L 18001:127.0.0.1:8001 `
  ecs-user@<GPU_ECS_PUBLIC_IP>
```

另一个本地终端验证：

```powershell
curl http://127.0.0.1:18001/v1/models
```

停止服务不会删除模型缓存：

```bash
bash scripts/deploy_gpu_vllm.sh down
```

禁止在此流程中下载 7B、启动训练或全量评测、运行 `scripts/deploy_aliyun.sh`，以及启动 `docker/aliyun/docker-compose.yml`。

## 2026-08-09 首次验证结果

- 部署基线：Git `4bee591`，GPU 专用配置作为未提交叠加文件部署。
- 镜像：`vllm/vllm-openai:v0.11.0`；模型：`Qwen/Qwen3-VL-2B-Instruct`。
- `/health` 与 `/v1/models` 均返回 HTTP 200；容器端口为 `127.0.0.1:8001->8000/tcp`。
- 权重下载约 347.37 秒，模型加载占用约 4.237 GiB；初始化后静态显存约 14.6 GiB。
- `data/samples/images/cafe_001.jpg` 的单次多模态请求返回 HTTP 200，耗时 9,749 ms，采样到的显存峰值为 14,617 MiB。输出描述了图中深棕色八边形与浅色背景，和样本可见内容一致。
- 首次下载因镜像默认 `hf_transfer` 失败；禁用该下载器后复用缓存并成功。

以上仅为服务与多模态链路验证，不是完整模型效果评测。

## 2026-08-09 首次验证结果

- 部署基线：Git `4bee591`，GPU 专用配置作为未提交叠加文件部署。
- 镜像：`vllm/vllm-openai:v0.11.0`；模型：`Qwen/Qwen3-VL-2B-Instruct`。
- `/health` 与 `/v1/models` 均返回 HTTP 200；容器端口为 `127.0.0.1:8001->8000/tcp`。
- 权重下载约 347.37 秒，模型加载占用约 4.237 GiB；初始化后静态显存约 14.6 GiB。
- `data/samples/images/cafe_001.jpg` 的单次多模态请求返回 HTTP 200，耗时 9,749 ms，采样到的显存峰值为 14,617 MiB。输出描述了图中深棕色八边形与浅色背景，和样本可见内容一致。
- 首次下载因镜像默认 `hf_transfer` 失败；禁用该下载器后复用缓存并成功。

以上仅为服务与多模态链路验证，不是完整模型效果评测。
