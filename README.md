# yocr

基于 **YOLO + PaddleOCR** 的视觉识别服务，面向 Android / AAOS 自动化测试场景。
FastAPI 提供 REST API，接收截图后按参数选择模型，返回 UI 元素坐标、类别与文字内容。

## 特性

- **多模型加载**：内置 `android_ui_detection_yolov8`（本地 `.pt`）与 `ScreenParser`（HuggingFace: docling-project）两个模型，支持通过环境变量注册更多别名，全部懒加载 + 缓存
- **元素检测**：返回 bbox（xyxy/xywh/中心点）、类别名、置信度
- **文字识别**：PaddleOCR（PP-OCRv5 mobile 模型，x86 CPU 上低延迟；可切换 server 模型提升精度）
- **图文融合**：`/analyze` 将 OCR 文本行按"包含关系"挂到最小的 UI 元素上，直接得到"哪个控件上有什么字"
- **低延迟**：模型常驻内存、oneDNN 加速（不兼容时自动降级重试）、首次调用后 OCR ~0.7s / 检测 ~2.7s（CPU 实测）
- **双输入格式**：multipart 文件上传 或 JSON `image_base64`，也接受原始二进制 body

> 📖 **实战教程（中文）**：[docs/tutorial.zh-CN.md](docs/tutorial.zh-CN.md) —— 覆盖有文字控件、弹窗按钮（有字/无字 X）、纯图标、列表多候选、开关/输入框、等待元素出现等场景，附可直接复用的客户端工具类 [`examples/yocr_client.py`](examples/yocr_client.py)。

## 快速开始

### 本地运行 (uv)

```bash
# 依赖: Python 3.12+, uv
uv sync                      # 安装依赖
uv run yocr serve --port 8000
```

Windows 上同样适用（需先装 Python/uv）；若使用 GPU 推理设置 `YOCR_DEVICE=cuda:0`。

### Docker

```bash
mkdir -p models   # 放入 android_ui_detection_yolov8.pt（可选）
docker compose up -d --build
curl http://127.0.0.1:8000/api/v1/healthz
```

## 模型配置

| 名称 | 来源 | 说明 |
|---|---|---|
| `android_ui_detection_yolov8` | 本地文件 | 把 `.pt` 权重放到 `./models/android_ui_detection_yolov8.pt` |
| `ScreenParser` | HuggingFace 自动下载 | `docling-project/ScreenParser`，55 类 UI 元素 |

自定义别名：

```bash
YOCR_MODEL_ALIASES="ui=android_ui_detection_yolov8.pt,mydet=org/repo/best.pt"
```

请求时用 `?model=<别名>` 选择；缺省为 `android_ui_detection_yolov8`。

## API

前缀 `/api/v1`，交互式文档见 `/docs`。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/healthz` | 健康检查、已注册模型列表 |
| GET | `/models` | 模型详情（是否已加载、类别表） |
| POST | `/detect` | YOLO 元素检测 |
| POST | `/ocr` | PaddleOCR 文字识别 |
| POST | `/analyze` | 检测 + OCR + 文本归属合并（推荐） |

### POST /detect

```
curl -F "file=@screen.png" \
     "http://127.0.0.1:8000/api/v1/detect?model=screenparser&conf=0.3"
```

Query 参数：`model`（别名）、`conf`、`iou`、`imgsz`（推理分辨率，默认 1280）。

响应：

```json
{
  "model": "ScreenParser",
  "image": {"width": 1080, "height": 600},
  "elements": [
    {
      "id": 0, "label": "Text", "class_id": 49, "confidence": 0.4479,
      "box": {"xyxy": [659,301,957,348], "xywh": [659,301,298,47], "center": [808,324]},
      "text": null, "text_confidence": null
    }
  ],
  "timing": {"total_ms": 2692.1, "detect_ms": 2689.0, "ocr_ms": null}
}
```

### POST /ocr

```bash
curl -F "file=@screen.png" http://127.0.0.1:8000/api/v1/ocr
# 或 JSON:
curl -X POST http://127.0.0.1:8000/api/v1/ocr \
     -H 'content-type: application/json' \
     -d '{"image_base64": "<...>"}'
```

响应包含逐行文本 + 位置框 + `full_text`。

### POST /analyze

检测 + OCR 一次完成，并把每条 OCR 文本归入包含其中心的**最小**元素（`text` / `text_confidence` 字段）：

```bash
curl -F "file=@screen.png" \
     "http://127.0.0.1:8000/api/v1/analyze?model=screenparser&conf=0.25"
```

```json
{
  "model": "ScreenParser",
  "elements": [
    {"label": "Link", "confidence": 0.3032,
     "box": {"xyxy": [60,200,501,281], "center": [280,240]},
     "text": "Bluetooth", "text_confidence": 1.0}
  ],
  "lines": [...],
  "full_text": "Settings\nBluetooth\nAAOS Test 123",
  "timing": {"total_ms": 3823.5, "detect_ms": 2689.0, "ocr_ms": 1131.0}
}
```

自动化测试中即可用 `text` 匹配目标控件，取 `box.center` 作为目标坐标（yocr 只做图像分析，不做任何设备操作）。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `YOCR_HOST` / `YOCR_PORT` | 0.0.0.0 / 8000 | 监听地址 |
| `YOCR_MODELS_DIR` | `./models` | 本地 .pt 目录 |
| `YOCR_MODEL_ALIASES` | — | 额外别名 `name=path,name2=repo/file.pt` |
| `YOCR_PRELOAD_MODELS` | — | 启动即加载的模型，逗号分隔（如 `screenparser`） |
| `YOCR_PRELOAD_OCR` | `1` | 启动时初始化 PaddleOCR |
| `YOCR_DEVICE` | `cpu` | YOLO 设备：`cpu` / `cuda:0` |
| `YOCR_INFER_SIZE` | `1280` | YOLO 推理分辨率（越大越准越慢） |
| `YOCR_CONF` / `YOCR_IOU` | 0.25 / 0.45 | 默认置信度 / NMS 阈值 |
| `YOCR_OCR_LANG` | `ch` | OCR 语言 |
| `YOCR_OCR_DET_MODEL` / `YOCR_OCR_REC_MODEL` | PP-OCRv5_mobile_* | 换 `PP-OCRv5_server_*` 提精度 |
| `YOCR_OCR_MKLDNN` | `1` | oneDNN 加速开关（不兼容自动关闭） |
| `YOCR_LOG_LEVEL` | `INFO` | 日志级别 |

## 项目结构

```
src/yocr/
├── api.py          # REST 路由 (/detect /ocr /analyze /models /healthz)
├── app.py          # FastAPI 工厂 + lifespan 预热
├── cli.py          # yocr serve 命令
├── config.py       # YOCR_* 环境变量配置
├── detectors.py    # 多 YOLO 模型注册表（懒加载/HF下载/线程安全）
├── imaging.py      # 图像编解码（兼容被传输层损坏 \\r\\n 的 PNG）
├── ocr_engine.py   # PaddleOCR 封装（2.x/3.x 兼容 + oneDNN 降级）
├── pipeline.py     # 检测/OCR/文本归属分析流水线
└── schemas.py      # Pydantic 请求/响应模型
examples/
└── yocr_client.py  # 测试框架可复用的 Python 客户端工具类
docs/
└── tutorial.zh-CN.md  # 中文实战教程
tests/              # pytest 单元测试
Dockerfile
docker-compose.yml
```

## 测试

```bash
uv run pytest tests/ -q
```
