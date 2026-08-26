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
- **素材库 + 快速定位**：注册素材（id / 图片 / 描述）后，上传期望截图即可与全部素材逐一比对，毫秒级返回"包含哪些素材 + 位置框"，内置 Vue 管理界面

> 📖 **实战教程（中文）**：[docs/tutorial.zh-CN.md](docs/tutorial.zh-CN.md) —— 覆盖有文字控件、弹窗按钮（有字/无字 X）、纯图标、列表多候选、开关/输入框、等待元素出现等场景，附可直接复用的客户端工具类 [`examples/yocr_client.py`](examples/yocr_client.py)。

## 快速开始

```bash
make help        # 查看全部任务
```

### 本地运行 (uv)

```bash
# 依赖: Python 3.12+, uv
uv sync                      # 安装依赖
uv run yocr serve --port 8000
# 或: make run PORT=8000
```

Windows 上同样适用（需先装 Python/uv）；若使用 GPU 推理设置 `YOCR_DEVICE=cuda:0`。

### Docker

```bash
mkdir -p models   # 放入 android_ui_detection_yolov8.pt（可选）
docker compose up -d --build      # 或: make docker-up
curl http://127.0.0.1:8000/api/v1/healthz
```

常用：`make docker-logs`（日志）、`make docker-restart`、`make docker-down`

### systemd（Linux 常驻 + 开机自启）

以当前仓库目录 + `.venv` 方式常驻，服务以当前用户身份运行。
**一条命令完成依赖安装 + 模型下载（uvx hf）+ 服务安装**：

```bash
make systemd-install    # 依赖 + 前端构建 + uvx hf 下载模型到 .cache/ + 渲染安装 unit(需输 sudo 密码)
make systemd-start      # 启动并设置开机自启
make health             # 探活 http://127.0.0.1:8000/api/v1/healthz -> "ocr_loaded": true
make systemd-logs       # journalctl -u yocr -f
make systemd-stop       # 停止并取消自启
make systemd-uninstall  # 卸载 unit
```

启动后打开 `http://<host>:8000/` 即是 Vue 素材管理界面（`make systemd-install`
会自动构建前端；服务器没有 npm 时沿用已有 `frontend/dist`，没有构建产物则仅提供 API）。
素材库数据落在 `<repo>/data/sucai`（unit 内 `YOCR_SUCAI_DIR`），随服务持久化。

常用变量（均可覆盖）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `CACHE_DIR` | `./.cache` | 模型缓存根目录（HF_HOME / paddlex 缓存都放这） |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 下载源；置空则直连官方 |
| `PADDLE_OCR_MODELS` | `PP-OCRv5_mobile_det PP-OCRv5_mobile_rec` | 要预下载的 OCR 模型，可换 server 版提精度 |
| `HF_OFFLINE` | `1` | 预下载后离线运行（只读缓存） |
| `SKIP_MODELS` | `0` | 置 1 跳过下载（如已手工拷贝缓存） |
| `PORT` / `DEVICE` | `8000` / `cpu` | 监听端口 / 推理设备 |

示例：

```bash
# GPU + 高精度 OCR 模型
make systemd-install DEVICE=cuda:0 PADDLE_OCR_MODELS="PP-OCRv5_server_det PP-OCRv5_server_rec"
# 已有缓存，只重装服务
make systemd-install SKIP_MODELS=1
```

单独补下模型：`make models-download`；额外环境变量写入 `/etc/yocr/yocr.env`
（优先级高于 unit 内置值），例如换缓存目录时无需重装服务。

### 离线部署：用 `uvx hf` 预下载模型

生产服务器无法访问 HuggingFace 时，启动会报
`No available model hosting platforms detected...`（PaddleX 找不到模型源）。
解决方法：在**任意有网的机器**上用 `hf` CLI 把模型下载到指定目录布局，再整包拷到服务器。

```bash
export HF_ENDPOINT=https://hf-mirror.com        # 国内加速；海外可去掉
CACHE=/data/yocr-cache                           # 缓存根目录（自定）

# ① ScreenParser 权重 → 标准 HF 缓存布局
HF_HOME=$CACHE/huggingface \
uvx --from "huggingface_hub[cli]" hf download docling-project/ScreenParser best.pt

# ②③ PaddleOCR det/rec 模型 → 必须 按 paddlex 的目录约定摆放(--local-dir)
for m in PP-OCRv5_mobile_det PP-OCRv5_mobile_rec; do
  uvx --from "huggingface_hub[cli]" hf download "PaddlePaddle/$m" \
      --local-dir "$CACHE/.paddlex/official_models/$m"
done
```

> 注意 ②③ 必须带 `--local-dir ".../official_models/<模型名>"`，
> paddlex 只认这个路径布局；直接下进 HF 缓存它看不到。

把 `$CACHE` 整包 scp 到服务器后，配置环境变量并重启：

```bash
sudo tee /etc/yocr/yocr.env <<'EOF'
HF_HOME=/data/yocr-cache/huggingface
PADDLE_PDX_CACHE_HOME=/data/yocr-cache/.paddlex
HF_HUB_OFFLINE=1
EOF
make systemd-restart && make health    # 应输出 "ocr_loaded": true
```

- `PADDLE_PDX_CACHE_HOME`：PaddleX 查找 OCR det/rec 模型的位置
- `HF_HOME`：ScreenParser 等走 `hf_hub_download` 的权重位置
- `HF_HUB_OFFLINE=1`：只读缓存、不再联网尝试（避免启动卡超时）
- 服务默认**不联网下载**：权重缺失会直接报错并提示执行 `make models-download`；
  临时恢复运行时自动下载可设 `YOCR_ALLOW_DOWNLOAD=1`

默认 UI 检测模型 `android_ui_detection_yolov8` 自动从 HuggingFace 下载
（`yasirfaizahmed/android_ui_detection_yolov8`，Apache-2.0）；`make models-download`
会把 `best.pt` 预下载进 `HF_HOME` 缓存供离线使用。把自己的 `.pt` 放进服务器
`models/android_ui_detection_yolov8.pt` 即可覆盖（本地文件优先）。
想提精度可换成 server 版：循环里改为 `PP-OCRv5_server_det` / `PP-OCRv5_server_rec`，
并在 yocr.env 加 `YOCR_OCR_DET_MODEL` / `YOCR_OCR_REC_MODEL` 对应项。

#### hf 下载报 401？

两个已验证的原因（Makefile 已内置第一种的修复）：

| 原因 | 现象 | 解决 |
|---|---|---|
| HF 新 Xet 存储后端无法被镜像代理（`cas-server.xethub.hf.co ... 401`） | 走 `HF_ENDPOINT=hf-mirror.com` 时偶发/必现 401 | 加 `HF_HUB_DISABLE_XET=1` 走传统 HTTP 通道（models-download 已自动带上） |
| 环境残留无效凭证 | `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` 过期或写错 | `unset HF_TOKEN HUGGING_FACE_HUB_TOKEN`；公共仓库匿名即可下载 |

手动命令示例：

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
uvx --from "huggingface_hub[cli]" hf download PaddlePaddle/PP-OCRv5_mobile_det
```

## 模型配置

| 名称 | 来源 | 说明 |
|---|---|---|
| `android_ui_detection_yolov8` | `models/` 本地文件或 HF 缓存 | 放 `.pt` 到 `./models/android_ui_detection_yolov8.pt` 可覆盖；缓存源 `yasirfaizahmed/android_ui_detection_yolov8` |
| `ScreenParser` | HF 缓存（`make models-download` 预下载） | `docling-project/ScreenParser`，55 类 UI 元素 |
| `IconFinder` | `models/yolov8s-worldv2.pt`（`make models-download` 预下载） | YOLO-World 开放词汇模型，按**文字**找具体图标（设置齿轮/wifi/蓝牙/返回箭头…）；词表用 `YOCR_ICON_CLASSES` 自定义（逗号分隔） |

自定义别名：

```bash
YOCR_MODEL_ALIASES="ui=android_ui_detection_yolov8.pt,mydet=org/repo/best.pt"
```

请求时用 `?model=<别名>` 选择；缺省为 `android_ui_detection_yolov8`。

### IconFinder：按文字找具体图标（设置齿轮等）

通用 UI 检测器只会说 "App Icon"；`IconFinder`（YOLO-World 开放词汇）能直接认出
具体是哪个图标。内置 20 个常见类别（settings 齿轮、wifi、蓝牙、返回箭头、home、
相机、搜索、菜单、删除…）。权重由 `make models-download` 下载到 `models/`，
并顺带预热 CLIP 文本编码器缓存；未预热时首次加载需联网一次。

```bash
# 找设置齿轮图标，直接点击坐标
curl -F "file=@screen.png" \
     "http://127.0.0.1:8000/api/v1/detect?model=iconfinder&label=gear&conf=0.15"

# 自定义词表后重启
YOCR_ICON_CLASSES="a gear settings icon, a wifi icon, a download icon" \
    make systemd-restart
```

客户端用法：`c.find_icon(png, labels=["gear settings icon"])` 或
`c.locate(png, model="iconfinder", label="gear")`。
开放词汇检测建议把 `conf` 降到 0.1~0.2 再按置信度筛选。

## API

前缀 `/api/v1`，交互式文档见 `/docs`。

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/healthz` | 健康检查、已注册模型列表 |
| GET | `/models` | 模型详情（是否已加载、类别表） |
| POST | `/detect` | YOLO 元素检测（支持 `text`/`label`/`q` 目标参数，返回 `found`/`matched`） |
| POST | `/ocr` | PaddleOCR 文字识别 |
| POST | `/analyze` | 检测 + OCR + 文本归属合并（推荐，同支持 `found`/`matched`） |
| POST | `/match` | 模板匹配：在场景图(file)中定位模板图(template)，返回 found/score/box/scale |
| GET/POST | `/sucai` | 素材库列表 / 注册素材（`file` + 可选 `id`、`describe`） |
| GET/PUT/DELETE | `/sucai/{id}` | 素材详情 / 更新描述与图片 / 删除 |
| GET | `/sucai/{id}/image` | 素材图片 (PNG) |
| POST | `/sucai/find` | 场景图与全部素材比对，按 score 降序返回命中项及定位框 |

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

### POST /match：两张图模板定位

上传**模板小图**与**场景大图**，返回第一张在第二张中的位置（多尺度灰度 NCC，
毫秒级、无需模型、跨 DPI 自动缩放尝试）：

```bash
curl -F "file=@screen.png" -F "template=@button.png" \
     "http://127.0.0.1:8000/api/v1/match?threshold=0.8"
```

```json
{
  "found": true, "score": 0.9731, "threshold": 0.8, "scale": 1.0,
  "box": {"xyxy": [100,60,140,90], "center": [120,75]},
  "image": {"width": 1080, "height": 600}, "template": {"width": 40, "height": 30},
  "timing": {"total_ms": 12.5}
}
```

也支持 JSON：`{"image_base64": "...", "template_base64": "..."}`。
客户端：`c.match(tpl_png, scene_png)` → 强类型响应；`c.locate_template(...)`
直接返回可点击中心坐标。

### 素材库与快速定位（/sucai）

把按钮/图标等小图注册为**素材**（id + 图片 + describe），之后上传任意期望截图，
一次调用即可与全部素材比对并定位——适合"断言某界面包含某控件"的测试场景：

```bash
# 注册素材（id 省略则自动生成）
curl -F "file=@confirm_btn.png" -F "id=btn-confirm" -F "描述: 确认按钮" \
     http://127.0.0.1:8000/api/v1/sucai

# 期望截图 vs 全部素材：返回每个素材的 score / found / box / center（按 score 降序）
curl -F "file=@screen.png" "http://127.0.0.1:8000/api/v1/sucai/find?threshold=0.8"
```

```json
{
  "found_any": true, "sucai_count": 3, "threshold": 0.8,
  "results": [
    {"id": "btn-confirm", "describe": "确认按钮", "found": true, "score": 0.9812,
     "scale": 1.0, "box": {"xyxy": [812,540,902,585], "center": [857,562]},
     "hits": [{"score": 0.9812, "scale": 1.0, "box": {...}, "center": [...]}],
     "elapsed_ms": 4.2}
  ],
  "timing": {"total_ms": 13.8}
}
```

匹配精度：多尺度 NCC + **逐级尺度细化**（亚步长定位）+ **颜色校验门**
（同形状不同颜色的控件不再互相误报），`all_instances=true` 时用跨尺度 NMS
返回同一素材的**全部出现位置**（`hits` 数组，默认只返回最佳位置）。

素材持久化在 `YOCR_SUCAI_DIR`（默认 `./data/sucai`，meta.json + PNG），重启不丢、可整目录拷贝。

### 前端管理界面（Vue 3）

```bash
make frontend-install && make frontend-build   # 构建到 frontend/dist
make run                                        # 打开 http://127.0.0.1:8000/
```

- **素材库**：拖拽/选择图片 + id + 描述 → 注册；卡片列表支持预览与删除
- **查找定位**：上传或 **Ctrl+V 直接粘贴截图** → 与全部素材比对，画布上框出命中位置，
  结果列表按得分排序（未过阈值的也会显示最高得分，便于调阈值）；
  勾选"标记所有出现位置"可同时框出同一素材的每一次出现

前端开发模式（热更新，`/api` 自动代理到 8000）：`make frontend-dev`。
Docker 镜像已内置构建好的前端，无需额外步骤。

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
| `YOCR_PRELOAD_MODELS` | `all` | 启动即加载的模型：`all` 全部内置模型、空串关闭、或逗号列表（如 `screenparser,iconfinder`） |
| `YOCR_ALLOW_DOWNLOAD` | `0` | 是否允许服务运行时联网下载权重；默认关闭，统一用 `make models-download` 供给 |
| `YOCR_ICON_CLASSES` | 内置20类 | IconFinder 开放词汇类别表（逗号分隔） |
| `YOCR_PRELOAD_OCR` | `1` | 启动时初始化 PaddleOCR |
| `YOCR_DEVICE` | `cpu` | YOLO 设备：`cpu` / `cuda:0` |
| `YOCR_INFER_SIZE` | `1280` | YOLO 推理分辨率（越大越准越慢） |
| `YOCR_CONF` / `YOCR_IOU` | 0.25 / 0.45 | 默认置信度 / NMS 阈值 |
| `YOCR_OCR_LANG` | `ch` | OCR 语言 |
| `YOCR_OCR_DET_MODEL` / `YOCR_OCR_REC_MODEL` | PP-OCRv5_mobile_* | 换 `PP-OCRv5_server_*` 提精度 |
| `YOCR_OCR_MKLDNN` | `1` | oneDNN 加速开关（不兼容自动关闭） |
| `YOCR_LOG_LEVEL` | `INFO` | 日志级别 |
| `YOCR_SUCAI_DIR` | `data/sucai` | 素材库存储目录（meta.json + 图片） |
| `YOCR_STATIC_DIR` | `frontend/dist` | 前端构建产物目录（存在 index.html 时托管在 `/`） |

## 项目结构

```
src/yocr/
├── api.py          # REST 路由 (/detect /ocr /analyze /sucai /match /models /healthz)
├── app.py          # FastAPI 工厂 + lifespan 预热 + SPA 静态托管
├── cli.py          # yocr serve 命令
├── config.py       # YOCR_* 环境变量配置
├── detectors.py    # 多 YOLO 模型注册表（懒加载/HF下载/线程安全）
├── imaging.py      # 图像编解码（兼容被传输层损坏 \\r\\n 的 PNG）
├── matching.py     # 检测结果目标匹配（text/label/q）
├── ocr_engine.py   # PaddleOCR 封装（2.x/3.x 兼容 + oneDNN 降级）
├── pipeline.py     # 检测/OCR/文本归属分析流水线 + 素材批量比对
├── schemas.py      # Pydantic 请求/响应模型
├── sucai.py        # 素材库持久化存储（meta.json + PNG）
└── template.py     # 多尺度灰度 NCC 模板定位
examples/
└── yocr_client.py  # 测试框架可复用的 Python 客户端工具类
frontend/           # Vue 3 + Vite 素材管理界面（构建后由服务托管）
docs/
└── tutorial.zh-CN.md  # 中文实战教程
tests/              # pytest 单元测试
deploy/             # systemd unit 模板 (make systemd-install 渲染安装)
Dockerfile / docker-compose.yml
Makefile            # 开发/Docker/systemd 部署任务入口
```

## 测试

```bash
uv run pytest tests/ -q
```
