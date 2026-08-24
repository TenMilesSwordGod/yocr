# yocr 使用教程（实战场景版）

> **yocr 是一个纯粹的图像分析服务**：你上传截图，它返回元素坐标、类别和文字。
> 它不对任何设备做操作——截屏、点击、滑动等动作由你的测试框架自行执行，
> yocr 只负责回答"目标在哪、上面写着什么"。
>
> 本文所有示例代码均已在真实运行的服务上验证过。

---

## 0. 准备

```bash
# 启动服务（默认 8000 端口）
uv run yocr serve --port 8000
# 或 docker compose up -d --build

# 客户端依赖
pip install requests

# 可选：把 examples/yocr_client.py 拷进你的测试工程，下文所有示例都基于它
```

交互式调试页面：**http://127.0.0.1:8000/docs** （Swagger UI，可直接上传图片试）

**图片来源由你的框架决定**（截屏工具、模拟器导出、现成素材均可），
接口只接收图片字节流。下文示例统一从本地文件读取：

```python
png = open("screen.png", "rb").read()
```

---

## 1. 三个识别接口怎么选

| 接口 | 干什么 | 什么时候用 |
|---|---|---|
| `POST /api/v1/detect` | 只做 YOLO 元素检测 | 只要图标/控件框，不需要文字；速度最快 |
| `POST /api/v1/ocr` | 只做 PaddleOCR 文字识别 | 只想拿屏面上的全部文字及位置 |
| `POST /api/v1/analyze` | 检测 + OCR 融合 | **推荐默认用它**：每个元素自带 `text` 字段 |

三种输入方式（任选）：
- multipart 文件：`curl -F "file=@screen.png" ...`
- JSON：`{"image_base64": "<base64字符串>"}`
- 原始二进制 body（Content-Type 随意，body 就是图片字节）

常用 query 参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `model` | `android_ui_detection_yolov8` | 模型别名：`screenparser` / 你注册的别名 |
| `conf` | `0.25` | 置信度阈值。找不到目标时可降到 0.15~0.2 再试 |
| `imgsz` | `1280` | 推理分辨率，小图标多时可调大 |
| `with_ocr` | `true` | `/analyze` 里关掉 OCR 可提速 |

---

## 2. 看懂响应（坐标就在 box.center 里）

```jsonc
{
  "model": "ScreenParser",
  "image": {"width": 1280, "height": 720},     // 截图分辨率
  "elements": [                                 // YOLO 检出的 UI 元素
    {
      "id": 0,
      "label": "Button",                        // 类别名（定位靠它）
      "confidence": 0.71,
      "box": {
        "xyxy":   [321, 436, 481, 496],         // 左上角、右下角
        "xywh":   [321, 436, 160, 60],          // 左上角 + 宽高
        "center": [401, 466]                    // ★ 目标中心点
      },
      "text": "Deny",                           // OCR 归属到该元素上的文字
      "text_confidence": 0.99
    }
  ],
  "lines":   [...],                             // 全部 OCR 文本行（独立于元素）
  "full_text": "...",                           // 整屏文本，方便 assert
  "timing": {"total_ms": 4056.1, "detect_ms": 2847.6, "ocr_ms": 1204.6}
}
```

**定位 = 在 `elements` 里找到你要的那个 → 取 `box.center`（或用 `xywh` 自行计算其它锚点）。**
拿到坐标后怎么用（点击/高亮/比对基线）完全是你的测试框架的事。

### ScreenParser 常用类别（label）速查

| 想找的东西 | 常见 label |
|---|---|
| 图标（齿轮、返回箭头、三点菜单…） | `App Icon`、`Utility Button`、`File Icon`、`Image` |
| 按钮（有字没字都是它） | `Button`、`Utility Button` |
| 弹窗容器 | `Alert`、`Window`、`PopUp Menu`、`ContextMenu` |
| 开关/滑杆/勾选 | `Switch`、`Toggles`、`Slider`、`Checkbox`、`Radiobox` |
| 输入类 | `Text Input`、`Search Field`、`Search Bar` |
| 列表 | `List`、`List Item` |
| 文本 | `Text`、`Heading`、`Link`、`Badge` |
| 导航 | `Navigation Bar`、`Status Bar`、`Tab Bar`、`Bottom navigation`、`Tab` |

> 用你自己训练的 `android_ui_detection_yolov8` 时，类别以 `GET /api/v1/models` 返回的 `classes` 为准。
> 注意：客户端里按 label 匹配是包含式的，传 `Button` 也会命中 `Utility Button`。

---

## 3. 场景实战

以下代码基于 `examples/yocr_client.py`：

```python
from yocr_client import YocrClient
c = YocrClient("http://127.0.0.1:8000/api/v1")
KW = dict(model="screenparser", conf=0.25)   # 后文 KW 都指这个
png = open("screen.png", "rb").read()
```

### 场景一：有文字的控件（Settings、"确定"、蓝牙开关行…）

最常见也最稳的场景 —— **按文字找，取中心点**：

```python
result = c.analyze(png, **KW)

el = c.find(result, text="Settings")            # 包含匹配、忽略大小写和空格
x, y = c.center(el)                              # -> (115, 667)

el2 = c.find(result, text="允许")                # 中文一样用
```

快捷写法一行搞定：

```python
x, y = c.find_by_text(png, "Bluetooth", **KW)
```

文字匹配三种模式：

```python
c.find(r, text="Allow",        match_mode="contains")  # 默认：包含
c.find(r, text="Allow",        match_mode="exact")     # 全等
c.find(r, text="only onc",     match_mode="fuzzy")     # 容错（OCR 少字符/粘连）
```

> 实测 fuzzy：OCR 结果 `Only once`，查询 `only onc` 也能命中 (600, 465)。

### 场景二：弹窗中的按钮 · 有文字（Allow / Deny / 确定 / 取消…）

直接按文字找即可。但要注意**弹窗遮罩下背景可能有相同文字**，
稳妥做法是用客户端的 `find_dialog_button`：它会先找弹窗容器（`Alert`/`Window` 等），
把搜索范围限制在容器内；容器没检出时自动退化为全屏查找：

```python
allow_xy = c.find_dialog_button(png, text="Allow", **KW)
deny_xy  = c.find_dialog_button(png, text="Deny",  **KW)
# 实测: Allow -> (836, 466)，Deny -> (401, 466)
```

手动等价写法（region 过滤）：

```python
r = c.analyze(png, **KW)
alert = c.find(r, label="Alert") or c.find(r, label="Window")
region = tuple(alert["box"]["xyxy"]) if alert else None
el = c.find(r, text="Allow", region=region)      # 元素中心必须在弹窗矩形内才算数
```

### 场景三：弹窗中的按钮 · 无文字（X 关闭钮、齿轮、箭头…）

无文字就走 **label 匹配**。实测一张权限弹窗（右上角画了个 ✕）：

```python
r = c.analyze(png, **KW)

# 方法A：按类别找
el = c.find(r, label="Utility Button")           # X 按钮被识别为 Utility Button
x, y = c.center(el)                              # -> (941, 233)

# 方法B：很多图标形状会被 OCR 读成字符，比如 ✕ -> 'X'
x, y = c.find_by_text(png, "X", **KW)            # 同样命中 (941, 233)

# 方法C：封装好的弹窗按钮查找
x, y = c.find_dialog_button(png, text="X", **KW)
```

### 场景四：纯图标（桌面齿轮 Settings 入口、返回箭头等）

策略：**先按候选 label 链找，找不到再降级按文字找**（图标下方常有小字标签）：

```python
def locate_settings(c, png):
    # 1) 图标本身
    xy = c.find_icon(png, region=(0, 500, 400, 720), **KW)   # region 限定屏幕左下角区域
    if xy:
        return xy
    # 2) 降级：图标下面的文字标签
    return c.find_by_text(png, "Settings", **KW)
```

要点：
- `find_icon` 依次尝试 `App Icon -> Utility Button -> Button`
- **加 region 缩小范围**是纯图标定位的关键（桌面上图标太多），先粗看一遍元素列表再定区域：

```python
for e in c.analyze(png, **KW)["elements"]:
    print(e["label"], round(e["confidence"],2), e["box"]["xyxy"], repr(e.get("text")))
```

> 注意：真实截图里的齿轮是图形 icon，通常能被识别为 `App Icon`；
> 如果你的目标是极简线条图标且检不出来，先把 `conf` 降到 0.15~0.2，或换 `imgsz=1600`。

### 场景五：多个相似元素（列表项、重复卡片）

`find_all` 拿全部命中，按序号选择；或用 `index` 直接取第 N 个：

```python
r = c.analyze(png, **KW)

items = c.find_all(r, label="List Item")          # 所有列表项，按置信度降序
if len(items) >= 3:
    x, y = c.center(items[2])                     # 第 3 项的中心

el = c.find(r, text="更多", index=1)              # 第 2 个含"更多"的元素
```

配合文字+位置可以精确锁定："设置列表里第 N 行右侧的开关"：

```python
row = c.find(r, text="蓝牙")                       # 先找行
x1, y1, x2, y2 = row["box"]["xyxy"]
sw  = c.find(r, label="Switch", region=(x2, y1, x2 + 400, y2))  # 行右侧 400px 内找开关
```

### 场景六：开关 Switch / 滑杆 Slider / 输入框

```python
# 开关：找到它当前的位置与状态区域
sw = c.find(c.analyze(png, **KW), label="Switch")
print(sw["box"]["center"])                        # 交给你的执行器去切换

# 输入框：拿到聚焦位置
ti = c.find(r, label="Text Input") or c.find(r, label="Search Field")
if ti:
    print(ti["box"]["center"])                    # 你的框架在此处输入文字

# 滑杆：纯几何计算任意刻度的坐标（不涉及设备操作）
sl = c.find(r, label="Slider")
x1, _, x2, _ = sl["box"]["xyxy"]
target_x = int(x1 + (x2 - x1) * 0.7)              # 70% 刻度处的 x 坐标
```

### 场景七：等待元素出现（跳转、加载、动画）

自动化最常见的坑：截屏太早。用轮询等待代替 sleep ——
`wait_for` 接收一个"返回最新图片字节"的可调用对象，每 `interval` 秒分析一次新图：

```python
def screenshot() -> bytes:
    """由你的测试环境提供最新截图（这里以读文件为例）。"""
    return open("latest.png", "rb").read()

center = c.wait_for(screenshot, text="Allow", timeout=15, interval=1.0, **KW)
if center is None:
    raise AssertionError("15 秒内弹窗未出现")
x, y = center                                      # 出现后立刻拿到坐标
```

`wait_for(text=...)` / `wait_for(label=...)` 都支持；轮询期间单次失败视为"还没出现"，不会中断等待。

### 场景八：整屏文字断言（不关心坐标）

```python
r = c.analyze(png, **KW)
assert "AAOS" in r["full_text"]

lines = [(l["text"], l["box"]["center"]) for l in r["lines"]]
```

---

## 4. 封装成 pytest fixture 的建议

```python
# conftest.py
import pytest
from yocr_client import YocrClient

@pytest.fixture(scope="session")
def yocr():
    return YocrClient("http://127.0.0.1:8000/api/v1")

def test_allow_dialog_visible(yocr):
    png = open("screen.png", "rb").read()          # 截图来源由你的环境决定
    el = yocr.find(yocr.analyze(png, model="screenparser"), text="Allow")
    assert el is not None, "未找到 Allow 按钮"
    x, y = yocr.center(el)
    assert 0 <= x <= 1920 and 0 <= y <= 1080       # 坐标在合理范围内
```

---

## 5. 参数调优速查

| 目标 | 做法 |
|---|---|
| 找不到目标元素 | `conf` 降到 `0.15`；`imgsz` 升到 `1600`；确认模型选对（`GET /models` 看类别表） |
| 误检太多 | `conf` 升到 `0.35`；用 `region` 限定搜索范围 |
| 提速 | `/analyze?with_ocr=false`；只用 `/detect`；预热：启动时设 `YOCR_PRELOAD_MODELS=screenparser,YOCR_PRELOAD_OCR=1` |
| 提精度（OCR） | 服务端设 `YOCR_OCR_DET_MODEL=PP-OCRv5_server_det`、`YOCR_OCR_REC_MODEL=PP-OCRv5_server_rec` 后重启 |
| 提精度（检测） | 有 GPU 时 `YOCR_DEVICE=cuda:0` |
| 坐标系不一致 | `box.*` 是相对你上传的那张图的像素坐标。若你的目标坐标系分辨率不同，按比例换算：`x' = cx * target_w / img_w` |

## 6. FAQ

**Q: 第一次请求特别慢？**
首次调用要加载权重 + 首帧推理建图，之后就是热路径（CPU 实测：检测 ~2.7s→2.4s，OCR ~0.75s）。
生产上务必配置启动预热：`YOCR_PRELOAD_MODELS=screenparser` + `YOCR_PRELOAD_OCR=1`。

**Q: requests 报 502 Bad Gateway？**
系统代理劫持了 localhost。客户端里已内置 `session.trust_env = False`；
用 curl 的话加 `--noproxy '*'`。

**Q: 怎么看我的模型有哪些类别？**
服务启动后 `GET /api/v1/models`，`classes` 字段就是 id->类别名 映射；
自己的 `.pt` 放进 `./models/` 即可用 `?model=android_ui_detection_yolov8` 加载。

**Q: 中文能识别吗？**
能。默认 `YOCR_OCR_LANG=ch` 支持中英混排。

**Q: 返回的坐标是相对什么的？**
相对你上传的那张图片的像素坐标系（响应里 `image.width/height` 就是该图的尺寸）。
若需要换算到别的坐标系（如不同分辨率的渲染层），按第 5 节公式缩放即可。
