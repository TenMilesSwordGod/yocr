from yocr.matching import best_match, text_score
from yocr.schemas import Box, Element


def el(label="Button", text=None, conf=0.9, x=0, y=0):
    box = Box.from_xyxy(x, y, x + 100, y + 40)
    return Element(id=0, label=label, class_id=1, confidence=conf, box=box, text=text)


def test_text_score_modes():
    assert text_score("Allow", "Allow") == 1.0
    assert text_score("allow", "  Allow ") >= 1.0 - 1e-9      # 归一化后 exact 也命中 contains
    assert text_score("all", "Allow") == 1.0                  # contains
    assert text_score("Deny", "Allow") == 0.0
    assert text_score("only onc", "Only once", mode="fuzzy") > 0.75
    assert text_score("xyz", "Only once", mode="fuzzy") == 0.0


def test_best_match_by_text():
    elements = [el(text="Bluetooth"), el(label="Button", text="Allow", conf=0.7)]
    m = best_match(elements, text="allow")
    assert m is not None and m.text == "Allow"


def test_best_match_by_label_and_q():
    elements = [el(label="App Icon"), el(label="Button", text="OK")]
    assert best_match(elements, label="icon").label == "App Icon"
    # q 泛搜索: 文本或类别任一命中
    assert best_match(elements, q="ok").text == "OK"
    assert best_match(elements, q="app").label == "App Icon"


def test_best_match_and_condition_and_none():
    elements = [el(label="Button", text="Allow"), el(label="Button", text="Deny")]
    m = best_match(elements, text="deny", label="Button")
    assert m is not None and m.text == "Deny"
    assert best_match(elements, text="missing") is None
    assert best_match(elements) is None                       # 无条件不匹配
