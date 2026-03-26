"""
expense_engine.py — Core ML/NLP engine (no GUI, no Voice hardware)
Dùng cho FastAPI server.
"""

import re
import json
import pickle
import os
import threading
from datetime import datetime, timedelta

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

try:
    from underthesea import word_tokenize
    USE_TOKENIZER = True
except ImportError:
    USE_TOKENIZER = False

try:
    from transformers import pipeline as hf_pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

MODEL_PATH        = "models/expense_classifier.pkl"
USER_DATA_PATH    = "models/user_corrections.json"

# ══════════════════════════════════════════════════════════════════
# 1. KEYWORD MAPPING (Rule-based)
#    Mỗi từ khoá có điểm trọng số: 1.0 = match chính xác cụm,
#    0.7 = match từ đơn yếu hơn
# ══════════════════════════════════════════════════════════════════
KEYWORD_RULES: dict[str, list[tuple[str, float]]] = {
    "ăn uống": [
        ("cà phê", 1.0), ("cafe", 1.0), ("cf ", 0.9), ("phở", 1.0),
        ("bún bò", 1.0), ("cơm tấm", 1.0), ("bánh mì", 1.0),
        ("grab food", 1.0), ("baemin", 1.0), ("shopee food", 1.0),
        ("trà sữa", 1.0), ("gong cha", 1.0), ("kfc", 1.0),
        ("mcdonald", 1.0), ("highlands", 1.0), ("coffee house", 1.0),
        ("nhậu", 0.9), ("bia", 0.8), ("lẩu", 0.9), ("đồ ăn", 0.9),
        ("ăn sáng", 1.0), ("ăn trưa", 1.0), ("ăn tối", 1.0),
        ("uống", 0.7), ("nước ngọt", 0.9), ("sinh tố", 0.9),
        ("bánh ngọt", 0.9), ("đồ ăn vặt", 1.0), ("order", 0.6),
    ],
    "di chuyển": [
        ("xăng", 1.0), ("tiền xăng", 1.0), ("đổ xăng", 1.0),
        ("grab bike", 1.0), ("grab car", 1.0), ("be bike", 1.0),
        ("taxi", 1.0), ("uber", 1.0), ("xe ôm", 1.0),
        ("xe buýt", 1.0), ("vé tàu", 1.0), ("vé xe", 0.9),
        ("vé máy bay", 1.0), ("đặt vé bay", 1.0),
        ("gửi xe", 0.9), ("phí gửi xe", 1.0), ("thuê xe", 0.9),
        ("gọi grab", 1.0), ("gọi be", 1.0),
    ],
    "mua sắm": [
        ("shopee", 1.0), ("lazada", 1.0), ("tiki", 1.0),
        ("mua quần", 0.9), ("mua áo", 0.9), ("mua giày", 1.0),
        ("mua đồng hồ", 1.0), ("mua túi", 0.9), ("mua phụ kiện", 0.9),
        ("mua tai nghe", 1.0), ("vinmart", 1.0), ("coopmart", 1.0),
        ("siêu thị", 0.8), ("mua đồ dùng", 0.8), ("mua mỹ phẩm", 0.9),
    ],
    "giải trí": [
        ("xem phim", 1.0), ("cgv", 1.0), ("lotte cinema", 1.0),
        ("netflix", 1.0), ("spotify", 1.0), ("youtube premium", 1.0),
        ("chơi game", 1.0), ("mua game", 0.9), ("steam", 1.0),
        ("du lịch", 1.0), ("đặt khách sạn", 1.0), ("karaoke", 1.0),
        ("bowling", 1.0), ("concert", 1.0), ("vé vào cổng", 1.0),
    ],
    "y tế": [
        ("thuốc", 0.9), ("mua thuốc", 1.0), ("khám bệnh", 1.0),
        ("bệnh viện", 1.0), ("phòng khám", 1.0), ("tiêm phòng", 1.0),
        ("xét nghiệm", 1.0), ("nha khoa", 1.0), ("bác sĩ", 1.0),
        ("vitamin", 0.9), ("siêu âm", 1.0), ("mắt kính", 1.0),
    ],
    "giáo dục": [
        ("học phí", 1.0), ("đóng học phí", 1.0), ("sách giáo khoa", 1.0),
        ("udemy", 1.0), ("coursera", 1.0), ("gia sư", 1.0),
        ("học tiếng anh", 1.0), ("học lập trình", 1.0),
        ("lớp học thêm", 1.0), ("tài liệu học", 0.9), ("khóa học", 0.9),
    ],
    "hóa đơn": [
        ("tiền điện", 1.0), ("trả tiền điện", 1.0), ("tiền nước", 1.0),
        ("tiền nhà", 1.0), ("tiền phòng", 1.0), ("tiền trọ", 1.0),
        ("cước internet", 1.0), ("wifi", 0.8), ("điện thoại trả sau", 1.0),
        ("bảo hiểm xe", 1.0), ("bảo hiểm y tế", 1.0),
        ("phí chung cư", 1.0), ("tiền gas", 1.0),
    ],
    "nhà cửa": [
        ("nội thất", 1.0), ("bàn ghế", 1.0), ("đồ gia dụng", 1.0),
        ("sửa chữa nhà", 1.0), ("thợ sửa", 0.9), ("thuê dọn dẹp", 1.0),
        ("đồ trang trí", 0.9), ("cây cảnh", 1.0), ("mua đồ nội thất", 1.0),
    ],
    "công việc": [
        ("marketing", 1.0), ("quảng cáo facebook", 1.0), ("ads google", 1.0),
        ("chạy ads", 0.9), ("phần mềm", 0.8), ("trả lương", 1.0),
        ("freelancer", 1.0), ("domain", 1.0), ("hosting", 1.0),
        ("chi phí vận hành", 1.0),
    ],
    "làm đẹp": [
        ("cắt tóc", 1.0), ("uốn tóc", 1.0), ("nhuộm tóc", 1.0),
        ("spa", 0.9), ("skincare", 1.0), ("nước hoa", 1.0),
        ("làm nail", 1.0), ("nail", 0.8), ("massage", 1.0),
        ("gội đầu", 1.0), ("mỹ phẩm", 0.9),
    ],
    "thú cưng": [
        ("thức ăn cho chó", 1.0), ("cát mèo", 1.0), ("thú cưng", 1.0),
        ("spa thú cưng", 1.0), ("tiêm phòng chó", 1.0), ("tiêm phòng mèo", 1.0),
    ],
    "đầu tư": [
        ("cổ phiếu", 1.0), ("chứng khoán", 1.0), ("crypto", 1.0),
        ("trading", 1.0), ("nạp tiền trading", 1.0), ("phí sàn", 1.0),
        ("phí giao dịch", 0.8),
    ],
    "quà tặng": [
        ("quà sinh nhật", 1.0), ("mua quà", 0.9), ("tặng quà", 1.0),
        ("mừng cưới", 1.0), ("đám cưới", 1.0), ("đám giỗ", 1.0),
    ],
    "gia đình": [
        ("đồ cho con", 1.0), ("tiền sữa", 1.0), ("sữa cho bé", 1.0),
        ("bỉm tã", 1.0), ("học phí cho con", 1.0), ("đồ chơi trẻ em", 1.0),
    ],
    "phí dịch vụ": [
        ("phí ngân hàng", 1.0), ("phí chuyển khoản", 1.0),
        ("phí duy trì", 1.0), ("phí app", 1.0), ("subscription", 1.0),
        ("phí dịch vụ", 1.0),
    ],
    "thể thao": [
        ("đi gym", 1.0), ("phòng gym", 1.0), ("gym", 0.8),
        ("whey protein", 1.0), ("thực phẩm bổ sung", 1.0),
        ("dụng cụ tập", 1.0), ("yoga", 1.0), ("học yoga", 1.0),
    ],
    "phát sinh": [
        ("sửa xe đột xuất", 1.0), ("khẩn cấp", 1.0),
        ("mất đồ", 0.9), ("phạt giao thông", 1.0), ("đột xuất", 0.9),
    ],
    "khác": [
        ("chuyển tiền", 0.7), ("lì xì", 1.0), ("từ thiện", 1.0),
        ("nạp tiền điện thoại", 1.0), ("nạp thẻ", 0.9), ("rút tiền", 1.0),
    ],
}

ALL_CATEGORIES = sorted(KEYWORD_RULES.keys())


# ══════════════════════════════════════════════════════════════════
# 2. TRAINING DATA
# ══════════════════════════════════════════════════════════════════
TRAINING_DATA: list[tuple[str, str]] = [
    ("uống cà phê buổi sáng", "ăn uống"),
    ("cafe với bạn", "ăn uống"),
    ("cf sáng nay", "ăn uống"),
    ("ăn phở bò", "ăn uống"),
    ("ăn trưa văn phòng", "ăn uống"),
    ("ăn tối gia đình", "ăn uống"),
    ("ăn sáng bánh mì", "ăn uống"),
    ("order grab food", "ăn uống"),
    ("đặt baemin", "ăn uống"),
    ("order shopee food", "ăn uống"),
    ("uống trà sữa", "ăn uống"),
    ("trà sữa gong cha", "ăn uống"),
    ("nhậu với bạn bè", "ăn uống"),
    ("uống bia", "ăn uống"),
    ("ăn bún bò", "ăn uống"),
    ("ăn cơm tấm", "ăn uống"),
    ("ăn KFC", "ăn uống"),
    ("ăn McDonald", "ăn uống"),
    ("mua nước ngọt", "ăn uống"),
    ("sinh tố hoa quả", "ăn uống"),
    ("ăn lẩu cuối tuần", "ăn uống"),
    ("đồ ăn vặt", "ăn uống"),
    ("bánh ngọt tiệm bánh", "ăn uống"),
    ("cà phê highlands", "ăn uống"),
    ("the coffee house", "ăn uống"),
    ("đổ xăng xe máy", "di chuyển"),
    ("tiền xăng", "di chuyển"),
    ("đi grab bike", "di chuyển"),
    ("gọi grab car", "di chuyển"),
    ("đi taxi", "di chuyển"),
    ("gọi be bike", "di chuyển"),
    ("đi xe buýt", "di chuyển"),
    ("gửi xe máy", "di chuyển"),
    ("vé xe khách", "di chuyển"),
    ("vé tàu hỏa", "di chuyển"),
    ("vé máy bay", "di chuyển"),
    ("đặt vé bay", "di chuyển"),
    ("phí gửi xe", "di chuyển"),
    ("thuê xe ô tô", "di chuyển"),
    ("đi xe ôm", "di chuyển"),
    ("uber đi làm", "di chuyển"),
    ("mua quần áo", "mua sắm"),
    ("mua giày mới", "mua sắm"),
    ("shopee mua đồ", "mua sắm"),
    ("lazada mua hàng", "mua sắm"),
    ("tiki mua sách", "mua sắm"),
    ("mua phụ kiện điện thoại", "mua sắm"),
    ("mua tai nghe", "mua sắm"),
    ("siêu thị vinmart", "mua sắm"),
    ("đi coopmart mua đồ", "mua sắm"),
    ("mua đồ dùng nhà bếp", "mua sắm"),
    ("mua mỹ phẩm", "mua sắm"),
    ("mua túi xách", "mua sắm"),
    ("mua đồng hồ", "mua sắm"),
    ("xem phim rạp", "giải trí"),
    ("mua vé CGV", "giải trí"),
    ("lotte cinema", "giải trí"),
    ("chơi game", "giải trí"),
    ("mua game steam", "giải trí"),
    ("đăng ký netflix", "giải trí"),
    ("spotify premium", "giải trí"),
    ("youtube premium", "giải trí"),
    ("đi du lịch", "giải trí"),
    ("đặt khách sạn", "giải trí"),
    ("vé vào cổng khu vui chơi", "giải trí"),
    ("karaoke với bạn", "giải trí"),
    ("đi bowling", "giải trí"),
    ("concert âm nhạc", "giải trí"),
    ("mua thuốc cảm", "y tế"),
    ("mua thuốc", "y tế"),
    ("khám bệnh viện", "y tế"),
    ("phòng khám đa khoa", "y tế"),
    ("tiêm phòng", "y tế"),
    ("xét nghiệm máu", "y tế"),
    ("nha khoa nhổ răng", "y tế"),
    ("mắt kính mới", "y tế"),
    ("bác sĩ gia đình", "y tế"),
    ("thuốc bổ vitamin", "y tế"),
    ("siêu âm thai", "y tế"),
    ("học phí tháng này", "giáo dục"),
    ("đóng học phí", "giáo dục"),
    ("mua sách giáo khoa", "giáo dục"),
    ("khóa học online udemy", "giáo dục"),
    ("học tiếng Anh", "giáo dục"),
    ("gia sư toán", "giáo dục"),
    ("học lập trình", "giáo dục"),
    ("coursera khóa học", "giáo dục"),
    ("mua tài liệu học", "giáo dục"),
    ("lớp học thêm", "giáo dục"),
    ("tiền điện tháng này", "hóa đơn"),
    ("trả tiền điện", "hóa đơn"),
    ("tiền nước sinh hoạt", "hóa đơn"),
    ("tiền nhà thuê", "hóa đơn"),
    ("tiền phòng trọ", "hóa đơn"),
    ("cước internet wifi", "hóa đơn"),
    ("điện thoại trả sau", "hóa đơn"),
    ("bảo hiểm xe máy", "hóa đơn"),
    ("bảo hiểm y tế", "hóa đơn"),
    ("phí chung cư", "hóa đơn"),
    ("tiền gas nấu ăn", "hóa đơn"),
    ("mua đồ nội thất", "nhà cửa"),
    ("mua bàn ghế", "nhà cửa"),
    ("mua đồ gia dụng", "nhà cửa"),
    ("sửa chữa nhà", "nhà cửa"),
    ("thuê thợ sửa điện", "nhà cửa"),
    ("thuê dọn dẹp", "nhà cửa"),
    ("mua đồ trang trí", "nhà cửa"),
    ("mua cây cảnh", "nhà cửa"),
    ("chi phí marketing", "công việc"),
    ("chạy quảng cáo facebook", "công việc"),
    ("chạy ads google", "công việc"),
    ("mua phần mềm làm việc", "công việc"),
    ("trả lương nhân viên", "công việc"),
    ("thuê freelancer", "công việc"),
    ("chi phí vận hành", "công việc"),
    ("mua domain website", "công việc"),
    ("mua hosting", "công việc"),
    ("cắt tóc", "làm đẹp"),
    ("uốn tóc", "làm đẹp"),
    ("nhuộm tóc", "làm đẹp"),
    ("spa chăm sóc da", "làm đẹp"),
    ("mua mỹ phẩm skincare", "làm đẹp"),
    ("mua nước hoa", "làm đẹp"),
    ("làm nail", "làm đẹp"),
    ("massage thư giãn", "làm đẹp"),
    ("gội đầu thư giãn", "làm đẹp"),
    ("mua thức ăn cho chó", "thú cưng"),
    ("mua cát mèo", "thú cưng"),
    ("đưa thú cưng đi khám", "thú cưng"),
    ("tiêm phòng chó mèo", "thú cưng"),
    ("spa thú cưng", "thú cưng"),
    ("mua cổ phiếu", "đầu tư"),
    ("nạp tiền chứng khoán", "đầu tư"),
    ("mua crypto", "đầu tư"),
    ("nạp tiền trading", "đầu tư"),
    ("đóng phí sàn", "đầu tư"),
    ("phí giao dịch", "đầu tư"),
    ("mua quà sinh nhật", "quà tặng"),
    ("tặng quà", "quà tặng"),
    ("mừng cưới", "quà tặng"),
    ("đi đám cưới", "quà tặng"),
    ("đi đám giỗ", "quà tặng"),
    ("mua đồ cho con", "gia đình"),
    ("tiền sữa cho bé", "gia đình"),
    ("bỉm tã", "gia đình"),
    ("học phí cho con", "gia đình"),
    ("đồ chơi trẻ em", "gia đình"),
    ("phí ngân hàng", "phí dịch vụ"),
    ("phí chuyển khoản", "phí dịch vụ"),
    ("phí duy trì tài khoản", "phí dịch vụ"),
    ("phí app", "phí dịch vụ"),
    ("phí subscription", "phí dịch vụ"),
    ("đi gym", "thể thao"),
    ("mua whey protein", "thể thao"),
    ("mua thực phẩm bổ sung", "thể thao"),
    ("đăng ký phòng gym", "thể thao"),
    ("mua dụng cụ tập", "thể thao"),
    ("học yoga", "thể thao"),
    ("sửa xe đột xuất", "phát sinh"),
    ("chi phí khẩn cấp", "phát sinh"),
    ("mất đồ phải mua lại", "phát sinh"),
    ("phạt giao thông", "phát sinh"),
    ("chuyển tiền bạn bè", "khác"),
    ("tiền lì xì", "khác"),
    ("từ thiện", "khác"),
    ("nạp tiền điện thoại", "khác"),
    ("rút tiền ATM", "khác"),
]


# ══════════════════════════════════════════════════════════════════
# 3. TOKENIZER
# ══════════════════════════════════════════════════════════════════
def tokenize(text: str) -> str:
    text = text.lower().strip()
    if USE_TOKENIZER:
        return word_tokenize(text, format="text")
    return text


# ══════════════════════════════════════════════════════════════════
# 4. RULE-BASED SCORER
#    Trả về dict {category: score} đã chuẩn hoá về [0, 1]
# ══════════════════════════════════════════════════════════════════
def rule_based_score(text: str) -> dict[str, float]:
    t = text.lower()
    scores: dict[str, float] = {}

    for category, keywords in KEYWORD_RULES.items():
        cat_score = 0.0
        for kw, weight in keywords:
            if kw in t:
                bonus = min(len(kw.split()) * 0.15, 0.3)
                cat_score = max(cat_score, weight + bonus)
        if cat_score > 0:
            scores[category] = min(cat_score, 1.0)

    return scores


# ══════════════════════════════════════════════════════════════════
# 5. ML / NLP SCORER
# ══════════════════════════════════════════════════════════════════
def train_model(data: list[tuple[str, str]]) -> Pipeline:
    print("🔄 Đang train model...")
    texts  = [tokenize(t) for t, _ in data]
    labels = [c for _, c in data]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("svm",   LinearSVC(C=1.0, max_iter=2000, dual=False)),
    ])

    if len(set(labels)) >= 2 and len(texts) >= 6:
        cv = min(3, len(texts) // len(set(labels)))
        if cv >= 2:
            scores = cross_val_score(model, texts, labels, cv=cv, scoring="accuracy")
            print(f"📊 Cross-val accuracy: {scores.mean():.1%} ± {scores.std():.1%}")

    model.fit(texts, labels)

    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved → {MODEL_PATH}")
    return model


def load_or_train(data: list[tuple[str, str]]) -> Pipeline:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return train_model(data)


def nlp_score(model: Pipeline, text: str) -> dict[str, float]:
    tok    = tokenize(text)
    labels = model.classes_
    raw    = model.decision_function([tok])[0]

    e      = np.exp(raw - raw.max())
    probs  = e / e.sum()

    return {cat: float(p) for cat, p in zip(labels, probs)}


# ══════════════════════════════════════════════════════════════════
# 6. WEIGHTED FUSION
# ══════════════════════════════════════════════════════════════════
RULE_WEIGHT = 0.70
NLP_WEIGHT  = 0.30


def fuse(rule_scores: dict[str, float], nlp_scores: dict[str, float]) -> tuple[str, float, str]:
    all_cats = set(rule_scores) | set(nlp_scores)
    fused: dict[str, float] = {}

    has_rule_match = bool(rule_scores)

    for cat in all_cats:
        r = rule_scores.get(cat, 0.0)
        n = nlp_scores.get(cat, 0.0)

        if has_rule_match:
            fused[cat] = RULE_WEIGHT * r + NLP_WEIGHT * n
        else:
            fused[cat] = n

    best_cat   = max(fused, key=fused.__getitem__)
    best_score = fused[best_cat]

    if not has_rule_match:
        method = "nlp"
    elif rule_scores.get(best_cat, 0) > 0.5:
        method = "rule"
    else:
        method = "hybrid"

    return best_cat, best_score, method


# ══════════════════════════════════════════════════════════════════
# 7. EXTRACT AMOUNT & DATE
# ══════════════════════════════════════════════════════════════════

# Bảng scale dùng chung cho Pass 4 của extract_amount
_SCALE_MAP_INLINE = {
    "nghìn": 1_000, "ngàn": 1_000, "k": 1_000,
    "ca": 1_000, "cá": 1_000, "kê": 1_000, "ke": 1_000,
    "triệu": 1_000_000, "củ": 1_000_000, "tr": 1_000_000,
    "lít": 100_000, "chai": 100_000,
    "tỷ": 1_000_000_000, "tỉ": 1_000_000_000,
}

def extract_amount(text: str) -> int:
    """
    Trích xuất số tiền từ text — hỗ trợ cả dạng viết và dạng nói:
      Viết : 50k, 3tr, 200.000, 1,5 triệu
      Nói  : năm mươi nghìn, ba lít, hai chai, năm củ, năm mươi ca
      Artifact: năm mươi 1k → 50k  (PhoWhisper: nghìn → 1k)

    FIX: "50 nghìn" (digit + đơn vị tiếng Việt) nay được xử lý đúng
         ở Pass 1; Pass 4 là lớp phòng thủ bổ sung.
    """
    t = text.lower().strip()

    # ── Tiền xử lý: xoá artifact "1" trước scale khi đứng sau số chữ ──
    t = re.sub(
        r'(?<=[a-zàáâãèéêìíòóôõùúăđơưạảấầẩẫậắặẳẵặềếệỉịọỏốồổỗộớờởỡợụủứừữựỳỷỹ])\s+1\s*(k|nghìn|ngàn|ca|cá|kê|ke)\b',
        r' \1', t)

    # ── Pass 1: chuẩn hoá dạng viết có đơn vị ────────────────────

    # Ký hiệu tắt kiểu VN: {số}{đơn vị}{1-3 chữ số lẻ}
    # Quy tắc: phần lẻ N chữ số = N / 10^len(N) * đơn vị
    #   1tr4   = 1,400,000   (lẻ 1 cs = /10)
    #   1tr47  = 1,470,000   (lẻ 2 cs = /100)
    #   1tr470 = 1,470,000   (lẻ 3 cs = /1000)
    #   30k35  = 30,350      (lẻ 2 cs = /100 * 1000)
    #   30k350 = 30,350      (lẻ 3 cs = /1000 * 1000)

    def _shorthand(base, frac_str, unit):
        """base=30, frac_str='350', unit=1000 → 30*1000 + 350 = 30350"""
        scale = 10 ** len(frac_str)
        return int(round((int(base) + int(frac_str) / scale) * unit))

    t = re.sub(r'(\d+)\s*(tỉ|tỷ)\s*(\d{1,3})\b',
               lambda m: str(_shorthand(m.group(1), m.group(3), 1_000_000_000)), t,
               flags=re.IGNORECASE)
    t = re.sub(r'(\d+)\s*(triệu|củ|tr)\s*(\d{1,3})\b',
               lambda m: str(_shorthand(m.group(1), m.group(3), 1_000_000)), t,
               flags=re.IGNORECASE)
    t = re.sub(r'(\d+)\s*(k|nghìn|ngàn|ca|cá|kê|ke)\s*(\d{1,3})\b',
               lambda m: str(_shorthand(m.group(1), m.group(3), 1_000)), t,
               flags=re.IGNORECASE)
    # "1 trăm 2" → 120k, "2 trăm 5" → 250k  (trăm + lẻ không có đơn vị)
    # Phải chạy TRƯỚC "trăm" plain để không bị bắt nhầm
    t = re.sub(r'(\d+)\s*trăm\s*(\d{1,2})\b(?!\s*(?:nghìn|ngàn|k\b|\d))',
               lambda m: str(_shorthand(m.group(1), m.group(2), 100_000)), t,
               flags=re.IGNORECASE)

    # Convert đơn vị có kèm số thập phân
    t = re.sub(r'(\d+[.,]\d+)\s*(triệu|tr)\b',
               lambda m: str(int(float(m.group(1).replace(',', '.')) * 1_000_000)), t)
    t = re.sub(r'(\d+[.,]\d+)\s*(tỉ|tỷ)\b',
               lambda m: str(int(float(m.group(1).replace(',', '.')) * 1_000_000_000)), t,
               flags=re.IGNORECASE)

    # Convert đơn vị nguyên — thứ tự: tr trước, tỉ sau
    # → "3 tỉ 500tr" thành "3 tỉ 500000000" → "3000000000 500000000" → sum = 3,500,000,000
    t = re.sub(r'(\d+)\s*(triệu|củ|tr)\b',
               lambda m: str(int(m.group(1)) * 1_000_000), t)

    # "X tỉ Y" khi Y không có đơn vị và Y <= 3 chữ số → Y là triệu
    # "50 tỉ 500" → 50,500,000,000  |  "3 tỉ 500000000" (đã convert) → không khớp vì >3 chữ số
    t = re.sub(r'(\d+)\s*(tỉ|tỷ)\s+(\d{1,3})\b',
               lambda m: str(int(m.group(1)) * 1_000_000_000 + int(m.group(3)) * 1_000_000),
               t, flags=re.IGNORECASE)

    t = re.sub(r'(\d+)\s*(tỉ|tỷ)\b',
               lambda m: str(int(m.group(1)) * 1_000_000_000), t,
               flags=re.IGNORECASE)
    t = re.sub(r'(\d+)\s*(lít|chai)\b',
               lambda m: str(int(m.group(1)) * 100_000), t)
    # Số thập phân + k: "30.5k" → 30500, "1.5k" → 1500
    t = re.sub(r'(\d+[.,]\d+)\s*(k|nghìn|ngàn|ca|cá|kê|ke)\b',
               lambda m: str(int(float(m.group(1).replace(',', '.')) * 1_000)), t,
               flags=re.IGNORECASE)
    # "50 nghìn", "50k", "50 ca", v.v. → "50000"
    t = re.sub(r'(\d+)\s*(nghìn|ngàn|k|ca|cá|kê|ke)\b',
               lambda m: str(int(m.group(1)) * 1_000), t,
               flags=re.IGNORECASE)
    # "N trăm" standalone (không theo sau bởi nghìn/ngàn) → N × 100,000
    # "1 trăm" → 100000, "1 trăm 20k" → "100000 20000" → sum 120,000
    t = re.sub(r'(\d+)\s*trăm\b(?!\s*(?:nghìn|ngàn|k\b))',
               lambda m: str(int(m.group(1)) * 100_000), t,
               flags=re.IGNORECASE)

    # ── Pass 2: lấy số digit ─────────────────────────────────────
    numbers = re.findall(r'\d+', t)
    amounts = [int(n) for n in numbers if 1_000 <= int(n) <= 100_000_000_000]

    # Chỉ check từ số chữ KHÔNG NHẬP NHẰNG — loại các từ hay xuất hiện
    # trong văn cảnh khác như "tư" (đầu tư), "năm" (năm nay), "ba" (ba mẹ)
    _unambiguous_vi_num = {
        "mươi", "lăm", "mốt", "nhăm", "rưỡi", "trăm",
        "hai mươi", "ba mươi", "bốn mươi", "năm mươi",
        "sáu mươi", "bảy mươi", "tám mươi", "chín mươi",
    }
    # Thêm: detect pattern "scale_word + unit_word" như "triệu hai", "tỷ hai"
    # → cần suffix scan để parse đúng fraction
    _scale_spoken = {"triệu", "củ", "tỉ", "tỷ", "tỏi", "nghìn", "ngàn"}
    _unit_spoken  = set(_VI_UNITS.keys())

    t_lower = text.lower()
    t_words = t_lower.split()

    has_vi_number = any(
        re.search(r'\b' + re.escape(w) + r'\b', t_lower)
        for w in _unambiguous_vi_num
    )
    # "một tỷ hai", "ba triệu năm" → scale word theo sau bởi unit word
    if not has_vi_number:
        for idx, w in enumerate(t_words):
            if w in _scale_spoken and idx + 1 < len(t_words):
                if t_words[idx + 1] in _unit_spoken:
                    has_vi_number = True
                    break
            # Hoặc unit word theo trước scale word: "một tỷ", "hai triệu"
            if w in _unit_spoken and idx + 1 < len(t_words):
                if t_words[idx + 1] in _scale_spoken:
                    has_vi_number = True
                    break

    if amounts and not has_vi_number:
        # Không có số chữ → lấy kết quả digit ngay, cộng tổng nếu nhiều số
        return sum(amounts) if len(amounts) > 1 else amounts[0]

    # ── Pass 3: suffix scan — tìm cụm số bắt đầu từ word nào ────
    # Xử lý mix chữ+digit: "mua laptop ba mươi lăm triệu 996k"
    # → scan từ "ba": normalize("ba mươi lăm triệu 996k") = "35996k"
    # → pass1("35996k") = "35996000" → 35,996,000 ✅
    _num_words = set(_VI_UNITS) | set(_VI_TENS) | set(_VI_SCALE)

    orig_words = text.lower().strip().split()
    for start in range(len(orig_words)):
        w = orig_words[start]
        if not (w in _num_words or re.match(r'^\d', w)):
            continue
        suffix  = " ".join(orig_words[start:])
        spoken  = _normalize_spoken_amount(suffix)
        if spoken == suffix:
            continue   # normalize không đổi gì → bỏ qua
        # Chạy Pass 1 trên spoken
        t2 = spoken
        t2 = re.sub(r'(\d+[.,]\d+)\s*(triệu|tr)\b',
                    lambda m: str(int(float(m.group(1).replace(',','.')) * 1_000_000)), t2)
        t2 = re.sub(r'(\d+)\s*(triệu|củ|tr)\b',
                    lambda m: str(int(m.group(1)) * 1_000_000), t2)
        t2 = re.sub(r'(\d+[.,]\d+)\s*(k|nghìn|ngàn|ca|cá|kê|ke)\b',
                    lambda m: str(int(float(m.group(1).replace(',','.')) * 1_000)), t2,
                    flags=re.IGNORECASE)
        t2 = re.sub(r'(\d+)\s*(nghìn|ngàn|k|ca|cá|kê|ke)\b',
                    lambda m: str(int(m.group(1)) * 1_000), t2, flags=re.IGNORECASE)
        nums2 = re.findall(r'\d+', t2)
        amts2 = [int(n) for n in nums2 if 1_000 <= int(n) <= 100_000_000_000]
        if amts2:
            return sum(amts2) if len(amts2) > 1 else amts2[0]
        # fallback: spoken là số gọn "1200tr", "1200000000"
        m = re.match(r'^(\d+)(k|tr)?$', spoken)
        if m:
            val = int(m.group(1))
            if m.group(2) == 'k':  val *= 1_000
            if m.group(2) == 'tr': val *= 1_000_000
            if 1_000 <= val <= 100_000_000_000: return val

    # Nếu Pass 2 tìm được nhiều số mà Pass 3 không cải thiện → cộng tổng
    if amounts:
        return sum(amounts) if len(amounts) > 1 else amounts[0]

    # ── Pass 4: phòng thủ — "50 nghìn" còn sót sau normalize ─────
    m2 = re.match(
        r'^(\d+)\s*(' + '|'.join(_SCALE_MAP_INLINE.keys()) + r')\b',
        spoken, re.IGNORECASE)
    if m2:
        val = int(m2.group(1)) * _SCALE_MAP_INLINE.get(m2.group(2).lower(), 1)
        if 1_000 <= val <= 100_000_000_000:
            return val

    return 0


def extract_date(text: str) -> str:
    today = datetime.today()
    t     = text.lower()
    if any(w in t for w in ["hôm nay", "today", "tối nay", "sáng nay", "trưa nay"]):
        return today.strftime("%d/%m/%Y")
    if any(w in t for w in ["hôm qua", "yesterday", "tối qua", "sáng qua"]):
        return (today - timedelta(days=1)).strftime("%d/%m/%Y")
    if "tuần trước" in t:
        return (today - timedelta(weeks=1)).strftime("%d/%m/%Y")
    thu_map = {"hai": 0, "ba": 1, "tư": 2, "năm": 3, "sáu": 4, "bảy": 5, "chủ nhật": 6}
    for thu, offset in thu_map.items():
        if f"thứ {thu}" in t:
            days_ago = (today.weekday() - offset) % 7
            return (today - timedelta(days=days_ago)).strftime("%d/%m/%Y")
    match = re.search(r'\b(\d{1,2})[/\-](\d{1,2})\b', text)
    if match:
        return f"{match.group(1).zfill(2)}/{match.group(2).zfill(2)}/{today.year}"
    return today.strftime("%d/%m/%Y")


# ══════════════════════════════════════════════════════════════════
# 7b. MULTI-TRANSACTION SPLITTER  (chỉ dùng cho manual input)
#
#  Separator: dấu chấm phẩy  ";"  hoặc  dấu gạch chéo  "\"
#  Ví dụ:
#    "cafe 50k; gội đầu 100k; mua giấy vệ sinh 20k"
#    "cafe 50k\ gội đầu 100k\ mua giấy vệ sinh 20k"
#    → ["cafe 50k", "gội đầu 100k", "mua giấy vệ sinh 20k"]
# ══════════════════════════════════════════════════════════════════

def split_multi_transaction(text: str) -> list[str]:
    """
    Tách chuỗi multi-transaction thành list các giao dịch đơn.
    Separator: ;  hoặc  \
    Trả về list 1 phần tử nếu không phát hiện multi.
    """
    raw = text.strip()
    for sep in (';', '\\'):
        if sep in raw:
            parts = [p.strip() for p in raw.split(sep) if p.strip()]
            if len(parts) > 1:
                return parts
    return [raw]


# ══════════════════════════════════════════════════════════════════
# 8. USER CORRECTIONS (feedback loop)
# ══════════════════════════════════════════════════════════════════
def load_user_corrections() -> list[tuple[str, str]]:
    if os.path.exists(USER_DATA_PATH):
        with open(USER_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [(d["text"], d["category"]) for d in data]
    return []


def load_correction_cache() -> dict[str, str]:
    return {t.lower(): c for t, c in load_user_corrections()}


def save_user_correction(text: str, category: str):
    corrections = []
    if os.path.exists(USER_DATA_PATH):
        with open(USER_DATA_PATH, "r", encoding="utf-8") as f:
            corrections = json.load(f)
    corrections = [c for c in corrections if c["text"].lower() != text.lower()]
    corrections.append({"text": text, "category": category,
                         "timestamp": datetime.now().isoformat()})
    os.makedirs("models", exist_ok=True)
    with open(USER_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(corrections, f, ensure_ascii=False, indent=2)


def fuzzy_correction_lookup(text: str, cache: dict[str, str]) -> str | None:
    t = text.lower().strip()

    if t in cache:
        return cache[t]

    for key, cat in cache.items():
        if key in t or t in key:
            return cat

    t_tokens = set(t.split())
    best_overlap, best_cat = 0.0, None
    for key, cat in cache.items():
        k_tokens = set(key.split())
        if not k_tokens:
            continue
        overlap = len(t_tokens & k_tokens) / max(len(t_tokens), len(k_tokens))
        if overlap >= 0.6 and overlap > best_overlap:
            best_overlap, best_cat = overlap, cat

    return best_cat


# ══════════════════════════════════════════════════════════════════
# 9. ENGINE CLASS
# ══════════════════════════════════════════════════════════════════
class ExpenseEngine:
    def __init__(self):
        user_data         = load_user_corrections()
        all_data          = TRAINING_DATA + user_data
        self.model        = load_or_train(all_data)
        self.corr_cache   = load_correction_cache()
        print("✅ Engine sẵn sàng!")

    def parse(self, text: str) -> dict:
        cached_cat = fuzzy_correction_lookup(text, self.corr_cache)
        if cached_cat:
            return {
                "original_text": text,
                "amount"       : extract_amount(text),
                "category"     : cached_cat,
                "confidence"   : 1.0,
                "method"       : "cache",
                "date"         : extract_date(text),
                "rule_scores"  : {},
                "top3"         : [(cached_cat, 1.0)],
            }

        r_scores   = rule_based_score(text)
        n_scores   = nlp_score(self.model, text)
        category, confidence, method = fuse(r_scores, n_scores)

        from collections import defaultdict
        combined: dict[str, float] = defaultdict(float)
        for cat in set(r_scores) | set(n_scores):
            r = r_scores.get(cat, 0.0)
            n = n_scores.get(cat, 0.0)
            combined[cat] = RULE_WEIGHT * r + NLP_WEIGHT * n if r_scores else n
        top3 = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:3]

        return {
            "original_text": text,
            "amount"       : extract_amount(text),
            "category"     : category,
            "confidence"   : confidence,
            "method"       : method,
            "date"         : extract_date(text),
            "rule_scores"  : r_scores,
            "top3"         : top3,
        }

    def retrain_with_correction(self, text: str, correct_category: str):
        save_user_correction(text, correct_category)
        self.corr_cache[text.lower().strip()] = correct_category
        user_data       = load_user_corrections()
        all_data        = TRAINING_DATA + user_data
        self.model      = train_model(all_data)
        print(f"✅ Đã học: '{text}' → {correct_category}")


# ══════════════════════════════════════════════════════════════════
# 10. VOICE COMMAND SCHEME PARSER
# ══════════════════════════════════════════════════════════════════

_VI_UNITS = {
    "không": 0, "một": 1, "hai": 2, "ba": 3, "bốn": 4,
    "năm": 5,   "sáu": 6, "bảy": 7, "tám": 8, "chín": 9,
    # Biến thể đọc số thông dụng
    "lăm": 5,   # "ba mươi lăm" = 35
    "mốt": 1,   # "hai mươi mốt" = 21
    "tư":  4,   # "hai mươi tư"  = 24
    "nhăm": 5,  # biến thể của lăm
    "rưỡi": 5,  # "ba triệu rưỡi" = 3,500,000
}
_VI_TENS = {
    "hai mươi": 20, "ba mươi": 30, "bốn mươi": 40, "năm mươi": 50,
    "sáu mươi": 60, "bảy mươi": 70, "tám mươi": 80, "chín mươi": 90,
    "mười": 10, "mươi": 10,
}
_VI_SCALE = {
    "trăm": 100,
    "nghìn": 1_000, "ngàn": 1_000,
    "k": 1_000,
    "ca": 1_000, "cá": 1_000, "kê": 1_000, "ke": 1_000,
    "lít": 100_000, "chai": 100_000,
    "triệu": 1_000_000, "củ": 1_000_000,
    "tỉ": 1_000_000_000, "tỷ": 1_000_000_000,
    "tỏi": 1_000_000_000,   # PhoWhisper hay nhầm "tỉ" → "tỏi"
}

_AMOUNT_STARTERS = set(_VI_UNITS) | set(_VI_TENS) | set(_VI_SCALE)

_VERBS_ALL  = {"mua", "ăn", "uống", "đi", "gọi", "đặt", "order",
               "trả", "nạp", "đóng", "thanh toán", "chuyển", "sắm"}
_NOUNS_BILL = {"tiền", "phí", "cước", "hóa đơn"}


def _split_compound_tokens(words: list[str]) -> list[str]:
    """
    Tách token ghép digit+scale mà PhoWhisper hay sinh ra.
    "1k" → ["1","k"]  |  "50k" → ["50","k"]  |  "3tr" → ["3","tr"]
    Sau đó xoá artifact "1" / "một" thừa giữa tens-word và scale-word.
    """
    _scale_pat = r'(k|ca|cá|kê|ke|tr|nghìn|ngàn|triệu|củ|lít|chai|trăm|tỷ|tỉ)'

    split: list[str] = []
    for w in words:
        m = re.match(r'^(\d+)\s*' + _scale_pat + r'$', w)
        if m:
            split.append(m.group(1))
            split.append(m.group(2))
        else:
            split.append(w)

    _scale_words  = set(_VI_SCALE.keys())
    _number_words = set(_VI_UNITS.keys()) | set(_VI_TENS.keys())
    result: list[str] = []
    i = 0
    while i < len(split):
        w = split[i]
        if w in ("1", "một") and i > 0 and i + 1 < len(split):
            prev_is_num = result and (result[-1] in _number_words
                                      or re.match(r'^\d+$', result[-1]))
            next_is_scale = split[i + 1] in _scale_words
            if prev_is_num and next_is_scale:
                i += 1
                continue
        result.append(w)
        i += 1

    return result


def _normalize_spoken_amount(text: str) -> str:
    r"""
    Chuyển cụm số nói tiếng Việt → dạng viết gọn.
      "năm mươi ca"      → "50k"
      "năm mươi 1k"      → "50k"   (PhoWhisper artifact)
      "hai trăm nghìn"   → "200k"
      "50 nghìn"         → "50k"   <- FIX: không còn bị bypass sai nữa
      "ba lít"           → "300k"
      "năm triệu"        → "5tr"
      "50k"              → "50k"

    FIX: regex early-return trước đây là:
         ^\d[\d.,]*\s*(?:k|tr|nghìn|triệu|đồng|tỷ|củ|lít|chai)?$
         → "50 nghìn" khớp → trả nguyên → không convert → BUG
    Nay thu hẹp thành ^\d[\d.,]*\s*(?:k|tr)?$
    → chỉ bypass khi đã là số gọn ("50k", "3tr", "200000")
    """
    t = text.lower().strip()

    # Chỉ bypass khi ĐÃ ở dạng số gọn không có đơn vị tiếng Việt dài
    if re.match(r'^\d[\d.,]*\s*(?:k|tr)?$', t):
        return t

    total, current = 0, 0
    words = _split_compound_tokens(t.split())
    i, parsed_any = 0, False

    last_major_scale = 0   # track scale lớn nhất đã flush (triệu, tỉ...)

    while i < len(words):
        w   = words[i]
        two = " ".join(words[i:i+2]) if i + 1 < len(words) else ""

        if two in _VI_TENS:
            current += _VI_TENS[two]; i += 2; parsed_any = True
        elif w in _VI_TENS:
            current += _VI_TENS[w];  i += 1; parsed_any = True
        elif w in _VI_UNITS:
            current += _VI_UNITS[w]; i += 1; parsed_any = True
        elif w in _VI_SCALE:
            scale = _VI_SCALE[w]
            if current == 0: current = 1
            if scale >= 1_000:
                total += current * scale; current = 0
                if scale > last_major_scale:
                    last_major_scale = scale
            else:
                current *= scale
            i += 1; parsed_any = True
        elif re.match(r'^\d+$', w):
            current += int(w); i += 1; parsed_any = True
        elif re.match(r'^\d+[.,]\d+$', w):
            # Số thập phân như "1,5" hoặc "1.5" — dùng float
            current += float(w.replace(',', '.')); i += 1; parsed_any = True
        else:
            i += 1

    # Nếu còn current nhỏ sau khi đã flush scale lớn (triệu/tỉ):
    # → đây là shorthand fraction, không phải cộng thẳng
    # "một triệu hai"    : current=2,  last=1tr  → 2 × 100k  = 200,000
    # "một triệu hai mươi": current=20, last=1tr  → 20 × 10k  = 200,000
    # "một triệu hai trăm": current=200,last=1tr  → 200 × 1k  = 200,000
    # Rule: current × (last_major_scale // 10^len(str(int(current))))
    if current > 0 and last_major_scale >= 1_000_000:
        digits = len(str(int(current)))
        fraction_unit = last_major_scale // (10 ** digits)
        if fraction_unit >= 1_000:
            total += current * fraction_unit
        else:
            total += current
    elif current > 0 and last_major_scale == 0 and total == 0 and 100 <= current < 1_000:
        # "trăm" terminal — money shorthand, fraction rule giống triệu/tỉ:
        # "một trăm"        = 100 → 100,000
        # "một trăm hai"    = 102 → 1×100k + 2/10×100k  = 120,000
        # "một trăm hai mươi" = 120 → 1×100k + 20/100×100k = 120,000
        # "hai trăm rưỡi"   = 205 → 2×100k + 5/10×100k  = 250,000
        hundreds  = int(current) // 100       # phần trăm nghìn
        remainder = int(current) % 100        # phần lẻ
        if remainder == 0:
            total = hundreds * 100_000
        else:
            # fraction: remainder/10^digits × 100k
            digits = len(str(remainder))
            fraction_unit = 100_000 // (10 ** digits)
            total = hundreds * 100_000 + remainder * fraction_unit
    else:
        total += current
    if not parsed_any or total == 0:
        return text

    total_int = int(round(total))
    if total_int % 1_000_000 == 0: return f"{total_int // 1_000_000}tr"
    if total_int % 1_000 == 0:     return f"{total_int // 1_000}k"
    return str(total_int)


def _is_amount_start(word: str) -> bool:
    return (word in _AMOUNT_STARTERS
            or re.match(r'^\d', word) is not None)


def parse_voice_scheme(raw: str) -> dict:
    t = re.sub(r'[.,!?;:]', '', raw.lower().strip())
    t = re.sub(r'\s+', ' ', t)
    words = t.split()

    amount_phrase, head = "", t
    window = 6

    best_start     = -1
    best_phrase    = ""

    for start in range(max(0, len(words) - window), len(words)):
        if not _is_amount_start(words[start]):
            continue
        candidate  = " ".join(words[start:])
        normalized = _normalize_spoken_amount(candidate)
        if normalized != candidate and re.match(r'^\d[\d.,]*(k|tr)?$', normalized):
            if best_start == -1:
                best_start  = start
                best_phrase = normalized

    if best_start != -1:
        amount_phrase = best_phrase
        head          = " ".join(words[:best_start]).strip()

    hw = head.split()

    # S1: verb + item + amount
    if hw and hw[0] in _VERBS_ALL and len(hw) >= 2 and amount_phrase:
        item = " ".join(hw[1:])
        return {"text": f"{hw[0]} {item} {amount_phrase}", "scheme": "S1",
                "item": item, "amount_raw": amount_phrase}

    # S2: noun_bill + item + amount
    if hw and hw[0] in _NOUNS_BILL and len(hw) >= 2 and amount_phrase:
        item = " ".join(hw[1:])
        return {"text": f"{hw[0]} {item} {amount_phrase}", "scheme": "S2",
                "item": item, "amount_raw": amount_phrase}

    # S3: keyword + amount
    all_kw = {kw for kws in KEYWORD_RULES.values() for kw, _ in kws}
    if amount_phrase:
        search_in = head if head else t
        matched = next((kw for kw in sorted(all_kw, key=len, reverse=True)
                        if kw in search_in), "")
        if matched or not head:
            return {"text": f"{head} {amount_phrase}".strip(), "scheme": "S3",
                    "item": head, "amount_raw": amount_phrase}

    # Fallback
    return {"text": raw.strip(), "scheme": "fallback",
            "item": raw.strip(), "amount_raw": ""}


def _clean_raw_transcript(text: str) -> str:
    """
    Làm sạch transcript thô từ PhoWhisper trước khi parse/hiển thị.
    Xử lý các artifact phổ biến:
      - "năm mươi 1k"        → "năm mươi k"
      - "cafe năm mươi nghìn." → "cafe năm mươi nghìn"  (dấu câu cuối)
      - "1000002"            → "1tr2"   (PhoWhisper đọc "một triệu hai" thành số)
      - "2500000"            → "2tr5"   (PhoWhisper đọc "hai triệu rưỡi")
    """
    t = text.lower().strip()
    # PhoWhisper thường thêm dấu câu cuối câu (., !, ?) — xoá hết
    t = re.sub(r'[.,!?;:]+$', '', t).strip()

    # ── Bước 1: Chuẩn hoá số có dấu phân cách → số thuần ────────
    # PhoWhisper trên một số version/platform ra "1.000.002" hoặc "1,000,002"
    # → strip dấu phân cách để regex bên dưới nhận dạng được
    # Chỉ xử lý khi pattern là "digit(sep)3digits" lặp lại (dấu ngăn hàng nghìn)
    t = re.sub(r'(\d{1,3})([.,])(\d{3})([.,]\d{3})+',
               lambda m: re.sub(r'[.,]', '', m.group(0)), t)
    # "1.000.002" → "1000002",  "1,000,002" → "1000002"
    t = re.sub(r'(\d{1,3})[.,](\d{3})\b',
               lambda m: m.group(1) + m.group(2)
               if m.group(0).replace(',','').replace('.','').isdigit() else m.group(0),
               t)

    # ── Bước 2: Normalize dạng text "X triệu Y" / "X nghìn Y" ───
    # PhoWhisper transcribe đúng text nhưng _normalize_spoken_amount
    # parse "1 triệu 2" thành 1,000,002 thay vì 1,200,000.
    # Fix: convert "X triệu Y" (Y không có đơn vị, 1-3 chữ số) → "XtrY"
    #   "1 triệu 2"    → "1tr2"    → 1,200,000
    #   "2 triệu 50"   → "2tr50"   → 2,500,000
    #   "1 triệu 200"  → "1tr200"  → 1,200,000  (nhưng "1 triệu 200 nghìn" không bị bắt)
    # Bước 1b: chuẩn hoá "tỏi" → "tỉ" (PhoWhisper nhầm)
    t = re.sub(r'\btỏi\b', 'tỉ', t)

    # Negative lookahead: không áp dụng khi Y theo sau bởi đơn vị hoặc thêm chữ số
    _unit_pat = r'(?!\s*(?:triệu|tr\b|nghìn|ngàn|k\b|tỉ|tỷ|tỏi|củ|lít|chai|\d))'
    t = re.sub(
        r'(\d+)\s*(tỉ|tỷ|tỏi)\s+(\d{1,3})' + _unit_pat,
        lambda m: f"{m.group(1)}tỉ{m.group(3)} ",
        t, flags=re.IGNORECASE)
    t = re.sub(
        r'(\d+)\s*(triệu|củ)\s+(\d{1,3})' + _unit_pat,
        lambda m: f"{m.group(1)}tr{m.group(3)} ",
        t, flags=re.IGNORECASE)
    t = re.sub(
        r'(\d+)\s*(nghìn|ngàn|k)\s+(\d{1,3})' + _unit_pat,
        lambda m: f"{m.group(1)}k{m.group(3)} ",
        t, flags=re.IGNORECASE)
    # "1 trăm 2" → "120k",  "2 trăm 5" → "250k"  (trăm + lẻ không có đơn vị)
    t = re.sub(
        r'(\d+)\s*trăm\s+(\d{1,2})\b(?!\s*(?:nghìn|ngàn|k\b|\d))',
        lambda m: f"{int(m.group(1))*100_000 + int(m.group(2))*10_000}",
        t, flags=re.IGNORECASE)
    t = t.strip()

    # ── Bước 3: Normalize số dạng artifact PhoWhisper (số thuần) ─
    # Case 1 — "1000002" (7+ cs): X triệu + Y lẻ < 1000 → "XtrY"
    # Case 2 — "100002"  (6 cs):  PhoWhisper nhầm triệu → trăm nghìn → "XtrY"
    def _fix_number_artifact(m):
        raw = m.group(0)
        n = int(raw)
        millions  = n // 1_000_000
        remainder = n % 1_000_000
        if millions > 0 and 0 < remainder < 1_000:
            return f"{millions}tr{remainder}"
        hund_k     = n // 100_000
        remainder2 = n % 100_000
        if hund_k > 0 and n < 1_000_000 and 0 < remainder2 < 1_000:
            return f"{hund_k}tr{remainder2}"
        return raw

    t = re.sub(r'\b\d{6,10}\b', _fix_number_artifact, t)

    words  = t.split()
    tokens = _split_compound_tokens(words)

    _scale_words  = set(_VI_SCALE.keys())
    _number_words = set(_VI_UNITS.keys()) | set(_VI_TENS.keys())

    result: list[str] = []
    i = 0
    while i < len(tokens):
        w = tokens[i]
        if w in ("1", "một") and result and i + 1 < len(tokens):
            prev_ok = (result[-1] in _number_words
                       or re.match(r'^\d+$', result[-1]))
            next_ok = tokens[i + 1] in _scale_words
            if prev_ok and next_ok:
                i += 1
                continue
        result.append(w)
        i += 1

    return " ".join(result)

    _scale_words  = set(_VI_SCALE.keys())
    _number_words = set(_VI_UNITS.keys()) | set(_VI_TENS.keys())

    result: list[str] = []
    i = 0
    while i < len(tokens):
        w = tokens[i]
        if w in ("1", "một") and result and i + 1 < len(tokens):
            prev_ok = (result[-1] in _number_words
                       or re.match(r'^\d+$', result[-1]))
            next_ok = tokens[i + 1] in _scale_words
            if prev_ok and next_ok:
                i += 1
                continue
        result.append(w)
        i += 1

    return " ".join(result)

