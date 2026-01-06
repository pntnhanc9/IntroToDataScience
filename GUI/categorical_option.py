import os, json
import pandas as pd

# ====== PATHS ======
META_PATH = "../notebooks/modeling/best_model_meta.json"
DATA_PATH = "../data/processed/data_clean.csv"   # đổi nếu data ở chỗ khác
# ===================

need_cols = ["washing_machine_type", "lock_type", "wc_type", "water_type", "area_group"]

# 1) Load meta
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

# 2) Load data
df = pd.read_csv(DATA_PATH)

# 3) Check columns exist
missing_in_df = [c for c in need_cols if c not in df.columns]
if missing_in_df:
    raise ValueError(f"❌ Dataset không có các cột: {missing_in_df}. Hiện có: {list(df.columns)}")

# 4) Build categorical_options
meta.setdefault("categorical_options", {})
for c in need_cols:
    opts = (
        df[c]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    # sort cho dễ chọn (nếu là số dạng string thì vẫn sort theo chuỗi)
    meta["categorical_options"][c] = sorted(opts)

# (khuyến nghị) nếu muốn lưu luôn options district/ward cho fallback:
for c in ["district", "ward"]:
    if c in df.columns:
        opts = df[c].dropna().astype(str).str.strip().unique().tolist()
        meta["categorical_options"][c] = sorted(opts)

# 5) Save back
os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("✅ Đã bổ sung categorical_options vào meta cho:", need_cols)
print("✅ Saved:", META_PATH)
for c in need_cols:
    print(f"- {c}: {len(meta['categorical_options'][c])} options")
