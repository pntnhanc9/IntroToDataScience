import json
import os
import joblib
import pandas as pd
import streamlit as st
import unicodedata

st.set_page_config(page_title="Dự đoán giá phòng trọ", layout="centered")

# =========================
# Paths (theo yêu cầu)
# =========================
BASE_DIR   = "../notebooks/modeling"
MODEL_PATH = os.path.join(BASE_DIR, "best_model.joblib")
META_PATH  = os.path.join(BASE_DIR, "best_model_meta.json")

# =========================
# ✅ MAP TAY: District -> Wards (lọc ward realtime theo district)
# (Chỉ dùng để lọc UI; giá trị gửi vào model vẫn sẽ canonical theo meta)
# =========================
DISTRICT_WARDS = {
    'Quận 1': ['21','bình lợi','bến nghé','bến thành','cô giang','cầu kho','cầu ông lãnh','nguyễn cư trinh','nguyễn thái bình','phạm ngũ lão','tân định','võ thị sáu','đa kao'],
    'Quận 2': ['1','10','11','12','13','14','15','16','17'],
    'Quận 3': ['1','10','11','12','13','14','2','3','4','5','6','7','8','9'],
    'Quận 4': ['1','10','11','12','13','14','15','16','18','2','3','4','5','6','8','9'],
    'Quận 5': ['1','10','11','12','13','14','15','2','3','4','5','6','7','8','9'],
    'Quận 6': ['1','10','11','12','13','14','2','3','4','5','6','7','8','9'],
    'Quận 7': ['bình thuận','phú thuận','phú mỹ','tân hưng','tân kiểng','tân phú','tân phong','tân quy','tân thuận đông','tân thuận tây'],
    'Quận 8': ['1','10','11','12','13','14','15','16','2','3','4','5','6','7','8','9'],
    'Quận 9': ['hiệp phú','long thạnh mỹ','long trường','phú hữu','tăng nhơn phú a','tăng nhơn phú b','trường thạnh'],
    'Quận 10': ['1','10','11','12','13','14','15','2','3','4','5','6','7','8','9'],
    'Quận 11': ['1','10','11','12','13','14','15','16','2','3','4','5','6','7','8','9'],
    'Quận 12': ['an phú đông','hiệp thành','thạnh lộc','thạnh xuân','tân chánh hiệp','tân hưng thuận','tân thới hiệp','tân thới nhất','thới an','trung mỹ tây','đông hưng thuận','đông thạnh','đông thạnh( có thạnh lộc )','đông thạnh ( có thạnh lộc )','đông thạnh ( có thạnh lộc)','đông thạnh (có thạnh lộc)'],
    'Bình Thạnh': ['1','10','11','12','13','14','15','17','19','2','21','22','24','25','26','27','28','3','5','6','7','8'],
    'Bình Tân': ['an lạc','an lạc a','bình hưng hòa','bình hưng hòa a','bình hưng hòa b','bình trị đông','bình trị đông a','bình trị đông b','tân tạo','tân tạo a'],
    'Gò Vấp': ['1','10','11','12','13','14','15','16','17','3','4','5','6','7','8','9'],
    'Tân Bình': ['1','10','11','12','13','14','15','2','3','4','5','6','7','8','9'],
    'Tân Phú': ['hiệp tân','hòa thạnh','phú thạnh','phú thọ hòa','phú trung','sơn kỳ','tân quý','tân thành','tân thới hòa'],
    'Thủ Đức': ['an khánh','bình chiểu','bình thọ','cát lái','hiệp bình chánh','hiệp bình phước','linh chiểu','linh trung','linh tây','linh xuân','linh đông','phước bình','phước long a','phước long b','tam bình','tam phú','thảo điền','thủ thiêm','trường thọ'],
    'Bình Chánh': ['bình chánh','bình hưng','bình lợi','lê minh xuân','phạm văn hai','tân nhựt','tân quý tây','vĩnh lộc a','vĩnh lộc b'],
    'Củ Chi': ['phước hiệp','phước vĩnh an','tân an hội','tân phú trung','trung an','trung lập hạ','trung lập thượng'],
    'Hóc Môn': ['bà điểm','tân hiệp','tân thới nhì','thới tam thôn','trung chánh','xuân thới sơn'],
    'Nhà Bè': ['hiệp phước','nhà bè','phước kiển','phước lộc','phú xuân','nhơn đức'],
    'Phú Nhuận': ['1','10','11','12','13','14','15','17','2','3','4','5','7','8','9'],
}

# District value gửi vào model (đúng như dataset/model đã train)
DISTRICT_VALUE_MAP = {
    "Quận 1": "1", "Quận 2": "2", "Quận 3": "3", "Quận 4": "4", "Quận 5": "5", "Quận 6": "6",
    "Quận 7": "7", "Quận 8": "8", "Quận 9": "9", "Quận 10": "10", "Quận 11": "11", "Quận 12": "12",
    "Bình Thạnh": "bình thạnh", "Bình Tân": "bình tân", "Gò Vấp": "gò vấp", "Tân Bình": "tân bình",
    "Tân Phú": "tân phú", "Thủ Đức": "thủ đức", "Bình Chánh": "bình chánh", "Củ Chi": "củ chi",
    "Hóc Môn": "hóc môn", "Nhà Bè": "nhà bè", "Phú Nhuận": "phú nhuận",
}

# =========================
# Helpers
# =========================
def format_vnd(x: float) -> str:
    try:
        return f"{float(x):,.0f} đ".replace(",", ".")
    except Exception:
        return str(x)

def safe_strip_list(xs):
    return [str(x).strip() for x in (xs or [])]

def safe_strip_dict_keys(d):
    return {str(k).strip(): v for k, v in (d or {}).items()}

def norm(s: str) -> str:
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKC", s)
    return s

def pick_canonical(user_value, options_list):
    """
    Trả về đúng string nằm trong options_list (theo meta) nếu match được khi normalize.
    Nếu không match thì giữ nguyên user_value.
    """
    if not options_list:
        return user_value
    m = {norm(x): x for x in options_list}
    return m.get(norm(user_value), user_value)

@st.cache_resource
def load_artifacts(model_path: str, meta_path: str):
    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

# =========================
# UI
# =========================
st.title("🏠 Dự đoán giá phòng trọ (Stacking Ensemble)")

if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
    st.error(
        "Không tìm thấy file model/meta.\n\n"
        f"- Model: `{MODEL_PATH}`\n"
        f"- Meta : `{META_PATH}`\n\n"
        "Hãy kiểm tra lại đường dẫn hoặc đảm bảo 2 file nằm đúng vị trí."
    )
    st.stop()

model, meta = load_artifacts(MODEL_PATH, META_PATH)

feature_cols = safe_strip_list(meta.get("feature_cols", []))
numeric_features = set(safe_strip_list(meta.get("numeric_features", [])))
binary_features  = set(safe_strip_list(meta.get("binary_features", [])))
categorical_features = set(safe_strip_list(meta.get("categorical_features", [])))
categorical_options = safe_strip_dict_keys(meta.get("categorical_options", {}))  # col -> [options...]

# strip options values
for k in list(categorical_options.keys()):
    if isinstance(categorical_options[k], list):
        categorical_options[k] = [str(v).strip() for v in categorical_options[k]]

# =========================
# ✅ CHỐT THEO YÊU CẦU CỦA CÔ:
# - amenity_score/location_score/security_score tự tính, KHÔNG CHO CHỌN
# - has_private_facilities là binary (checkbox), KHÔNG phải numeric nhập tay
# =========================
DERIVED = {"amenity_score", "location_score", "security_score"}

# fix feature typing
numeric_features = set([c for c in numeric_features if c not in DERIVED])
# area là numeric chính (có thể giữ thêm numeric khác nếu cô muốn)
# nhưng theo chốt: derived không nhập => đã loại khỏi numeric
if "has_private_facilities" in numeric_features:
    numeric_features.remove("has_private_facilities")

binary_features.add("has_private_facilities")  # ép về checkbox

district_present = "district" in feature_cols
ward_present     = "ward" in feature_cols

st.caption(f"Đang dùng model: `{MODEL_PATH}` | Số feature: {len(feature_cols)}")

# =========================
# District -> Ward (lọc realtime) - outside form
# =========================
st.subheader("📍 Khu vực")

def on_district_change():
    st.session_state["ward_selected"] = "-- Chọn --"

district_display = None
ward_selected = None

if district_present:
    district_opts = ["-- Chọn --"] + list(DISTRICT_WARDS.keys())
    district_display = st.selectbox(
        "district",
        options=district_opts,
        index=0,
        key="district_display",
        on_change=on_district_change
    )

if ward_present:
    if district_display and district_display != "-- Chọn --":
        ward_opts = ["-- Chọn --"] + DISTRICT_WARDS.get(district_display, [])
    else:
        ward_opts = ["-- Chọn --"]
    ward_selected = st.selectbox(
        "ward",
        options=ward_opts,
        index=0,
        key="ward_selected"
    )

st.divider()

# =========================
# ❗Bắt buộc categorical phải có options (trừ district/ward vì UI dùng map tay)
# =========================
cat_cols_needed = [c for c in feature_cols if (c in categorical_features) and (c not in ("district", "ward"))]
missing_opt_cols = [c for c in cat_cols_needed if c not in categorical_options or not categorical_options[c]]
if missing_opt_cols:
    st.error(
        "Thiếu danh sách options cho các cột categorical sau trong meta (`categorical_options`):\n\n"
        + "\n".join([f"- {c}" for c in missing_opt_cols]) +
        "\n\n=> Hãy bổ sung `categorical_options` cho các cột này (từ dataset) rồi chạy lại app."
    )
    st.stop()

# =========================
# Inputs in a form
# =========================
with st.form("input_form"):
    st.subheader("🧾 Nhập thông tin đặc trưng")

    inputs = {}

    # district/ward đưa vào inputs đúng giá trị model đã train
    if district_present:
        inputs["district"] = (
            DISTRICT_VALUE_MAP.get(district_display)
            if district_display and district_display != "-- Chọn --"
            else "-- Chọn --"
        )
    if ward_present:
        inputs["ward"] = ward_selected

    # Numeric (không render derived)
    with st.expander("1) Numeric", expanded=True):
        for col in feature_cols:
            if col in ("district", "ward"):
                continue
            if col in DERIVED:
                continue
            if col in numeric_features:
                inputs[col] = st.number_input(
                    label=col, value=0.0, step=0.1, format="%.4f"
                )

    # Boolean -> lưu 0/1 luôn
    with st.expander("2) Boolean (0/1)", expanded=True):
        for col in feature_cols:
            if col in ("district", "ward"):
                continue
            if col in DERIVED:
                continue
            if col in binary_features:
                inputs[col] = 1 if st.checkbox(label=col, value=False, key=f"bin_{col}") else 0

    # Categorical (không render district/ward vì đã ở trên)
    with st.expander("3) Categorical", expanded=True):
        for col in feature_cols:
            if col in ("district", "ward"):
                continue
            if col in DERIVED:
                continue
            if col in categorical_features:
                opts = ["-- Chọn --"] + list(categorical_options[col])
                inputs[col] = st.selectbox(label=col, options=opts, index=0, key=f"cat_{col}")

    submitted = st.form_submit_button("🔮 Dự đoán")

# =========================
# Predict
# =========================
if submitted:
    # Validate district/ward
    if district_present and (inputs.get("district") in (None, "", "-- Chọn --")):
        st.warning("Bạn chưa chọn **district**.")
        st.stop()

    if ward_present and (inputs.get("ward") in (None, "", "-- Chọn --")):
        st.warning("Bạn chưa chọn **ward**.")
        st.stop()

    # Validate all categorical selectboxes (except district/ward)
    not_selected = [c for c in cat_cols_needed if inputs.get(c) == "-- Chọn --"]
    if not_selected:
        st.warning("Bạn chưa chọn: " + ", ".join(not_selected))
        st.stop()

    # =========================
    # ✅ TỰ TÍNH DERIVED FEATURES (KHÔNG CHO NHẬP TAY)
    # =========================
    # amenity_score: tổng tiện nghi (0/1)
    amenity_cols = [
        "has_air_conditioner", "has_kitchen", "has_fridge", "has_wardrobe", "has_bed",
        "has_mattress", "has_sofa", "has_window", "has_wifi", "has_parking",
        "has_balcony", "has_private_facilities"
    ]
    inputs["amenity_score"] = sum(int(inputs.get(c, 0)) for c in amenity_cols)

    # location_score: tổng near_*
    loc_cols = ["near_school", "near_market", "near_supermarket"]
    inputs["location_score"] = sum(int(inputs.get(c, 0)) for c in loc_cols)

    # security_score: mặc định 1
    inputs["security_score"] = 1

    # =========================
    # Chuẩn hoá type + canonical categorical theo meta
    # =========================
    inputs = {str(k).strip(): v for k, v in inputs.items()}

    # ép numeric
    for c in numeric_features:
        if c in inputs:
            try:
                inputs[c] = float(inputs[c])
            except:
                inputs[c] = 0.0

    # ép binary (đảm bảo 0/1)
    for c in binary_features:
        if c in inputs:
            inputs[c] = 1 if int(inputs[c]) != 0 else 0

    # ép string cho categorical + canonicalize theo options (nếu có)
    for c in categorical_features.union({"district", "ward"}):
        if c in inputs:
            inputs[c] = str(inputs[c]).strip()
            if c in categorical_options and isinstance(categorical_options[c], list) and categorical_options[c]:
                inputs[c] = pick_canonical(inputs[c], categorical_options[c])

    # Build dataframe đúng thứ tự
    X_new = pd.DataFrame([[inputs.get(c) for c in feature_cols]], columns=feature_cols)

    try:
        y_pred = model.predict(X_new)[0]
        st.success(f"✅ Giá dự đoán (price): **{format_vnd(y_pred)}**")

        with st.expander("Xem dữ liệu đã nhập (kèm score tự tính)"):
            st.write("Computed scores:")
            st.write({
                "amenity_score": inputs.get("amenity_score"),
                "location_score": inputs.get("location_score"),
                "security_score": inputs.get("security_score"),
            })
            st.dataframe(X_new)

    except Exception as e:
        st.error("❌ Lỗi khi dự đoán: " + str(e))
        with st.expander("Debug"):
            st.write("district_display:", district_display)
            st.write("district_value:", inputs.get("district"))
            st.write("ward_selected:", ward_selected)
            st.write("X_new:", X_new)
