import os
import requests
import pandas as pd
import streamlit as st

# ==============================
# APIエンドポイント（フォールバック）
# ==============================
DEFAULT_API_BASE = "http://127.0.0.1:8000"
api_base = os.getenv("API_BASE", DEFAULT_API_BASE)
try:
    api_base = st.secrets.get("API_BASE", api_base)
except Exception:
    pass
API_BASE = api_base

# ==============================
# 固定：説明変数 / 目的変数
# ==============================
# ※ 列名は学習時と完全一致（「下肢アップ」は末尾スペースなし）
FEATURE_NAMES = [
    "DW",
    "CTR",
    "50%TZ20ml",
    "リズミック",
    "レニベース",
    "積算除水量",
    "総除水量",
    "残し量",
    "下肢アップ",
    "液温",
]
TARGET_COLUMNS = ["収縮期血圧", "拡張期血圧", "脈拍"]

# ---- サイドバー初期値と範囲（指定どおり）----
DEFAULT_VALUES = {
    "DW": 50.0,
    "CTR": 50.0,
    "50%TZ20ml": 0.0,
    "リズミック": 0.0,     # 0/1
    "レニベース": 0.0,     # 0/1
    "積算除水量": 2000.0,
    "総除水量": 2000.0,
    "残し量": 0.0,
    "下肢アップ": 0.0,     # 0/1
    "液温": 36.0,
}
RANGES = {
    "DW": (30.0, 120.0, 0.1),
    "CTR": (20.0, 70.0, 0.1),
    "50%TZ20ml": (0.0, 5.0, 1.0),
    "リズミック": (0.0, 5.0, 0.5),
    "レニベース": (0.0, 5.0, 0.25),
    "積算除水量": (0.0, 6000.0, 10.0),
    "総除水量": (0.0, 6000.0, 100.0),
    "残し量": (0.0, 3000.0, 100.0),
    "下肢アップ": (0.0, 1.0, 1.0),
    "液温": (34.0, 38.0, 0.1),
}

# ==============================
# 画面設定
# ==============================
st.set_page_config(page_title="透析中の血圧予測アプリ", layout="wide")
st.title("透析中の血圧予測アプリ")
st.caption("サイドバーに説明変数を入力 → 予測ボタン → 目的変数3つを表示（保存なし）")

# ==============================
# サイドバー：入力フォーム
# ==============================
with st.sidebar:
    st.header("説明変数の入力")
    inputs = {}
    for name in FEATURE_NAMES:
        vmin, vmax, step = RANGES.get(name, (-1e9, 1e9, 1.0))
        default = float(DEFAULT_VALUES.get(name, 0.0))
        val = st.number_input(
            label=name,
            min_value=float(vmin),
            max_value=float(vmax),
            value=default,
            step=float(step),
            key=f"sb_{name}",
        )
        inputs[name] = val

    predict_btn = st.button("この条件で予測する", type="primary")

# ==============================
# 中央：入力プレビュー / 予測結果
# ==============================
# プレビューを広めに確保
col_preview, col_result = st.columns([1.8, 1.0])

with col_preview:
    st.subheader("入力内容プレビュー")
    df_one = pd.DataFrame([[inputs[n] for n in FEATURE_NAMES]], columns=FEATURE_NAMES)
    st.dataframe(df_one, use_container_width=True, height=160)

with col_result:
    st.subheader("予測結果")
    if predict_btn:
        try:
            # --- ご指定どおり {"instances":[{...}]} 形式で送信 ---
            url = f"{API_BASE}/predict"
            payload = {"instances": [inputs]}
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code != 200:
                raise RuntimeError(f"APIエラー: {resp.status_code} - {resp.text}")
            js = resp.json()

            # --- 返却predictionsを3列に正規化（dict/list両対応） ---
            preds = js.get("predictions", [])
            rows = []
            if len(preds) > 0 and isinstance(preds[0], dict):
                for d in preds:
                    rows.append([
                        d.get(TARGET_COLUMNS[0], 0),
                        d.get(TARGET_COLUMNS[1], 0),
                        d.get(TARGET_COLUMNS[2], 0),
                    ])
            elif len(preds) > 0 and isinstance(preds[0], (list, tuple)):
                for arr in preds:
                    rows.append(list(arr)[:3])
            else:
                rows.append(list(preds)[:3])  # flat list想定

            df_pred = pd.DataFrame(rows, columns=TARGET_COLUMNS)
            # 小数点なし表示
            for c in df_pred.columns:
                df_pred[c] = df_pred[c].apply(lambda x: int(round(float(x))))

            st.success("予測が完了しました")
            st.dataframe(df_pred, use_container_width=True, height=120)

            with st.expander("レスポンス詳細（デバッグ用）"):
                st.json(js)

        except Exception as e:
            st.error(f"予測に失敗しました: {e}")
