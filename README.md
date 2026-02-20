# 透析中の血圧予測 AI モデル & 推論アプリ

透析中のパラメータ（除水量・体重変化・液温など）から  
**収縮期血圧・拡張期血圧・脈拍**を推定する多出力回帰モデルと、  
FastAPI + Streamlit による推論アプリケーションです。

---

## 🎯 このプロジェクトで実装したこと

- 医療データの前処理設計（train/val/test分割）
- StandardScalerによる入力・出力の標準化
- PyTorchによる多出力回帰モデルの学習
- 学習済みモデルとスケーラーの保存（artifacts管理）
- FastAPIによる推論API化
- StreamlitによるUI実装
- モデル評価と課題分析

---

## 🏗 システム構成

Colab（学習）

↓

artifacts保存（model.pth / scaler_x.pkl / scaler_t.pkl）

↓

FastAPI（推論API）

↓  

Streamlit（UI）

---

## 🚀 ローカル実行方法

### 1. 仮想環境作成

```bash
python -m venv venv
venv\Scripts\activate

依存関係インストール
pip install -r requirements.txt

