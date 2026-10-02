import json
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Khởi tạo FastAPI app
app = FastAPI(
    title="Breast Cancer SVM Prediction API",
    version="1.0.0",
    description="API dự đoán ung thư vú sử dụng mô hình Support Vector Machine",
)

# 2. Nạp mô hình và metadata
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "breast_cancer_svm.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    raise RuntimeError("Chưa tìm thấy mô hình hoặc metadata. Hãy chạy scripts/train.py trước!")

model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
feature_names = metadata["feature_names"]


# 3. Định nghĩa Schemas dữ liệu đầu vào/đầu ra
class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description=f"Danh sách {len(feature_names)} giá trị đặc trưng theo thứ tự",
    )


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probability_malignant: float
    probability_benign: float
    model_version: str
    warning: str


def build_vector(payload: PredictionRequest) -> np.ndarray:
    if len(payload.features) != len(feature_names):
        raise HTTPException(
            status_code=400,
            detail=f"Cần truyền đúng {len(feature_names)} đặc trưng. Bạn đang truyền {len(payload.features)}.",
        )
    return np.array(payload.features, dtype=float).reshape(1, -1)


# 4. Các Endpoints API
@app.get("/")
def root():
    return {
        "message": "Breast Cancer SVM API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "features_count": len(feature_names),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    x = build_vector(payload)
    predicted_class = int(model.predict(x)[0])
    probabilities = model.predict_proba(x)[0]

    classes = list(model.named_steps["svc"].classes_)
    p = {int(c): float(v) for c, v in zip(classes, probabilities)}

    return PredictionResponse(
        predicted_class=predicted_class,
        predicted_label=metadata["class_mapping"][str(predicted_class)],
        probability_malignant=p[0],
        probability_benign=p[1],
        model_version=metadata["model_version"],
        warning=metadata["warning"],
    )