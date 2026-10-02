import json
from pathlib import Path
import joblib
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Nạp dữ liệu Breast Cancer Wisconsin
bundle = load_breast_cancer(as_frame=True)
X = bundle.data
y = bundle.target

# 2. Chia dữ liệu 80/20 với Stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 3. Tạo Pipeline chuẩn hóa + SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    (
        'svc',
        SVC(
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            random_state=42,
        ),
    ),
])

# 4. Tìm siêu tham số tối ưu (GridSearchCV)
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': ['scale', 0.001, 0.01, 0.1],
}

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='recall_macro',
    cv=CV,
    n_jobs=-1,
    refit=True,
)

search.fit(X_train, y_train)
model = search.best_estimator_

# 5. Đánh giá mô hình
pred = model.predict(X_test)
print('--- KẾT QUẢ ĐÁNH GIÁ ---')
print('Siêu tham số tốt nhất:', search.best_params_)
print('\nConfusion Matrix:\n', confusion_matrix(y_test, pred, labels=[0, 1]))
print(
    '\nClassification Report:\n',
    classification_report(
        y_test, pred, labels=[0, 1], target_names=['malignant', 'benign']
    ),
)
print(
    'Sensitivity (Độ nhạy ác tính):', recall_score(y_test, pred, pos_label=0)
)

# 6. Luu Artifacts (Model & Metadata)
ARTIFACT_DIR = Path('artifacts')
ARTIFACT_DIR.mkdir(exist_ok=True)

joblib.dump(model, ARTIFACT_DIR / 'breast_cancer_svm.joblib')

metadata = {
    'model_name': 'breast-cancer-svm-rbf',
    'model_version': '1.0.0',
    'feature_names': list(X.columns),
    'class_mapping': {'0': 'malignant', '1': 'benign'},
    'best_params': search.best_params_,
    'sklearn_version': sklearn.__version__,
    'warning': 'Educational use only - not a medical diagnosis.',
}

(ARTIFACT_DIR / 'metadata.json').write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8'
)

print('\n Output artifacts: artifacts/breast_cancer_svm.joblib & metadata.json')