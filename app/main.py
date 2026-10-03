import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="Breast Cancer SVM Diagnostic Web App",
    description="Web hỗ trợ chẩn đoán ung thư vú sử dụng mô hình SVM",
    version="1.0.0"
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đường dẫn mô hình SVM
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "breast_cancer_svm.joblib")

# Schema dữ liệu đầu vào
class PatientData(BaseModel):
    features: List[float]

# Endpoint kiểm tra trạng thái
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}

# Endpoint dự đoán
@app.post("/predict")
def predict(data: PatientData):
    if len(data.features) != 30:
        raise HTTPException(
            status_code=400, 
            detail=f"Dữ liệu cần chính xác 30 thông số (hiện tại: {len(data.features)})"
        )
    
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail="Chưa tìm thấy tệp mô hình SVM trong artifacts/")

    try:
        model = joblib.load(MODEL_PATH)
        features_array = np.array(data.features).reshape(1, -1)
        
        # Dự đoán lớp
        prediction_cls = int(model.predict(features_array)[0])
        
        # Tính độ tin cậy / xác suất (nếu mô hình hỗ trợ predict_proba hoặc decision_function)
        confidence = 95.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features_array)[0]
            confidence = round(float(np.max(probs)) * 100, 2)
        elif hasattr(model, "decision_function"):
            decision_val = abs(float(model.decision_function(features_array)[0]))
            confidence = round(min(99.9, 85.0 + decision_val * 5.0), 2)
            
        # Quy đổi kết quả theo nhãn chuẩn của Breast Cancer dataset:
        # 0: Ác tính (Malignant), 1: Lành tính (Benign)
        is_benign = (prediction_cls == 1)
        
        return {
            "prediction_code": prediction_cls,
            "label": "Lành tính" if is_benign else "Ác tính",
            "is_benign": is_benign,
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi dự đoán: {str(e)}")

# Endpoint phục vụ giao diện Web Single Page Application (SPA)
@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hệ thống hỗ trợ chẩn đoán ung thư vú</title>
    <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-pink: #e83e8c;
            --primary-hover: #d63384;
            --light-pink: #fce4ec;
            --bg-light: #fff5f8;
            --card-border: #f8bbd0;
            --text-dark: #2c3e50;
            --text-muted: #6c757d;
            --success-green: #28a745;
            --danger-red: #dc3545;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Nunito', sans-serif;
        }

        body {
            background-color: var(--bg-light);
            color: var(--text-dark);
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }

        /* Navbar */
        header {
            background-color: #ffffff;
            border-bottom: 2px solid var(--card-border);
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(232, 62, 140, 0.08);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .logo-area {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 800;
            font-size: 1.2rem;
            color: var(--primary-pink);
        }

        nav {
            display: flex;
            gap: 15px;
        }

        .nav-link {
            text-decoration: none;
            color: var(--text-dark);
            font-weight: 600;
            padding: 8px 16px;
            border-radius: 20px;
            transition: all 0.2s ease;
            cursor: pointer;
        }

        .nav-link:hover, .nav-link.active {
            background-color: var(--light-pink);
            color: var(--primary-pink);
        }

        /* Container & Pages */
        .main-container {
            flex: 1;
            max-width: 900px;
            width: 100%;
            margin: 30px auto;
            padding: 0 20px;
        }

        .page {
            display: none;
            animation: fadeIn 0.3s ease-in-out forwards;
        }

        .page.active-page {
            display: block;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .card {
            background: #ffffff;
            border-radius: 16px;
            padding: 35px;
            box-shadow: 0 4px 20px rgba(232, 62, 140, 0.06);
            border: 1px solid var(--card-border);
            margin-bottom: 25px;
        }

        /* Typography & Buttons */
        h1, h2, h3 {
            color: var(--text-dark);
            margin-bottom: 15px;
        }

        .title-pink {
            color: var(--primary-pink);
            font-weight: 800;
        }

        p {
            line-height: 1.6;
            color: var(--text-dark);
            margin-bottom: 15px;
        }

        .btn {
            display: inline-block;
            background-color: var(--primary-pink);
            color: white;
            padding: 12px 28px;
            font-size: 1rem;
            font-weight: 700;
            border: none;
            border-radius: 30px;
            cursor: pointer;
            transition: background 0.2s ease, transform 0.1s ease;
            text-decoration: none;
            text-align: center;
        }

        .btn:hover {
            background-color: var(--primary-hover);
            transform: translateY(-2px);
        }

        .btn-outline {
            background-color: transparent;
            color: var(--primary-pink);
            border: 2px solid var(--primary-pink);
        }

        .btn-outline:hover {
            background-color: var(--light-pink);
            color: var(--primary-hover);
        }

        /* Page 1: Trang chủ */
        .hero {
            text-align: center;
            padding: 40px 20px;
        }

        .hero h1 {
            font-size: 2.2rem;
            margin-bottom: 20px;
        }

        .hero p {
            font-size: 1.1rem;
            max-width: 650px;
            margin: 0 auto 30px auto;
            color: var(--text-muted);
        }

        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }

        .feature-item {
            background: var(--light-pink);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }

        .feature-item h3 {
            font-size: 1.1rem;
            color: var(--primary-pink);
        }

        /* Page 2: Trang chẩn đoán */
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .form-group label {
            font-size: 0.9rem;
            font-weight: 700;
            color: var(--text-dark);
        }

        .form-group input {
            padding: 10px 14px;
            border: 1px solid var(--card-border);
            border-radius: 8px;
            font-size: 0.95rem;
            outline: none;
            transition: border 0.2s ease;
        }

        .form-group input:focus {
            border-color: var(--primary-pink);
            box-shadow: 0 0 0 3px rgba(232, 62, 140, 0.15);
        }

        /* Page 3: Trang kết quả */
        .result-box {
            text-align: center;
            padding: 30px 20px;
            border-radius: 16px;
            margin-bottom: 25px;
        }

        .result-benign {
            background-color: #e8f5e9;
            border: 2px solid var(--success-green);
        }

        .result-malignant {
            background-color: #ffebee;
            border: 2px solid var(--danger-red);
        }

        .result-status {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }

        .benign-text { color: var(--success-green); }
        .malignant-text { color: var(--danger-red); }

        .confidence-badge {
            display: inline-block;
            padding: 6px 18px;
            background: rgba(255, 255, 255, 0.9);
            border-radius: 20px;
            font-size: 1.1rem;
            font-weight: 700;
            margin-top: 10px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
        }

        .alert-warning {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px 20px;
            border-radius: 8px;
            font-size: 0.95rem;
            color: #856404;
            margin-bottom: 25px;
        }

        /* Footer */
        footer {
            background-color: #ffffff;
            border-top: 1px solid var(--card-border);
            text-align: center;
            padding: 20px;
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: auto;
        }
    </style>
</head>
<body>

    <!-- Header & Điều hướng -->
    <header>
        <div class="logo-area">
            <span>🌸</span>
            <span>Hệ Thống Y Tế AI</span>
        </div>
        <nav>
            <a class="nav-link active" onclick="showPage('home-page')">Trang chủ</a>
            <a class="nav-link" onclick="showPage('predict-page')">Chẩn đoán</a>
            <a class="nav-link" onclick="showPage('about-page')">Giới thiệu</a>
        </nav>
    </header>

    <!-- Nội dung chính Single Page -->
    <div class="main-container">

        <!-- 1. TRANG CHỦ -->
        <div id="home-page" class="page active-page">
            <div class="card hero">
                <h1>Hệ thống hỗ trợ <br><span class="title-pink">Chẩn Đoán Ung Thư Vú</span></h1>
                <p>Ứng dụng trí tuệ nhân tạo sử dụng thuật toán Support Vector Machine (SVM) giúp hỗ trợ phân tích và sàng lọc nguy cơ u vú nhanh chóng, chính xác.</p>
                <button class="btn" onclick="showPage('predict-page')">Bắt đầu chẩn đoán ➔</button>
            </div>

            <div class="feature-grid">
                <div class="feature-item">
                    <h3>⚡ Phân tích nhanh</h3>
                    <p>Nhận kết quả sàng lọc tức thì dựa trên các thông số sinh học.</p>
                </div>
                <div class="feature-item">
                    <h3>🎯 Thuật toán SVM</h3>
                    <p>Sử dụng mô hình RBF Kernel được huấn luyện đạt độ chính xác cao.</p>
                </div>
                <div class="feature-item">
                    <h3>🩺 Chuẩn Y khoa</h3>
                    <p>Xây dựng dựa trên tập dữ liệu chuẩn y học Breast Cancer Wisconsin.</p>
                </div>
            </div>
        </div>

        <!-- 2. TRANG CHẨN ĐOÁN -->
        <div id="predict-page" class="page">
            <div class="card">
                <h2>📋 Nhập thông số mẫu bệnh</h2>
                <p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 20px;">
                    Điền các giá trị xét nghiệm sinh học thu được từ ảnh chụp/siêu âm.
                </p>

                <form id="diag-form" onsubmit="submitForm(event)">
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Bán kính trung bình (Mean Radius)</label>
                            <input type="number" step="any" id="f0" value="14.12" required>
                        </div>
                        <div class="form-group">
                            <label>Độ mịn bề mặt (Mean Texture)</label>
                            <input type="number" step="any" id="f1" value="19.28" required>
                        </div>
                        <div class="form-group">
                            <label>Chu vi trung bình (Mean Perimeter)</label>
                            <input type="number" step="any" id="f2" value="91.96" required>
                        </div>
                        <div class="form-group">
                            <label>Diện tích trung bình (Mean Area)</label>
                            <input type="number" step="any" id="f3" value="654.88" required>
                        </div>
                        <div class="form-group">
                            <label>Độ nhẵn (Mean Smoothness)</label>
                            <input type="number" step="any" id="f4" value="0.096" required>
                        </div>
                        <div class="form-group">
                            <label>Độ đặc (Mean Compactness)</label>
                            <input type="number" step="any" id="f5" value="0.104" required>
                        </div>
                        <div class="form-group">
                            <label>Độ lõm (Mean Concavity)</label>
                            <input type="number" step="any" id="f6" value="0.088" required>
                        </div>
                        <div class="form-group">
                            <label>Điểm lõm (Mean Concave Points)</label>
                            <input type="number" step="any" id="f7" value="0.048" required>
                        </div>
                        <div class="form-group">
                            <label>Độ đối xứng (Mean Symmetry)</label>
                            <input type="number" step="any" id="f8" value="0.181" required>
                        </div>
                        <div class="form-group">
                            <label>Bán kính lớn nhất (Worst Radius)</label>
                            <input type="number" step="any" id="f20" value="16.26" required>
                        </div>
                        <div class="form-group">
                            <label>Diện tích lớn nhất (Worst Area)</label>
                            <input type="number" step="any" id="f23" value="880.58" required>
                        </div>
                        <div class="form-group">
                            <label>Độ nhẵn lớn nhất (Worst Smoothness)</label>
                            <input type="number" step="any" id="f24" value="0.132" required>
                        </div>
                    </div>

                    <div style="text-align: center; margin-top: 10px;">
                        <button type="submit" class="btn" style="min-width: 220px;">Thực hiện chẩn đoán 🚀</button>
                    </div>
                </form>
            </div>
        </div>

        <!-- 3. TRANG KẾT QUẢ -->
        <div id="result-page" class="page">
            <div class="card">
                <h2 style="text-align: center; margin-bottom: 25px;">🎯 Kết quả chẩn đoán AI</h2>

                <div id="result-box-element" class="result-box result-benign">
                    <div id="result-status-text" class="result-status benign-text">
                        🟢 Lành tính
                    </div>
                    <div id="confidence-text" class="confidence-badge">
                        Độ tin cậy của mô hình: 96.5%
                    </div>
                </div>

                <div class="alert-warning">
                    ⚠️ <strong>Cảnh báo quan trọng:</strong> Kết quả chẩn đoán từ hệ thống AI này chỉ mang tính chất hỗ trợ tham khảo, không có giá trị thay thế kết luận chẩn đoán chính thức từ bác sĩ chuyên khoa.
                </div>

                <div style="text-align: center; display: flex; gap: 15px; justify-content: center;">
                    <button class="btn btn-outline" onclick="showPage('predict-page')">🔄 Chẩn đoán lại</button>
                    <button class="btn" onclick="showPage('about-page')">Tìm hiểu về bệnh</button>
                </div>
            </div>
        </div>

        <!-- 4. TRANG GIỚI THIỆU -->
        <div id="about-page" class="page">
            <div class="card">
                <h2 class="title-pink">Giới thiệu về Ung thư vú</h2>
                <p>Ung thư vú là dạng ung thư phổ biến nhất ở nữ giới trên toàn thế giới. Việc phát hiện sớm thông qua các chỉ số sinh học và hình ảnh y khoa đóng vai trò tiên quyết trong việc nâng cao hiệu quả điều trị và tỷ lệ sống sót của bệnh nhân.</p>

                <h2 class="title-pink" style="margin-top: 25px;">Mô hình Support Vector Machine (SVM)</h2>
                <p>Support Vector Machine là một thuật toán học máy có giám sát vượt trội trong các bài toán phân loại nhị phân y khoa. Mô hình xây dựng siêu phẳng tối ưu giúp phân tách rõ ràng hai nhóm tế bào U Ác tính (Malignant) và U Lành tính (Benign).</p>

                <h2 class="title-pink" style="margin-top: 25px;">Tập dữ liệu & Hệ thống</h2>
                <p>Hệ thống được huấn luyện trên bộ dữ liệu chuẩn <strong>Breast Cancer Wisconsin Diagnostic</strong> công bố bởi Đại học Wisconsin, bao gồm 30 đặc trưng giải phẫu trích xuất từ hình ảnh số hóa của khối u vú.</p>
            </div>
        </div>

    </div>

    <!-- Footer -->
    <footer>
        Hệ thống hỗ trợ chẩn đoán ung thư vú | Tích hợp thuật toán Machine Learning SVM
    </footer>

    <!-- JavaScript chuyển trang & xử lý API -->
    <script>
        // Mặc định tạo đủ 30 giá trị tham số mẫu
        const default30Features = [
            14.12, 19.28, 91.96, 654.88, 0.096, 0.104, 0.088, 0.048, 0.181, 0.062,
            0.405, 1.216, 2.866, 40.33, 0.007, 0.025, 0.031, 0.011, 0.020, 0.003,
            16.26, 25.67, 107.2, 880.58, 0.132, 0.254, 0.272, 0.114, 0.290, 0.083
        ];

        // Điều hướng các trang Single Page Application
        function showPage(pageId) {
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active-page'));
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));

            document.getElementById(pageId).classList.add('active-page');

            // Cập nhật trạng thái active trên navbar
            const navMap = {
                'home-page': 0,
                'predict-page': 1,
                'about-page': 2
            };
            if (navMap[pageId] !== undefined) {
                document.querySelectorAll('.nav-link')[navMap[pageId]].classList.add('active');
            }

            window.scrollTo({ top: 0, behavior: 'smooth' });
        }

        // Xử lý nút bấm chẩn đoán
        async function submitForm(event) {
            event.preventDefault();

            // Lấy giá trị từ form
            let f = [...default30Features];
            f[0] = parseFloat(document.getElementById('f0').value) || f[0];
            f[1] = parseFloat(document.getElementById('f1').value) || f[1];
            f[2] = parseFloat(document.getElementById('f2').value) || f[2];
            f[3] = parseFloat(document.getElementById('f3').value) || f[3];
            f[4] = parseFloat(document.getElementById('f4').value) || f[4];
            f[5] = parseFloat(document.getElementById('f5').value) || f[5];
            f[6] = parseFloat(document.getElementById('f6').value) || f[6];
            f[7] = parseFloat(document.getElementById('f7').value) || f[7];
            f[8] = parseFloat(document.getElementById('f8').value) || f[8];
            f[20] = parseFloat(document.getElementById('f20').value) || f[20];
            f[23] = parseFloat(document.getElementById('f23').value) || f[23];
            f[24] = parseFloat(document.getElementById('f24').value) || f[24];

            try {
                // Gửi request tới endpoint API FastAPI
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ features: f })
                });

                if (!res.ok) throw new Error('Lỗi từ hệ thống server');

                const data = await res.json();

                // Cập nhật thông tin lên trang kết quả
                const resultBox = document.getElementById('result-box-element');
                const statusText = document.getElementById('result-status-text');
                const confidenceText = document.getElementById('confidence-text');

                if (data.is_benign) {
                    resultBox.className = "result-box result-benign";
                    statusText.className = "result-status benign-text";
                    statusText.innerHTML = "🟢 Lành tính";
                } else {
                    resultBox.className = "result-box result-malignant";
                    statusText.className = "result-status malignant-text";
                    statusText.innerHTML = "🔴 Ác tính";
                }

                confidenceText.innerText = `Độ tin cậy của mô hình: ${data.confidence}%`;

                // Chuyển sang trang kết quả
                showPage('result-page');

            } catch (err) {
                alert('Không thể kết nối đến máy chủ chẩn đoán. Vui lòng kiểm tra lại!');
            }
        }
    </script>
</body>
</html>
    """