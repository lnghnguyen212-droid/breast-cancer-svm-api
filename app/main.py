import os
import sqlite3
import joblib
import numpy as np
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(
    title="Doctor Portal - Breast Cancer SVM Diagnostic System",
    description="Hệ thống hỗ trợ chẩn đoán dành cho Bác sĩ",
    version="2.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đường dẫn DB an toàn trên Render (ưu tiên /tmp nếu chạy trên Linux/Render)
if os.name == 'nt':
    DB_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "doctor_portal.db")
else:
    DB_PATH = "/tmp/doctor_portal.db"

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "breast_cancer_svm.joblib")

def get_db():
    dirname = os.path.dirname(DB_PATH)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS doctors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                hospital TEXT
            );
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patient_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doctor_username TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                patient_name TEXT NOT NULL,
                patient_age INTEGER NOT NULL,
                diagnosis_date TEXT NOT NULL,
                result_label TEXT NOT NULL,
                is_benign INTEGER NOT NULL,
                confidence REAL NOT NULL,
                notes TEXT
            );
        ''')
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"[DB Init Error]: {e}")

init_db()

# Middleware bắt toàn bộ lỗi server trả về JSON để không bị SyntaxError bên Frontend
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Lỗi hệ thống Server: {str(exc)}"}
    )

class DoctorRegister(BaseModel):
    username: str
    password: str
    full_name: str
    hospital: Optional[str] = "Bệnh viện Đa khoa"

class DoctorLogin(BaseModel):
    username: str
    password: str

class DiagnosticRequest(BaseModel):
    doctor_username: str
    patient_id: str
    patient_name: str
    patient_age: int
    notes: Optional[str] = ""
    features: List[float]

@app.get("/health")
def health_check():
    return {"status": "ok", "db_path": DB_PATH, "model_exists": os.path.exists(MODEL_PATH)}

@app.post("/api/register")
def register_doctor(doc: DoctorRegister):
    clean_username = doc.username.strip().lower()
    if not clean_username:
        raise HTTPException(status_code=400, detail="Tên tài khoản không được để trống!")

    conn = get_db()
    cursor = conn.cursor()
    hashed_pwd = pwd_context.hash(doc.password)
    
    try:
        cursor.execute(
            "INSERT INTO doctors (username, password_hash, full_name, hospital) VALUES (?, ?, ?, ?)",
            (clean_username, hashed_pwd, doc.full_name.strip(), doc.hospital.strip() if doc.hospital else "")
        )
        conn.commit()
        return {"status": "success", "message": "Đăng ký thành công!", "username": clean_username}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Tài khoản Bác sĩ này đã tồn tại trên hệ thống!")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi thao tác CSDL: {str(e)}")
    finally:
        cursor.close()
        conn.close()

@app.post("/api/login")
def login_doctor(doc: DoctorLogin):
    clean_username = doc.username.strip().lower()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash, full_name, hospital FROM doctors WHERE username = ?", (clean_username,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not row or not pwd_context.verify(doc.password, row[0]):
        raise HTTPException(status_code=401, detail="Tên đăng nhập hoặc mật khẩu không chính xác!")
        
    return {
        "status": "success",
        "username": clean_username,
        "full_name": row[1],
        "hospital": row[2]
    }

@app.post("/api/diagnose")
def diagnose_and_save(data: DiagnosticRequest):
    if len(data.features) != 30:
        raise HTTPException(status_code=400, detail="Cần nhập đủ 30 chỉ số giải phẫu.")
    
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail="Chưa tìm thấy mô hình SVM trong artifacts/")

    try:
        model = joblib.load(MODEL_PATH)
        features_array = np.array(data.features).reshape(1, -1)
        pred_cls = int(model.predict(features_array)[0])
        
        confidence = 95.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features_array)[0]
            confidence = round(float(np.max(probs)) * 100, 2)
        elif hasattr(model, "decision_function"):
            dec_val = abs(float(model.decision_function(features_array)[0]))
            confidence = round(min(99.9, 85.0 + dec_val * 5.0), 2)

        is_benign = (pred_cls == 1)
        result_label = "Lành tính" if is_benign else "Ác tính"
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO patient_records 
            (doctor_username, patient_id, patient_name, patient_age, diagnosis_date, result_label, is_benign, confidence, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (data.doctor_username, data.patient_id, data.patient_name, data.patient_age, now_str, result_label, 1 if is_benign else 0, confidence, data.notes))
        conn.commit()
        cursor.close()
        conn.close()

        return {
            "patient_name": data.patient_name,
            "result_label": result_label,
            "is_benign": is_benign,
            "confidence": confidence,
            "diagnosis_date": now_str
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích: {str(e)}")

@app.get("/api/records/{doctor_username}")
def get_doctor_records(doctor_username: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, patient_id, patient_name, patient_age, diagnosis_date, result_label, is_benign, confidence, notes
        FROM patient_records WHERE doctor_username = ? ORDER BY id DESC
    ''', (doctor_username.strip().lower(),))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    records = []
    for r in rows:
        records.append({
            "id": r[0], "patient_id": r[1], "patient_name": r[2], "patient_age": r[3],
            "diagnosis_date": r[4], "result_label": r[5], "is_benign": bool(r[6]),
            "confidence": r[7], "notes": r[8]
        })
    return records

@app.delete("/api/records/{record_id}")
def delete_record(record_id: int):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM patient_records WHERE id = ?", (record_id,))
    conn.commit()
    cursor.close()
    conn.close()
    return {"status": "success", "message": "Đã xóa hồ sơ bệnh nhân!"}

@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Doctor Portal - Cổng Chẩn Đoán AI Ung Thư Vú</title>
    <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-pink: #d81b60;
            --primary-hover: #c2185b;
            --light-pink: #fce4ec;
            --bg-light: #fff5f8;
            --card-border: #f8bbd0;
            --text-dark: #2c3e50;
            --text-muted: #6c757d;
            --success-green: #2e7d32;
            --danger-red: #c62828;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Nunito', sans-serif; }
        body { background-color: var(--bg-light); color: var(--text-dark); display: flex; flex-direction: column; min-height: 100vh; }
        header { background-color: #ffffff; border-bottom: 2px solid var(--card-border); padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 2px 10px rgba(216, 27, 96, 0.08); position: sticky; top: 0; z-index: 100; }
        .logo-area { display: flex; align-items: center; gap: 10px; font-weight: 800; font-size: 1.2rem; color: var(--primary-pink); }
        nav { display: flex; gap: 10px; align-items: center; }
        .nav-link { text-decoration: none; color: var(--text-dark); font-weight: 700; padding: 8px 16px; border-radius: 20px; cursor: pointer; transition: all 0.2s; }
        .nav-link:hover, .nav-link.active { background-color: var(--light-pink); color: var(--primary-pink); }
        .doctor-badge { background: var(--light-pink); color: var(--primary-pink); padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.9rem; margin-right: 10px; }
        .main-container { flex: 1; max-width: 1000px; width: 100%; margin: 25px auto; padding: 0 20px; }
        .page { display: none; }
        .page.active-page { display: block; animation: fadeIn 0.3s ease-in-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
        .card { background: #ffffff; border-radius: 16px; padding: 30px; box-shadow: 0 4px 20px rgba(216, 27, 96, 0.06); border: 1px solid var(--card-border); margin-bottom: 20px; }
        .btn { background-color: var(--primary-pink); color: white; padding: 10px 22px; font-size: 0.95rem; font-weight: 700; border: none; border-radius: 25px; cursor: pointer; transition: 0.2s; text-align: center; }
        .btn:hover { background-color: var(--primary-hover); }
        .btn-danger { background-color: var(--danger-red); }
        .btn-outline { background-color: transparent; color: var(--primary-pink); border: 2px solid var(--primary-pink); }
        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .form-group { display: flex; flex-direction: column; gap: 6px; }
        .form-group label { font-size: 0.88rem; font-weight: 700; }
        .form-group input, .form-group textarea { padding: 10px 12px; border: 1px solid var(--card-border); border-radius: 8px; font-size: 0.95rem; outline: none; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; }
        th, td { padding: 12px 14px; text-align: left; border-bottom: 1px solid var(--card-border); font-size: 0.92rem; }
        th { background-color: var(--light-pink); color: var(--primary-pink); font-weight: 800; }
        .badge-benign { background-color: #e8f5e9; color: var(--success-green); padding: 4px 10px; border-radius: 12px; font-weight: 700; }
        .badge-malignant { background-color: #ffebee; color: var(--danger-red); padding: 4px 10px; border-radius: 12px; font-weight: 700; }
        footer { background: #ffffff; border-top: 1px solid var(--card-border); text-align: center; padding: 18px; color: var(--text-muted); font-size: 0.88rem; margin-top: auto; }
    </style>
</head>
<body>
    <header>
        <div class="logo-area">
            <span>🩺</span>
            <span>Hệ Thống Chẩn Đoán AI - Doctor Portal</span>
        </div>
        <nav id="nav-menu"></nav>
    </header>

    <div class="main-container">
        <!-- 1. ĐĂNG NHẬP / ĐĂNG KÝ -->
        <div id="auth-page" class="page active-page">
            <div class="card" style="max-width: 450px; margin: 40px auto;">
                <h2 style="text-align: center; color: var(--primary-pink); margin-bottom: 20px;" id="auth-title">Đăng Nhập Bác Sĩ</h2>
                <form id="auth-form" onsubmit="handleAuth(event)">
                    <div class="form-group" style="margin-bottom: 15px;">
                        <label>Tài khoản Bác sĩ</label>
                        <input type="text" id="auth-username" required placeholder="dr_nguyen">
                    </div>
                    <div class="form-group" style="margin-bottom: 15px;">
                        <label>Mật khẩu</label>
                        <input type="password" id="auth-password" required>
                    </div>
                    <div id="register-fields" style="display: none;">
                        <div class="form-group" style="margin-bottom: 15px;">
                            <label>Họ và Tên Bác sĩ</label>
                            <input type="text" id="auth-fullname" placeholder="BS. Nguyễn Văn A">
                        </div>
                        <div class="form-group" style="margin-bottom: 15px;">
                            <label>Bệnh viện / Phòng khám</label>
                            <input type="text" id="auth-hospital" placeholder="Bệnh viện Đà Nẵng">
                        </div>
                    </div>
                    <button type="submit" class="btn" style="width: 100%; margin-top: 10px;" id="auth-submit-btn">Đăng Nhập ➔</button>
                </form>
                <div style="text-align: center; margin-top: 20px; font-size: 0.9rem;">
                    <span id="auth-toggle-text">Chưa có tài khoản Bác sĩ?</span>
                    <a href="#" onclick="toggleAuthMode()" style="color: var(--primary-pink); font-weight: 700; text-decoration: none;" id="auth-toggle-link"> Đăng ký ngay</a>
                </div>
            </div>
        </div>

        <!-- 2. PHÂN TÍCH CHẨN ĐOÁN -->
        <div id="diagnose-page" class="page">
            <div class="card">
                <h2 style="color: var(--primary-pink); margin-bottom: 20px;">📋 Nhập Hồ Sơ Bệnh Nhân & Chỉ Số Giải Phẫu</h2>
                <form onsubmit="submitDiagnosis(event)">
                    <h3 style="margin-bottom: 12px; font-size: 1.05rem;">1. Thông tin bệnh nhân</h3>
                    <div class="form-grid">
                        <div class="form-group"><label>Mã Bệnh Nhân</label><input type="text" id="p-id" required placeholder="BN-2026-001"></div>
                        <div class="form-group"><label>Họ và Tên Bệnh Nhân</label><input type="text" id="p-name" required placeholder="Trần Thị B"></div>
                        <div class="form-group"><label>Tuổi</label><input type="number" id="p-age" value="45" required></div>
                    </div>
                    <h3 style="margin-bottom: 12px; font-size: 1.05rem; margin-top: 15px;">2. Chỉ số sinh học (SVM Features)</h3>
                    <div class="form-grid">
                        <div class="form-group"><label>Mean Radius</label><input type="number" step="any" id="f0" value="14.12" required></div>
                        <div class="form-group"><label>Mean Texture</label><input type="number" step="any" id="f1" value="19.28" required></div>
                        <div class="form-group"><label>Mean Perimeter</label><input type="number" step="any" id="f2" value="91.96" required></div>
                        <div class="form-group"><label>Mean Area</label><input type="number" step="any" id="f3" value="654.88" required></div>
                        <div class="form-group"><label>Mean Smoothness</label><input type="number" step="any" id="f4" value="0.096" required></div>
                        <div class="form-group"><label>Worst Radius</label><input type="number" step="any" id="f20" value="16.26" required></div>
                        <div class="form-group"><label>Worst Area</label><input type="number" step="any" id="f23" value="880.58" required></div>
                        <div class="form-group"><label>Worst Smoothness</label><input type="number" step="any" id="f24" value="0.132" required></div>
                    </div>
                    <div class="form-group" style="margin-bottom: 20px;">
                        <label>Ghi chú lâm sàng của Bác sĩ</label>
                        <textarea id="p-notes" rows="2" placeholder="Bệnh nhân khám định kỳ..."></textarea>
                    </div>
                    <button type="submit" class="btn" style="width: 100%;">Thực Hiện Phân Tích Chẩn Đoán AI 🚀</button>
                </form>
            </div>
        </div>

        <!-- 3. LỊCH SỬ CHẨN ĐOÁN -->
        <div id="history-page" class="page">
            <div class="card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <h2 style="color: var(--primary-pink);">📂 Quản Lý Lịch Sử Bệnh Nhân</h2>
                    <button class="btn btn-outline" onclick="loadHistory()">🔄 Tải lại dữ liệu</button>
                </div>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Mã BN</th>
                                <th>Họ Tên BN</th>
                                <th>Tuổi</th>
                                <th>Ngày Khám</th>
                                <th>Kết Quả AI</th>
                                <th>Độ Tin Cậy</th>
                                <th>Ghi Chú</th>
                                <th>Thao Tác</th>
                            </tr>
                        </thead>
                        <tbody id="history-table-body"></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <footer>Doctor Portal - Hệ thống AI Hỗ trợ Chẩn đoán Ung thư Vú</footer>

    <script>
        let currentDoctor = null;
        let isRegisterMode = false;
        const default30 = [
            14.12, 19.28, 91.96, 654.88, 0.096, 0.104, 0.088, 0.048, 0.181, 0.062,
            0.405, 1.216, 2.866, 40.33, 0.007, 0.025, 0.031, 0.011, 0.020, 0.003,
            16.26, 25.67, 107.2, 880.58, 0.132, 0.254, 0.272, 0.114, 0.290, 0.083
        ];

        function toggleAuthMode(forceLogin = false) {
            if (forceLogin) {
                isRegisterMode = false;
            } else {
                isRegisterMode = !isRegisterMode;
            }
            document.getElementById('auth-title').innerText = isRegisterMode ? "Đăng Ký Tài Khoản Bác Sĩ" : "Đăng Nhập Bác Sĩ";
            document.getElementById('register-fields').style.display = isRegisterMode ? "block" : "none";
            document.getElementById('auth-submit-btn').innerText = isRegisterMode ? "Đăng Ký Tài Khoản" : "Đăng Nhập ➔";
            document.getElementById('auth-toggle-text').innerText = isRegisterMode ? "Đã có tài khoản Bác sĩ?" : "Chưa có tài khoản Bác sĩ?";
            document.getElementById('auth-toggle-link').innerText = isRegisterMode ? " Đăng nhập" : " Đăng ký ngay";
        }

        async function handleAuth(event) {
            event.preventDefault();
            const u = document.getElementById('auth-username').value;
            const p = document.getElementById('auth-password').value;

            if (isRegisterMode) {
                const fn = document.getElementById('auth-fullname').value || "Bác sĩ";
                const hp = document.getElementById('auth-hospital').value || "Bệnh viện";
                try {
                    const res = await fetch('/api/register', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ username: u, password: p, full_name: fn, hospital: hp })
                    });
                    
                    let data = {};
                    try {
                        data = await res.json();
                    } catch(e) {
                        data = { detail: "Lỗi phản hồi không đúng định dạng từ máy chủ." };
                    }

                    if (res.ok) {
                        alert(`✅ ĐĂNG KÝ THÀNH CÔNG!\n\nTài khoản: ${data.username}\n\nHệ thống chuyển sang màn hình Đăng Nhập.`);
                        toggleAuthMode(true);
                        document.getElementById('auth-username').value = data.username;
                        document.getElementById('auth-password').value = p;
                    } else {
                        alert("❌ Lỗi: " + (data.detail || "Tài khoản đã tồn tại!"));
                    }
                } catch(err) {
                    alert("❌ Lỗi mạng hoặc kết nối máy chủ không phản hồi!");
                }
            } else {
                try {
                    const res = await fetch('/api/login', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ username: u, password: p })
                    });
                    
                    let data = {};
                    try {
                        data = await res.json();
                    } catch(e) {
                        data = { detail: "Mã phản hồi từ máy chủ không hợp lệ." };
                    }

                    if (res.ok) {
                        currentDoctor = data;
                        renderNavbar();
                        showPage('diagnose-page');
                    } else {
                        alert("❌ Lỗi đăng nhập: " + (data.detail || "Sai thông tin!"));
                    }
                } catch(err) {
                    alert("❌ Lỗi kết nối!");
                }
            }
        }

        function renderNavbar() {
            const nav = document.getElementById('nav-menu');
            if (currentDoctor) {
                nav.innerHTML = `
                    <span class="doctor-badge">👨‍⚕️ ${currentDoctor.full_name} (${currentDoctor.hospital})</span>
                    <a class="nav-link active" onclick="showPage('diagnose-page')">Chẩn đoán mới</a>
                    <a class="nav-link" onclick="showPage('history-page'); loadHistory();">Lịch sử bệnh nhân</a>
                    <a class="nav-link" onclick="logout()" style="color: var(--danger-red);">Đăng xuất</a>
                `;
            } else {
                nav.innerHTML = `<a class="nav-link active" onclick="showPage('auth-page')">Đăng nhập Bác sĩ</a>`;
            }
        }

        function logout() { currentDoctor = null; renderNavbar(); showPage('auth-page'); }
        function showPage(pageId) {
            document.querySelectorAll('.page').forEach(p => p.classList.remove('active-page'));
            document.getElementById(pageId).classList.add('active-page');
        }

        async function submitDiagnosis(e) {
            e.preventDefault();
            if (!currentDoctor) return alert("Vui lòng đăng nhập!");
            let f = [...default30];
            f[0] = parseFloat(document.getElementById('f0').value) || f[0];
            f[1] = parseFloat(document.getElementById('f1').value) || f[1];
            f[2] = parseFloat(document.getElementById('f2').value) || f[2];
            f[3] = parseFloat(document.getElementById('f3').value) || f[3];
            f[4] = parseFloat(document.getElementById('f4').value) || f[4];
            f[20] = parseFloat(document.getElementById('f20').value) || f[20];
            f[23] = parseFloat(document.getElementById('f23').value) || f[23];
            f[24] = parseFloat(document.getElementById('f24').value) || f[24];

            const reqData = {
                doctor_username: currentDoctor.username,
                patient_id: document.getElementById('p-id').value,
                patient_name: document.getElementById('p-name').value,
                patient_age: parseInt(document.getElementById('p-age').value),
                notes: document.getElementById('p-notes').value,
                features: f
            };

            const res = await fetch('/api/diagnose', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(reqData)
            });

            if (res.ok) {
                const data = await res.json();
                alert(`KẾT QUẢ CHẨN ĐOÁN AI:\nBệnh nhân: ${data.patient_name}\nKết luận: ${data.result_label.toUpperCase()}\nĐộ tin cậy: ${data.confidence}%\n\nĐã lưu hồ sơ thành công!`);
                showPage('history-page'); loadHistory();
            } else { alert("Lỗi chẩn đoán!"); }
        }

        async function loadHistory() {
            if (!currentDoctor) return;
            const res = await fetch(`/api/records/${currentDoctor.username}`);
            const records = await res.json();
            const tbody = document.getElementById('history-table-body');
            tbody.innerHTML = '';
            if (records.length === 0) {
                tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted);">Chưa có hồ sơ bệnh nhân nào.</td></tr>`;
                return;
            }
            records.forEach(r => {
                const badgeClass = r.is_benign ? "badge-benign" : "badge-malignant";
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td><strong>${r.patient_id}</strong></td>
                    <td>${r.patient_name}</td>
                    <td>${r.patient_age}</td>
                    <td>${r.diagnosis_date}</td>
                    <td><span class="${badgeClass}">${r.result_label}</span></td>
                    <td>${r.confidence}%</td>
                    <td>${r.notes || '-'}</td>
                    <td><button class="btn btn-danger" style="padding: 4px 10px; font-size: 0.8rem;" onclick="deleteRecord(${r.id})">🗑 Xóa</button></td>
                `;
                tbody.appendChild(row);
            });
        }

        async function deleteRecord(id) {
            if (confirm("Xóa hồ sơ bệnh nhân này?")) {
                const res = await fetch(`/api/records/${id}`, { method: 'DELETE' });
                if (res.ok) { loadHistory(); }
            }
        }
        renderNavbar();
    </script>
</body>
</html>
    """