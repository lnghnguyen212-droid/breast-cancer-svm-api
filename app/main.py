import os
import sqlite3
import io
import csv
import joblib
import bcrypt
import numpy as np
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Doctor Portal - Breast Cancer SVM Diagnostic System",
    description="Hệ thống hỗ trợ chẩn đoán dành cho Bác sĩ",
    version="3.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đường dẫn Database SQLite chuẩn
if os.name == 'nt':
    DB_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "doctor_portal.db")
else:
    DB_PATH = "/tmp/doctor_portal.db"

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "breast_cancer_svm.joblib")

def hash_password(password: str) -> str:
    pwd_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        pwd_bytes = plain_password.encode('utf-8')[:72]
        hash_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(pwd_bytes, hash_bytes)
    except Exception:
        return False

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
                notes TEXT
            );
        ''')
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"[DB Init Error]: {e}")

init_db()

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
    kernel: Optional[str] = "rbf"
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
    hashed_pwd = hash_password(doc.password)

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

    if not row or not verify_password(doc.password, row[0]):
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
        loaded = joblib.load(MODEL_PATH)
        features_array = np.array(data.features).reshape(1, -1)

        selected_kernel = (data.kernel or "rbf").lower()
        if isinstance(loaded, dict) and selected_kernel in loaded:
            model = loaded[selected_kernel]
        elif hasattr(loaded, "set_params"):
            try:
                model = loaded
                if hasattr(model, "kernel"):
                    model.set_params(kernel=selected_kernel)
                elif hasattr(model, "named_steps"):
                    for step in model.named_steps.values():
                        if hasattr(step, "kernel"):
                            step.set_params(kernel=selected_kernel)
            except Exception:
                model = loaded
        else:
            model = loaded

        pred_cls = int(model.predict(features_array)[0])

        is_benign = (pred_cls == 1)
        result_label = "Lành tính" if is_benign else "Ác tính"
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO patient_records 
            (doctor_username, patient_id, patient_name, patient_age, diagnosis_date, result_label, is_benign, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (data.doctor_username, data.patient_id, data.patient_name, data.patient_age, now_str, result_label, 1 if is_benign else 0, data.notes))
        conn.commit()
        cursor.close()
        conn.close()

        return {
            "patient_name": data.patient_name,
            "result_label": result_label,
            "is_benign": is_benign,
            "kernel_used": selected_kernel.upper(),
            "diagnosis_date": now_str
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích: {str(e)}")

@app.get("/api/records/{doctor_username}")
def get_doctor_records(doctor_username: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, patient_id, patient_name, patient_age, diagnosis_date, result_label, is_benign, notes
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
            "notes": r[7]
        })
    return records

@app.get("/api/records/{doctor_username}/export-csv")
def export_patient_records_csv(doctor_username: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT patient_id, patient_name, patient_age, diagnosis_date, result_label, notes
        FROM patient_records WHERE doctor_username = ? ORDER BY id DESC
    ''', (doctor_username.strip().lower(),))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Mã BN", "Họ Tên BN", "Tuổi", "Ngày Khám", "Kết Quả Chẩn Đoán", "Ghi Chú"])
    for row in rows:
        writer.writerow(row)

    output.seek(0)
    filename = f"danh_sach_benh_nhan_{doctor_username}.csv"
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode('utf-8-sig')),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

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
    <title>DOCTOR PORTAL — BREAST CANCER AI DIAGNOSTICS</title>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --app-black: #1a1a1a;
            --app-gray: #666666;
            --app-light-bg: #f8f9fa;
            --app-border: #e2e8f0;
            --app-card-bg: #ffffff;
            --success-green: #2e7d32;
            --danger-red: #c62828;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            background-color: var(--app-light-bg); 
            color: var(--app-black); 
            font-family: 'Inter', sans-serif; 
            display: flex; 
            flex-direction: column; 
            min-height: 100vh;
        }

        /* TOP BANNER */
        .top-notice {
            background-color: var(--app-black);
            color: #ffffff;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.2em;
            text-align: center;
            padding: 8px 15px;
            font-weight: 500;
        }

        /* HEADER */
        header { 
            background-color: #ffffff; 
            border-bottom: 1px solid var(--app-border); 
            padding: 18px 40px; 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
            position: sticky; 
            top: 0; 
            z-index: 100; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        }
        .logo-area { 
            font-family: 'Playfair Display', serif; 
            font-size: 1.3rem; 
            font-weight: 700; 
            letter-spacing: 0.08em; 
            text-transform: uppercase; 
            color: var(--app-black); 
            display: flex;
            align-items: center;
            gap: 10px;
        }
        nav { display: flex; gap: 20px; align-items: center; }
        .nav-link { 
            text-decoration: none; 
            color: var(--app-black); 
            font-size: 0.8rem; 
            text-transform: uppercase; 
            letter-spacing: 0.1em; 
            font-weight: 600; 
            padding: 8px 16px; 
            border-radius: 20px;
            cursor: pointer; 
            transition: all 0.2s ease; 
        }
        .nav-link:hover, .nav-link.active { 
            background-color: #f1f5f9;
        }
        .doctor-badge { 
            background: #f1f5f9; 
            color: var(--app-black); 
            padding: 8px 18px; 
            border-radius: 20px; 
            font-size: 0.8rem; 
            font-weight: 600;
            border: 1px solid var(--app-border); 
            margin-right: 10px; 
        }

        /* HERO BANNER WITH ROUNDED CORNERS */
        .hero-section {
            position: relative;
            max-width: 1100px;
            width: calc(100% - 50px);
            margin: 25px auto 0 auto;
            height: 220px;
            overflow: hidden;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        .hero-section img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            filter: brightness(0.65);
        }
        .hero-overlay {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            color: #ffffff;
            text-align: center;
            padding: 20px;
        }
        .hero-title {
            font-family: 'Playfair Display', serif;
            font-size: 2rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .hero-subtitle {
            font-size: 0.82rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            opacity: 0.9;
            font-weight: 300;
        }

        /* MAIN LAYOUT */
        .main-container { flex: 1; max-width: 1100px; width: 100%; margin: 30px auto; padding: 0 25px; }
        .page { display: none; }
        .page.active-page { display: block; animation: fadeIn 0.3s ease-in-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

        /* ROUNDED CARDS */
        .card { 
            background: #ffffff; 
            border: 1px solid var(--app-border); 
            border-radius: 16px; 
            padding: 35px; 
            margin-bottom: 25px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.02);
        }
        .card-header {
            font-family: 'Playfair Display', serif;
            font-size: 1.4rem;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--app-border);
            color: var(--app-black);
        }

        /* ROUNDED BUTTONS */
        .btn { 
            background-color: var(--app-black); 
            color: #ffffff; 
            padding: 12px 28px; 
            font-size: 0.8rem; 
            font-weight: 600; 
            text-transform: uppercase; 
            letter-spacing: 0.12em; 
            border: 1px solid var(--app-black); 
            border-radius: 25px; 
            cursor: pointer; 
            transition: all 0.2s ease; 
            text-align: center; 
            display: inline-block;
        }
        .btn:hover { 
            background-color: #ffffff; 
            color: var(--app-black); 
        }
        .btn-danger { 
            background-color: var(--danger-red); 
            border-color: var(--danger-red); 
            color: #fff; 
        }
        .btn-danger:hover {
            background-color: #fff;
            color: var(--danger-red);
        }
        .btn-outline { 
            background-color: transparent; 
            color: var(--app-black); 
            border: 1px solid var(--app-border); 
        }
        .btn-outline:hover {
            border-color: var(--app-black);
            background-color: #f8fafc;
        }

        /* ROUNDED FORM INPUTS */
        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 18px; margin-bottom: 22px; }
        .form-group { display: flex; flex-direction: column; gap: 6px; }
        .form-group label { 
            font-size: 0.75rem; 
            text-transform: uppercase; 
            letter-spacing: 0.08em; 
            color: var(--app-gray); 
            font-weight: 600; 
        }
        .form-group input, .form-group select, .form-group textarea { 
            padding: 12px 16px; 
            border: 1px solid var(--app-border); 
            border-radius: 10px; 
            font-size: 0.9rem; 
            outline: none; 
            background-color: #f8fafc; 
            font-family: 'Inter', sans-serif;
            transition: all 0.2s ease;
        }
        .form-group input:focus, .form-group select:focus, .form-group textarea:focus {
            border-color: var(--app-black);
            background-color: #ffffff;
            box-shadow: 0 0 0 3px rgba(0,0,0,0.05);
        }

        /* TABLES */
        table { width: 100%; border-collapse: separate; border-spacing: 0; margin-top: 15px; }
        th, td { padding: 14px 16px; text-align: left; border-bottom: 1px solid var(--app-border); font-size: 0.88rem; }
        th { 
            background-color: #f1f5f9; 
            color: var(--app-black); 
            font-weight: 600; 
            text-transform: uppercase; 
            letter-spacing: 0.08em; 
            font-size: 0.75rem; 
        }
        th:first-child { border-top-left-radius: 10px; }
        th:last-child { border-top-right-radius: 10px; }
        .badge-benign { 
            background-color: #e8f5e9; 
            color: var(--success-green); 
            padding: 6px 14px; 
            border-radius: 20px; 
            font-size: 0.78rem; 
            font-weight: 600; 
        }
        .badge-malignant { 
            background-color: #ffebee; 
            color: var(--danger-red); 
            padding: 6px 14px; 
            border-radius: 20px; 
            font-size: 0.78rem; 
            font-weight: 600; 
        }

        /* FOOTER */
        footer { 
            background: #ffffff; 
            color: var(--app-gray); 
            text-align: center; 
            padding: 25px 20px; 
            font-size: 0.75rem; 
            text-transform: uppercase; 
            letter-spacing: 0.15em; 
            margin-top: auto; 
            border-top: 1px solid var(--app-border);
        }
    </style>
</head>
<body>

    <div class="top-notice">
        Clinical AI Precision • Breast Cancer Diagnostic System
    </div>

    <header>
        <div class="logo-area">
            <span>🩺 DOCTOR PORTAL AI</span>
        </div>
        <nav id="nav-menu"></nav>
    </header>

    <div class="hero-section">
        <img src="https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?q=80&w=1600&auto=format&fit=crop" alt="Medical Banner">
        <div class="hero-overlay">
            <h1 class="hero-title">Precision Diagnostics</h1>
            <p class="hero-subtitle">Support Vector Machine • Clinical Oncology AI</p>
        </div>
    </div>

    <div class="main-container">
        <!-- 1. ĐĂNG NHẬP / ĐĂNG KÝ -->
        <div id="auth-page" class="page active-page">
            <div class="card" style="max-width: 450px; margin: 20px auto;">
                <h2 class="card-header" id="auth-title" style="text-align: center;">Đăng Nhập Bác Sĩ</h2>
                <form id="auth-form" onsubmit="handleAuth(event)">
                    <div class="form-group" style="margin-bottom: 18px;">
                        <label>Tài khoản Bác sĩ</label>
                        <input type="text" id="auth-username" required placeholder="dr_nguyen">
                    </div>
                    <div class="form-group" style="margin-bottom: 18px;">
                        <label>Mật khẩu</label>
                        <input type="password" id="auth-password" required>
                    </div>
                    <div id="register-fields" style="display: none;">
                        <div class="form-group" style="margin-bottom: 18px;">
                            <label>Họ và Tên Bác sĩ</label>
                            <input type="text" id="auth-fullname" placeholder="BS. Nguyễn Văn A">
                        </div>
                        <div class="form-group" style="margin-bottom: 18px;">
                            <label>Bệnh viện / Phòng khám</label>
                            <input type="text" id="auth-hospital" placeholder="Bệnh viện Đà Nẵng">
                        </div>
                    </div>
                    <button type="submit" class="btn" style="width: 100%; margin-top: 10px;" id="auth-submit-btn">Đăng Nhập ➔</button>
                </form>
                <div style="text-align: center; margin-top: 20px; font-size: 0.82rem;">
                    <span id="auth-toggle-text">Chưa có tài khoản Bác sĩ?</span>
                    <a href="#" onclick="toggleAuthMode()" style="color: var(--app-black); font-weight: 700; text-decoration: underline;" id="auth-toggle-link"> Đăng ký ngay</a>
                </div>
            </div>
        </div>

        <!-- 2. PHÂN TÍCH CHẨN ĐOÁN -->
        <div id="diagnose-page" class="page">
            <div class="card">
                <h2 class="card-header">📋 Nhập Hồ Sơ Bệnh Nhân & Chỉ Số Giải Phẫu</h2>
                <form onsubmit="submitDiagnosis(event)">
                    <h3 style="margin-bottom: 12px; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--app-black);">1. Thông tin bệnh nhân</h3>
                    <div class="form-grid">
                        <div class="form-group"><label>Mã Bệnh Nhân</label><input type="text" id="p-id" required placeholder="BN-2026-001"></div>
                        <div class="form-group"><label>Họ và Tên Bệnh Nhân</label><input type="text" id="p-name" required placeholder="Trần Thị B"></div>
                        <div class="form-group"><label>Tuổi</label><input type="number" id="p-age" value="45" required></div>
                    </div>

                    <h3 style="margin-bottom: 12px; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--app-black); margin-top: 20px;">2. Chỉ số sinh học & Thuật toán (SVM Features)</h3>
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Chọn Kernel SVM</label>
                            <select id="p-kernel">
                                <option value="rbf" selected>RBF (Radial Basis Function)</option>
                                <option value="linear">Linear (Tuyến tính)</option>
                                <option value="poly">Polynomial (Đa thức)</option>
                                <option value="sigmoid">Sigmoid</option>
                            </select>
                        </div>
                        <div class="form-group"><label>Bán kính trung bình (Mean Radius)</label><input type="number" step="any" id="f0" value="14.12" required></div>
                        <div class="form-group"><label>Độ thô trung bình (Mean Texture)</label><input type="number" step="any" id="f1" value="19.28" required></div>
                        <div class="form-group"><label>Chu vi trung bình (Mean Perimeter)</label><input type="number" step="any" id="f2" value="91.96" required></div>
                        <div class="form-group"><label>Diện tích trung bình (Mean Area)</label><input type="number" step="any" id="f3" value="654.88" required></div>
                        <div class="form-group"><label>Độ nhẵn trung bình (Mean Smoothness)</label><input type="number" step="any" id="f4" value="0.096" required></div>
                        <div class="form-group"><label>Bán kính lớn nhất (Worst Radius)</label><input type="number" step="any" id="f20" value="16.26" required></div>
                        <div class="form-group"><label>Diện tích lớn nhất (Worst Area)</label><input type="number" step="any" id="f23" value="880.58" required></div>
                        <div class="form-group"><label>Độ nhẵn lớn nhất (Worst Smoothness)</label><input type="number" step="any" id="f24" value="0.132" required></div>
                    </div>

                    <div class="form-group" style="margin-bottom: 22px;">
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
                    <h2 class="card-header" style="margin-bottom: 0; border-bottom: none; padding-bottom: 0;">Quản Lý Lịch Sử Bệnh Nhân</h2>
                    <div>
                        <button class="btn btn-outline" onclick="exportCSV()" style="margin-right: 8px;">📥 Tải dữ liệu (CSV)</button>
                        <button class="btn btn-outline" onclick="loadHistory()">🔄 Tải lại dữ liệu</button>
                    </div>
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

    <footer>
        Doctor Portal • Breast Cancer Support Vector Machine AI
    </footer>

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
                    let data = await res.json();
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
                    let data = await res.json();
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

            const selectedKernel = document.getElementById('p-kernel').value;

            const reqData = {
                doctor_username: currentDoctor.username,
                patient_id: document.getElementById('p-id').value,
                patient_name: document.getElementById('p-name').value,
                patient_age: parseInt(document.getElementById('p-age').value),
                notes: document.getElementById('p-notes').value,
                kernel: selectedKernel,
                features: f
            };

            try {
                const res = await fetch('/api/diagnose', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(reqData)
                });
                let data = await res.json();
                if (res.ok) {
                    alert(`✅ KẾT QUẢ CHẨN ĐOÁN AI (${data.kernel_used}):\n\nBệnh nhân: ${data.patient_name}\nKết quả: ${data.result_label}`);
                    showPage('history-page');
                    loadHistory();
                } else {
                    alert("❌ Lỗi phân tích: " + (data.detail || "Không rõ lỗi"));
                }
            } catch(err) {
                alert("❌ Lỗi kết nối máy chủ!");
            }
        }

        async function loadHistory() {
            if (!currentDoctor) return;
            try {
                const res = await fetch(`/api/records/${currentDoctor.username}`);
                const records = await res.json();
                const tbody = document.getElementById('history-table-body');
                tbody.innerHTML = '';
                if (records.length === 0) {
                    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;">Chưa có dữ liệu chẩn đoán.</td></tr>`;
                    return;
                }
                records.forEach(r => {
                    const badgeClass = r.is_benign ? 'badge-benign' : 'badge-malignant';
                    tbody.innerHTML += `
                        <tr>
                            <td>${r.patient_id}</td>
                            <td><strong>${r.patient_name}</strong></td>
                            <td>${r.patient_age}</td>
                            <td>${r.diagnosis_date}</td>
                            <td><span class="${badgeClass}">${r.result_label}</span></td>
                            <td>${r.notes || '-'}</td>
                            <td><button class="btn btn-danger" style="padding: 5px 12px; font-size:0.75rem;" onclick="deleteRecord(${r.id})">Xóa</button></td>
                        </tr>
                    `;
                });
            } catch(err) {
                console.error("Lỗi tải lịch sử:", err);
            }
        }

        function exportCSV() {
            if (!currentDoctor) return alert("Vui lòng đăng nhập!");
            window.location.href = `/api/records/${currentDoctor.username}/export-csv`;
        }

        async function deleteRecord(id) {
            if (!confirm("Bạn có chắc chắn muốn xóa hồ sơ này?")) return;
            try {
                const res = await fetch(`/api/records/${id}`, { method: 'DELETE' });
                if (res.ok) {
                    loadHistory();
                }
            } catch(err) {
                alert("Lỗi khi xóa hồ sơ!");
            }
        }

        renderNavbar();
    </script>
</body>
</html>
    """