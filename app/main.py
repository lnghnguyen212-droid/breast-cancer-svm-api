import os
import pyodbc
import joblib
import numpy as np
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(
    title="Doctor Portal - Breast Cancer SVM Diagnostic System",
    description="Hệ thống hỗ trợ chẩn đoán dành cho Bác sĩ kết nối SQL Server 2022",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chuỗi kết nối SQL Server 2022 (Thay đổi thông số Server/User/Password phù hợp với máy/server của bạn)
DB_SERVER = os.getenv("DB_SERVER", "localhost,1433")
DB_NAME = os.getenv("DB_NAME", "DoctorPortalDB")
DB_USER = os.getenv("DB_USER", "sa")
DB_PASSWORD = os.getenv("DB_PASSWORD", "YourPassword123!")

CONN_STR = (
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={DB_SERVER};"
    f"DATABASE={DB_NAME};"
    f"UID={DB_USER};"
    f"PWD={DB_PASSWORD};"
    f"TrustServerCertificate=yes;"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "breast_cancer_svm.joblib")

def get_db_connection():
    try:
        conn = pyodbc.connect(CONN_STR, timeout=10)
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi kết nối CSDL SQL Server 2022: {str(e)}")

# Tự động tạo Bảng trong SQL Server 2022 nếu chưa tồn tại
def init_sql_server_tables():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Bảng Bác sĩ
        cursor.execute('''
            IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'doctors')
            CREATE TABLE doctors (
                id INT IDENTITY(1,1) PRIMARY KEY,
                username NVARCHAR(100) UNIQUE NOT NULL,
                password_hash NVARCHAR(255) NOT NULL,
                full_name NVARCHAR(200) NOT NULL,
                hospital NVARCHAR(200)
            )
        ''')
        
        # Bảng Lịch sử Bệnh nhân
        cursor.execute('''
            IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'patient_records')
            CREATE TABLE patient_records (
                id INT IDENTITY(1,1) PRIMARY KEY,
                doctor_username NVARCHAR(100) NOT NULL,
                patient_id NVARCHAR(50) NOT NULL,
                patient_name NVARCHAR(200) NOT NULL,
                patient_age INT NOT NULL,
                diagnosis_date DATETIME NOT NULL,
                result_label NVARCHAR(50) NOT NULL,
                is_benign BIT NOT NULL,
                confidence FLOAT NOT NULL,
                notes NVARCHAR(MAX)
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[SQL Server Init Warning]: {e}")

init_sql_server_tables()

# --- Schemas ---
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

# --- API Endpoints ---

@app.post("/api/register")
def register_doctor(doc: DoctorRegister):
    conn = get_db_connection()
    cursor = conn.cursor()
    hashed_pwd = pwd_context.hash(doc.password)
    try:
        cursor.execute(
            "INSERT INTO doctors (username, password_hash, full_name, hospital) VALUES (?, ?, ?, ?)",
            (doc.username, hashed_pwd, doc.full_name, doc.hospital)
        )
        conn.commit()
        return {"status": "success", "message": "Đăng ký Bác sĩ thành công trên SQL Server 2022!"}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Tài khoản Bác sĩ đã tồn tại hoặc lỗi SQL Server.")
    finally:
        conn.close()

@app.post("/api/login")
def login_doctor(doc: DoctorLogin):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash, full_name, hospital FROM doctors WHERE username = ?", (doc.username,))
    row = cursor.fetchone()
    conn.close()
    
    if not row or not pwd_context.verify(doc.password, row[0]):
        raise HTTPException(status_code=401, detail="Tên đăng nhập hoặc mật khẩu không chính xác!")
        
    return {
        "status": "success",
        "username": doc.username,
        "full_name": row[1],
        "hospital": row[2]
    }

@app.post("/api/diagnose")
def diagnose_and_save(data: DiagnosticRequest):
    if len(data.features) != 30:
        raise HTTPException(status_code=400, detail="Yêu cầu nhập đủ 30 thông số giải phẫu.")
    
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail="Mô hình SVM chưa được nạp.")

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
        now_dt = datetime.now()

        # Lưu thông tin bệnh nhân vào SQL Server 2022
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO patient_records 
            (doctor_username, patient_id, patient_name, patient_age, diagnosis_date, result_label, is_benign, confidence, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (data.doctor_username, data.patient_id, data.patient_name, data.patient_age, now_dt, result_label, 1 if is_benign else 0, confidence, data.notes))
        conn.commit()
        conn.close()

        return {
            "patient_name": data.patient_name,
            "result_label": result_label,
            "is_benign": is_benign,
            "confidence": confidence,
            "diagnosis_date": now_dt.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi phân tích chẩn đoán: {str(e)}")

@app.get("/api/records/{doctor_username}")
def get_doctor_records(doctor_username: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, patient_id, patient_name, patient_age, diagnosis_date, result_label, is_benign, confidence, notes
        FROM patient_records
        WHERE doctor_username = ?
        ORDER BY id DESC
    ''', (doctor_username,))
    rows = cursor.fetchall()
    conn.close()

    records = []
    for r in rows:
        records.append({
            "id": r[0],
            "patient_id": r[1],
            "patient_name": r[2],
            "patient_age": r[3],
            "diagnosis_date": str(r[4]),
            "result_label": r[5],
            "is_benign": bool(r[6]),
            "confidence": r[7],
            "notes": r[8]
        })
    return records

@app.delete("/api/records/{record_id}")
def delete_record(record_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM patient_records WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
    return {"status": "success", "message": "Đã xóa hồ sơ bệnh nhân khỏi SQL Server 2022!"}

