import streamlit as st
import requests
import pandas as pd

# Cấu hình trang
st.set_page_config(
    page_title="Chẩn Đoán Ung Thư Vú - AI Health",
    page_icon="🩺",
    layout="wide"
)

# Đường dẫn Backend API Render
API_URL = "https://breast-cancer-svm-api-6yrq.onrender.com/predict"

# CSS tùy chỉnh giao diện giống mẫu Y tế
st.markdown("""
    <style>
    .header-bar {
        background-color: #0E3B6A;
        padding: 15px 25px;
        color: white;
        border-radius: 6px;
        margin-bottom: 20px;
    }
    .header-title {
        font-size: 26px;
        font-weight: bold;
        display: inline;
    }
    .step-box {
        text-align: center;
        padding: 12px;
        font-weight: bold;
        font-size: 15px;
        border-radius: 4px;
        color: white;
    }
    .step-active {
        background-color: #0072CE;
    }
    .step-inactive {
        background-color: #FFF2B2;
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

# Thanh Header thương hiệu Y tế
st.markdown("""
    <div class="header-bar">
        <span class="header-title">🩺 HỆ THỐNG Y TẾ SỨC KHỎE VÚ | AI DIAGNOSTIC</span>
    </div>
""", unsafe_allow_html=True)

# Quản lý bước hiện tại (Step Control)
if "step" not in st.session_state:
    st.session_state.step = 1

# Thanh Tiến Trình 3 Bước (Giống mẫu SBB Healthcare)
step1_class = "step-active" if st.session_state.step == 1 else "step-inactive"
step2_class = "step-active" if st.session_state.step == 2 else "step-inactive"
step3_class = "step-active" if st.session_state.step == 3 else "step-inactive"

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.markdown(f'<div class="step-box {step1_class}">1. CHỌN TRIỆU CHỨNG & CHỈ SỐ</div>', unsafe_allow_html=True)
with col_s2:
    st.markdown(f'<div class="step-box {step2_class}">2. MÔ TẢ CHI TIẾT SỐ LIỆU</div>', unsafe_allow_html=True)
with col_s3:
    st.markdown(f'<div class="step-box {step3_class}">3. KẾT QUẢ & NGUY CƠ BỆNH LÝ</div>', unsafe_allow_html=True)

st.write("")
st.divider()

# ==================== BƯỚC 1: CHỌN TRIỆU CHỨNG & CHỈ SỐ SƠ BỘ ====================
if st.session_state.step == 1:
    st.subheader("📋 Bước 1: Chọn các triệu chứng biểu hiện lâm sàng & thông số chính")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(" **Biểu hiện lâm sàng nhận biết:**")
        symptom_1 = st.checkbox("Có khối u hoặc sờ thấy mảng cứng ở vú/nách")
        symptom_2 = st.checkbox("Thay đổi kích thước hoặc hình dạng vú")
        symptom_3 = st.checkbox("Dịch tiết bất thường ở núm vú")
        symptom_4 = st.checkbox("Da vú bị nhăn, co rút hoặc sần vỏ cam")
        
    with col_b:
        st.write(" **Chỉ số kích thước mô phỏng sơ bộ (Mean Values):**")
        mean_radius = st.number_input("Bán kính trung bình khối U (Mean Radius):", value=14.12, step=0.1)
        mean_texture = st.number_input("Độ mịn bề mặt (Mean Texture):", value=19.28, step=0.1)
        mean_perimeter = st.number_input("Chu vi khối U (Mean Perimeter):", value=91.96, step=0.1)
        mean_area = st.number_input("Diện tích khối U (Mean Area):", value=654.88, step=1.0)

    # Lưu biến vào session state
    st.session_state.mean_radius = mean_radius
    st.session_state.mean_texture = mean_texture
    st.session_state.mean_perimeter = mean_perimeter
    st.session_state.mean_area = mean_area

    st.write("")
    if st.button("Tiếp tục: Mô tả chi tiết ➔", type="primary"):
        st.session_state.step = 2
        st.rerun()

# ==================== BƯỚC 2: MÔ TẢ CHI TIẾT SỐ LIỆU ====================
elif st.session_state.step == 2:
    st.subheader("📊 Bước 2: Nhập đầy đủ 30 thông số giải phẫu bệnh học")
    
    st.info("Nhập chính xác các thông số xét nghiệm từ kết quả siêu âm/chụp X-quang tuyến vú.")
    
    t1, t2, t3 = st.tabs(["Thông số Trung bình (Mean)", "Sai số Chuẩn (SE Error)", "Chỉ số Giá trị Lớn nhất (Worst)"])
    
    with t1:
        c1, c2 = st.columns(2)
        with c1:
            m_rad = st.number_input("Mean Radius", value=st.session_state.get("mean_radius", 14.12))
            m_tex = st.number_input("Mean Texture", value=st.session_state.get("mean_texture", 19.28))
            m_per = st.number_input("Mean Perimeter", value=st.session_state.get("mean_perimeter", 91.96))
            m_are = st.number_input("Mean Area", value=st.session_state.get("mean_area", 654.88))
            m_smo = st.number_input("Mean Smoothness", value=0.096, format="%.4f")
        with c2:
            m_com = st.number_input("Mean Compactness", value=0.104, format="%.4f")
            m_con = st.number_input("Mean Concavity", value=0.088, format="%.4f")
            m_pts = st.number_input("Mean Concave Points", value=0.048, format="%.4f")
            m_sym = st.number_input("Mean Symmetry", value=0.181, format="%.4f")
            m_fra = st.number_input("Mean Fractal Dimension", value=0.062, format="%.4f")

    with t2:
        c1, c2 = st.columns(2)
        with c1:
            r_err = st.number_input("Radius Error", value=0.4051, format="%.4f")
            t_err = st.number_input("Texture Error", value=1.2168, format="%.4f")
            p_err = st.number_input("Perimeter Error", value=2.8660, format="%.4f")
            a_err = st.number_input("Area Error", value=40.337, format="%.4f")
            s_err = st.number_input("Smoothness Error", value=0.0070, format="%.4f")
        with c2:
            com_err = st.number_input("Compactness Error", value=0.0254, format="%.4f")
            con_err = st.number_input("Concavity Error", value=0.0319, format="%.4f")
            pts_err = st.number_input("Concave Points Error", value=0.0118, format="%.4f")
            sym_err = st.number_input("Symmetry Error", value=0.0205, format="%.4f")
            fra_err = st.number_input("Fractal Dimension Error", value=0.0038, format="%.4f")

    with t3:
        c1, c2 = st.columns(2)
        with c1:
            w_rad = st.number_input("Worst Radius", value=16.26)
            w_tex = st.number_input("Worst Texture", value=25.67)
            w_per = st.number_input("Worst Perimeter", value=107.26)
            w_are = st.number_input("Worst Area", value=880.58)
            w_smo = st.number_input("Worst Smoothness", value=0.132, format="%.4f")
        with c2:
            w_com = st.number_input("Worst Compactness", value=0.254, format="%.4f")
            w_con = st.number_input("Worst Concavity", value=0.272, format="%.4f")
            w_pts = st.number_input("Worst Concave Points", value=0.114, format="%.4f")
            w_sym = st.number_input("Worst Symmetry", value=0.290, format="%.4f")
            w_fra = st.number_input("Worst Fractal Dimension", value=0.083, format="%.4f")

    # Gom đủ 30 đặc trưng
    st.session_state.features = [
        m_rad, m_tex, m_per, m_are, m_smo, m_com, m_con, m_pts, m_sym, m_fra,
        r_err, t_err, p_err, a_err, s_err, com_err, con_err, pts_err, sym_err, fra_err,
        w_rad, w_tex, w_per, w_are, w_smo, w_com, w_con, w_pts, w_sym, w_fra
    ]

    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("⬅ Quay lại Bước 1"):
            st.session_state.step = 1
            st.rerun()
    with col_btn2:
        if st.button("Chẩn đoán AI ➔", type="primary"):
            st.session_state.step = 3
            st.rerun()

# ==================== BƯỚC 3: KẾT QUẢ & NGUY CƠ BỆNH LÝ ====================
elif st.session_state.step == 3:
    st.subheader("🩺 Bước 3: Đánh giá nguy cơ và kết quả chẩn đoán bệnh lý Ung thư Vú")
    
    features = st.session_state.get("features", [])
    
    if features:
        with st.spinner("Đang kết nối API Backend Render phân tích dữ liệu..."):
            try:
                response = requests.post(API_URL, json={"features": features}, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get("prediction")
                    
                    st.markdown("---")
                    if prediction == 0 or str(prediction).lower() in ["0", "malignant"]:
                        st.error("## ⚠️ KẾT QUẢ CHẨN ĐOÁN: NGUY CƠ U ÁC TÍNH (Malignant)")
                        st.warning("""
                        **Tác nhân & Khuyến cáo y tế:**
                        - Các chỉ số kích thước và diện tích khối U có dấu hiệu vượt ngưỡng bình thường.
                        - Bệnh nhân cần làm thêm xét nghiệm sinh thiết tế bào (Biopsy) và liên hệ bác sĩ chuyên khoa Ung Bướu ngay lập tức.
                        """)
                    else:
                        st.success("## ✅ KẾT QUẢ CHẨN ĐOÁN: U LÀNH TÍNH (Benign)")
                        st.info("""
                        **Tác nhân & Khuyến cáo y tế:**
                        - Khối U có các đặc trưng tế bào ở ngưỡng an toàn.
                        - Bệnh nhân nên duy trì khám định kỳ 6 tháng/lần để theo dõi diễn biến.
                        """)
                else:
                    st.error(f"Không thể lấy kết quả từ Backend Server (Status Code: {response.status_code})")
            except Exception as e:
                st.error(f"Lỗi kết nối tới Server API: {e}")
    
    if st.button("🔄 Thực hiện chẩn đoán lại từ đầu"):
        st.session_state.step = 1
        st.rerun()