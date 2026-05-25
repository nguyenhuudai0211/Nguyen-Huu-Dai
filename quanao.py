import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
from keras.models import Sequential
from keras.layers import Dense
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Nhận Dạng Quần Áo - Nhóm 1", page_icon="🤖", layout="wide")

chuoi_css_giao_dien = """
<style>
.stApp { 
    background-color: #f8fafc; 
    font-family: 'Inter', sans-serif; 
}
.main-title { 
    text-align: center; 
    font-size: 42px; 
    font-weight: 900; 
    color: #1e3a8a; 
    margin-bottom: 25px; 
    text-transform: uppercase;
}
.custom-card { 
    background: #ffffff; 
    padding: 25px; 
    border-radius: 15px; 
    box-shadow: 0 10px 15px rgba(0,0,0,0.05); 
    margin-bottom: 20px; 
    border: 1px solid #e2e8f0; 
}
.sidebar-box {
    background: #eff6ff;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #bfdbfe;
    margin-bottom: 20px;
    color: #1e3a8a;
    font-weight: 600;
}
div.stButton > button:first-child { 
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); 
    color: #ffffff; 
    font-size: 18px; 
    font-weight: bold; 
    border-radius: 12px; 
    height: 60px; 
    border: none; 
    width: 100%; 
    box-shadow: 0 4px 6px rgba(37,99,235,0.3);
}
div.stButton > button:first-child:hover { 
    transform: translateY(-2px); 
    box-shadow: 0 8px 15px rgba(37,99,235,0.4);
}
.result-box-0 { 
    background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
    color: white; 
    font-size: 26px; 
    font-weight: 800; 
    text-align: center; 
    padding: 18px; 
    border-radius: 12px; 
    margin-bottom: 25px; 
}
.result-box-1 { 
    background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); 
    color: white; 
    font-size: 26px; 
    font-weight: 800; 
    text-align: center; 
    padding: 18px; 
    border-radius: 12px; 
    margin-bottom: 25px; 
}
.result-box-2 { 
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
    color: white; 
    font-size: 26px; 
    font-weight: 800; 
    text-align: center; 
    padding: 18px; 
    border-radius: 12px; 
    margin-bottom: 25px; 
}
.result-box-3 { 
    background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); 
    color: white; 
    font-size: 26px; 
    font-weight: 800; 
    text-align: center; 
    padding: 18px; 
    border-radius: 12px; 
    margin-bottom: 25px; 
}
.text-label {
    font-size: 16px;
    font-weight: 700;
    color: #334155;
    margin-bottom: 8px;
}
</style>
"""
st.markdown(chuoi_css_giao_dien, unsafe_allow_html=True)

with st.sidebar:
    tieu_de_sidebar_1 = "### 🎓 THÔNG TIN ĐỒ ÁN"
    st.markdown(tieu_de_sidebar_1)
    
    thong_tin_du_an = '<div class="sidebar-box">Môn học: Nền tảng AI và Ứng dụng<br><br>Nhóm thực hiện: Nhóm 1<br><br>Dự án: 3I</div>'
    st.markdown(thong_tin_du_an, unsafe_allow_html=True)
    
    duong_ke_ngang_1 = "---"
    st.markdown(duong_ke_ngang_1)
    
    tieu_de_sidebar_2 = "### ⚙️ CẤU HÌNH BÚT VẼ"
    st.markdown(tieu_de_sidebar_2)
    
    huong_dan_net_but = "Tùy chỉnh độ dày nét mực (Stroke Width):"
    gia_tri_min_but = 1
    gia_tri_max_but = 30
    gia_tri_mac_dinh_but = 16
    buoc_nhay_but = 1
    
    do_day_net_but = st.slider(
        huong_dan_net_but,
        min_value=gia_tri_min_but,
        max_value=gia_tri_max_but,
        value=gia_tri_mac_dinh_but,
        step=buoc_nhay_but
    )
    
    huong_dan_mau_sac = "Tùy chọn màu mực:"
    mau_mac_dinh = "#000000"
    mau_sac_net_but = st.color_picker(huong_dan_mau_sac, mau_mac_dinh)
    
    duong_ke_ngang_2 = "---"
    st.markdown(duong_ke_ngang_2)
    
    tieu_de_sidebar_3 = "### 🧠 CẤU TRÚC MẠNG NƠ-RON"
    st.markdown(tieu_de_sidebar_3)
    
    chuoi_cau_truc = "Input: (28*28,)\nLớp ẩn 1: Dense(512, relu)\nLớp ẩn 2: Dense(256, relu)\nOutput: Dense(4, softmax)"
    ngon_ngu_code = "python"
    st.code(chuoi_cau_truc, language=ngon_ngu_code)

tieu_de_chinh = '<div class="main-title">HỆ THỐNG PHÂN TÍCH QUẦN ÁO AI</div>'
st.markdown(tieu_de_chinh, unsafe_allow_html=True)

khoang_trong_1 = "<br>"
st.markdown(khoang_trong_1, unsafe_allow_html=True)

@st.cache_resource
def huan_luyen_mo_hinh_ann():
    model = Sequential()
    
    so_noron_vao = 512
    ham_kich_hoat_vao = 'relu'
    kich_thuoc_dau_vao = (28*28,)
    lop_vao = Dense(so_noron_vao, activation=ham_kich_hoat_vao, input_shape=kich_thuoc_dau_vao)
    model.add(lop_vao)
    
    so_noron_an = 256
    ham_kich_hoat_an = 'relu'
    lop_an = Dense(so_noron_an, activation=ham_kich_hoat_an)
    model.add(lop_an)
    
    so_noron_ra = 4
    ham_kich_hoat_ra = 'softmax'
    lop_ra = Dense(so_noron_ra, activation=ham_kich_hoat_ra)
    model.add(lop_ra)
    
    toan_tu_toi_uu = 'rmsprop'
    ham_mat_mat = 'categorical_crossentropy'
    danh_gia = ['accuracy']
    
    model.compile(optimizer=toan_tu_toi_uu, loss=ham_mat_mat, metrics=danh_gia)
    
    ten_file_trong_so = "model_nhom1_chuan.weights.h5"
    kiem_tra_file = os.path.exists(ten_file_trong_so)
    
    if kiem_tra_file:
        model.load_weights(ten_file_trong_so)
        return model

    duong_dan_thu_muc = r"C:\Ổ D tạm\UEH\Nền tảng trí tuệ nhân tạo và ứng dụng\quan_ao_ve_tay_truyen_thong\anh train"
    
    nhan_suy_luan = 'inferred'
    che_do_nhan = 'categorical'
    che_do_mau = 'grayscale'
    kich_thuoc_anh = (28, 28)
    kich_thuoc_batch = 32
    
    dataset_goc = tf.keras.utils.image_dataset_from_directory(
        duong_dan_thu_muc,
        labels=nhan_suy_luan,
        label_mode=che_do_nhan,
        color_mode=che_do_mau,
        image_size=kich_thuoc_anh,
        batch_size=kich_thuoc_batch
    )
    
    def tien_xu_ly(anh, nhan):
        gia_tri_mau_toi_da = 255.0
        anh_dao_mau = gia_tri_mau_toi_da - anh
        anh_chuan_hoa = anh_dao_mau / gia_tri_mau_toi_da
        kich_thuoc_moi = [-1, 28*28]
        anh_vector = tf.reshape(anh_chuan_hoa, kich_thuoc_moi)
        return anh_vector, nhan
        
    dataset_da_xu_ly = dataset_goc.map(tien_xu_ly)
    
    so_vong_lap = 30
    model.fit(dataset_da_xu_ly, epochs=so_vong_lap)
    
    model.save_weights(ten_file_trong_so)
    return model

model = huan_luyen_mo_hinh_ann()

ti_le_cot_1 = 1.2
ti_le_cot_2 = 0.1
ti_le_cot_3 = 1.3
cot_trai, cot_giua, cot_phai = st.columns([ti_le_cot_1, ti_le_cot_2, ti_le_cot_3])

with cot_trai:
    mo_the_card_trai = '<div class="custom-card">'
    st.markdown(mo_the_card_trai, unsafe_allow_html=True)
    
    tieu_de_bang_ve = "<h3>🎨 Bảng vẽ điện tử tương tác</h3>"
    st.markdown(tieu_de_bang_ve, unsafe_allow_html=True)
    
    huong_dan_ve = "Phác họa trang phục ngay trung tâm để đạt độ chính xác cao nhất:"
    st.write(huong_dan_ve)
    
    mau_nen_canvas_rgb = "rgb(255, 255, 255)"
    mau_nen_canvas_hex = "#ffffff"
    chieu_cao_canvas = 400
    chieu_rong_canvas = 400
    che_do_ve = "freedraw"
    khoa_canvas = "canvas_chinh"
    
    canvas_result = st_canvas(
        fill_color=mau_nen_canvas_rgb, 
        stroke_width=do_day_net_but, 
        stroke_color=mau_sac_net_but,
        background_color=mau_nen_canvas_hex, 
        height=chieu_cao_canvas, 
        width=chieu_rong_canvas,
        drawing_mode=che_do_ve, 
        key=khoa_canvas,
    )
    
    khoang_trong_2 = "<br>"
    st.markdown(khoang_trong_2, unsafe_allow_html=True)
    
    chu_nut_bam = "🚀 PHÂN TÍCH VÀ NHẬN DẠNG"
    kiem_tra_nut_bam = st.button(chu_nut_bam)
    
    dong_the_card_trai = '</div>'
    st.markdown(dong_the_card_trai, unsafe_allow_html=True)

with cot_phai:
    kiem_tra_ton_tai_du_lieu = canvas_result.image_data is not None

    if kiem_tra_ton_tai_du_lieu:
        if kiem_tra_nut_bam:
            thong_bao_1 = "Đang trích xuất cấu trúc hình học..."
            icon_1 = "🧠"
            st.toast(thong_bao_1, icon=icon_1)
            
            thoi_gian_cho = 0.5
            time.sleep(thoi_gian_cho)
            
            kieu_du_lieu_anh = np.uint8
            mang_anh_mau = np.array(canvas_result.image_data, dtype=kieu_du_lieu_anh)
            
            he_mau_rgba_sang_bgr = cv2.COLOR_RGBA2BGR
            anh_chuyen_doi = cv2.cvtColor(mang_anh_mau, he_mau_rgba_sang_bgr)
            img = anh_chuyen_doi
            
            he_mau_bgr_sang_gray = cv2.COLOR_BGR2GRAY
            img_rbg = cv2.cvtColor(img, he_mau_bgr_sang_gray)
            
            img_gray = img_rbg
            kich_thuoc_thu_nho = (28, 28)
            img_gray = cv2.resize(img_gray, kich_thuoc_thu_nho)
            
            muc_dao_mau = 255
            img_inverted = muc_dao_mau - img_gray
            
            hinh_dang_ma_tran_moi = (1, 28*28)
            img_ready = img_inverted.reshape(hinh_dang_ma_tran_moi)
            
            kieu_du_lieu_mo_hinh = 'float32'
            img_ready = img_ready.astype(kieu_du_lieu_mo_hinh)
            
            img_ready /= muc_dao_mau
            
            preds = model.predict(img_ready)
            digit = np.argmax(preds)
            
            chi_so_mang = 0
            vi_tri_0 = 0
            vi_tri_1 = 1
            vi_tri_2 = 2
            vi_tri_3 = 3
            
            gia_tri_0 = float(preds[chi_so_mang][vi_tri_0])
            gia_tri_1 = float(preds[chi_so_mang][vi_tri_1])
            gia_tri_2 = float(preds[chi_so_mang][vi_tri_2])
            gia_tri_3 = float(preds[chi_so_mang][vi_tri_3])
            
            he_so_phan_tram = 100
            phan_tram_ao_coc = gia_tri_0 * he_so_phan_tram
            phan_tram_ao_dai = gia_tri_1 * he_so_phan_tram
            phan_tram_quan_dai = gia_tri_2 * he_so_phan_tram
            phan_tram_quan_dui = gia_tri_3 * he_so_phan_tram
            
            nhan_du_doan = ""
            css_class_ket_qua = ""
            
            if digit == 0:
                nhan_du_doan = "ÁO CỘC TAY (ÁO THUN) 👕"
                css_class_ket_qua = "result-box-0"
            elif digit == 1:
                nhan_du_doan = "ÁO DÀI TAY 🧥"
                css_class_ket_qua = "result-box-1"
            elif digit == 2:
                nhan_du_doan = "QUẦN DÀI 👖"
                css_class_ket_qua = "result-box-2"
            elif digit == 3:
                nhan_du_doan = "QUẦN ĐÙI 🩳"
                css_class_ket_qua = "result-box-3"
                
            chuoi_hien_thi_ket_qua = f'<div class="{css_class_ket_qua}">🎯 ĐÁP ÁN: {nhan_du_doan}</div>'
            st.markdown(chuoi_hien_thi_ket_qua, unsafe_allow_html=True)
            
            mo_the_card_ket_qua = '<div class="custom-card">'
            st.markdown(mo_the_card_ket_qua, unsafe_allow_html=True)
            
            tieu_de_xac_suat = "<h3>📊 Độ tin cậy (Confidence Score)</h3>"
            st.markdown(tieu_de_xac_suat, unsafe_allow_html=True)
            
            chuoi_ao_coc = f'<div class="text-label">👕 Áo cộc tay: {phan_tram_ao_coc:.2f}%</div>'
            st.markdown(chuoi_ao_coc, unsafe_allow_html=True)
            phan_tram_nguyen_0 = int(phan_tram_ao_coc)
            st.progress(phan_tram_nguyen_0)
            
            khoang_trong_3 = "<br>"
            st.markdown(khoang_trong_3, unsafe_allow_html=True)
            
            chuoi_ao_dai = f'<div class="text-label">🧥 Áo dài tay: {phan_tram_ao_dai:.2f}%</div>'
            st.markdown(chuoi_ao_dai, unsafe_allow_html=True)
            phan_tram_nguyen_1 = int(phan_tram_ao_dai)
            st.progress(phan_tram_nguyen_1)
            
            khoang_trong_4 = "<br>"
            st.markdown(khoang_trong_4, unsafe_allow_html=True)
            
            chuoi_quan_dai = f'<div class="text-label">👖 Quần dài: {phan_tram_quan_dai:.2f}%</div>'
            st.markdown(chuoi_quan_dai, unsafe_allow_html=True)
            phan_tram_nguyen_2 = int(phan_tram_quan_dai)
            st.progress(phan_tram_nguyen_2)
            
            khoang_trong_5 = "<br>"
            st.markdown(khoang_trong_5, unsafe_allow_html=True)
            
            chuoi_quan_dui = f'<div class="text-label">🩳 Quần đùi: {phan_tram_quan_dui:.2f}%</div>'
            st.markdown(chuoi_quan_dui, unsafe_allow_html=True)
            phan_tram_nguyen_3 = int(phan_tram_quan_dui)
            st.progress(phan_tram_nguyen_3)
            
            dong_the_card_ket_qua = '</div>'
            st.markdown(dong_the_card_ket_qua, unsafe_allow_html=True)
            
            mo_the_card_thi_giac = '<div class="custom-card">'
            st.markdown(mo_the_card_thi_giac, unsafe_allow_html=True)
            
            tieu_de_thi_giac = "<h3>👁️ Phân tích Computer Vision</h3>"
            st.markdown(tieu_de_thi_giac, unsafe_allow_html=True)
            
            chieu_ngang_bieu_do = 6
            chieu_cao_bieu_do = 3
            kich_thuoc_bieu_do = (chieu_ngang_bieu_do, chieu_cao_bieu_do)
            fig = plt.figure(figsize=kich_thuoc_bieu_do)
            
            mau_nen_bieu_do = 'white'
            fig.patch.set_facecolor(mau_nen_bieu_do)
            
            chi_so_hang = 1
            chi_so_cot = 2
            vi_tri_thu_nhat = 1
            vi_tri_thu_hai = 2
            
            plt.subplot(chi_so_hang, chi_so_cot, vi_tri_thu_nhat)
            plt.imshow(img)
            tieu_de_anh_goc = "anhr goc (RBG)"
            plt.title(tieu_de_anh_goc)
            
            plt.subplot(chi_so_hang, chi_so_cot, vi_tri_thu_hai)
            he_mau_xam = 'gray'
            plt.imshow(img_inverted, cmap=he_mau_xam)
            tieu_de_anh_du_doan = f"du dooan:{digit}"
            plt.title(tieu_de_anh_du_doan)
            
            st.pyplot(fig)
            
            dong_the_card_thi_giac = '</div>'
            st.markdown(dong_the_card_thi_giac, unsafe_allow_html=True)
            
            st.balloons()
        else:
            mo_the_card_cho = '<div class="custom-card" style="text-align:center; padding: 130px 0;">'
            st.markdown(mo_the_card_cho, unsafe_allow_html=True)
            
            chuoi_cho_doi = '<h4 style="color:#64748b;">Hệ thống đang sẵn sàng, hãy vẽ và nhấn kích hoạt... ⚡</h4>'
            st.markdown(chuoi_cho_doi, unsafe_allow_html=True)
            
            dong_the_card_cho = '</div>'
            st.markdown(dong_the_card_cho, unsafe_allow_html=True)