import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd
import datetime
import qrcode
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

try:
    from tensorflow.keras.layers import Dense
    _dense_init_goc = Dense.__init__

    def _dense_init_da_va(self, *args, **kwargs):
        if 'quantization_config' in kwargs:
            kwargs.pop('quantization_config')
        _dense_init_goc(self, *args, **kwargs)

    Dense.__init__ = _dense_init_da_va
except Exception:
    pass

st.set_page_config(page_title="Hệ Thống POS 3I", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #F1F5F9; }
    .header-text { font-size: 46px !important; font-weight: 900; color: #1E3A8A; text-align: center; margin-bottom: 30px; text-transform: uppercase; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    .bill-panel { background-color: white; padding: 30px; border-radius: 15px; box-shadow: 0 10px 20px -5px rgba(0,0,0,0.15); height: 100%; border-top: 6px solid #1E3A8A; }
    .total-bill { font-size: 38px !important; font-weight: 900; color: #DC2626; text-align: center; background: #FEF2F2; padding: 25px; border-radius: 12px; margin-top: 25px; border: 2px dashed #DC2626;}
    .camera-box { background-color: white; border: 2px solid #E2E8F0; border-radius: 15px; padding: 25px; box-shadow: 0 6px 10px -2px rgba(0,0,0,0.1); }
    
    .live-indicator {
        height: 24px; width: 24px; background-color: #22c55e; border-radius: 50%; display: inline-block;
        box-shadow: 0 0 14px #22c55e; animation: blink 1.2s infinite; margin-left: 15px; vertical-align: middle;
    }
    @keyframes blink { 0% {opacity: 1; transform: scale(1);} 50% {opacity: 0.4; transform: scale(0.85);} 100% {opacity: 1; transform: scale(1);} }
    
    table { font-size: 20px !important; }
    th { font-size: 22px !important; color: #1E3A8A !important;}
    h3 { font-size: 32px !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-text">🏪 TRẠM THANH TOÁN TỰ ĐỘNG 3I 🏪</div>', unsafe_allow_html=True)

DANH_SACH_MON = {
    0: 'Banh Egg Tart',
    1: 'Banh Cookies Dua',
    2: 'Banh chuoi nuong',
    3: 'Banh Muffin Viet Quat',
    4: 'Banh mi dua luoi',
    5: 'Croissant',
    6: 'Banh da lon',
    7: 'Banh mi bo(cua lon)',
    8: 'Banh Patechaud',
    9: 'Cha bong cay'
}

GIA_TIEN = {
    'Banh Egg Tart': 21000,
    'Croissant': 30000,
    'Banh Cookies Dua': 23000,
    'Cha bong cay': 27000,
    'Banh Patechaud': 30000,
    'Banh mi dua luoi': 15000,
    'Banh da lon': 23000,
    'Banh mi bo(cua lon)': 18000,
    'Banh Muffin Viet Quat': 25000,
    'Banh chuoi nuong': 19000
}

DUONG_DAN_KERAS = "mo_hinh_banh_3I.keras"
SO_BANH_TOI_DA = 15
NGUONG_TIN_CAY = 0.05 

@st.cache_resource(show_spinner="Đang khởi động AI Keras...")
def tai_mo_hinh():
    m_keras = load_model(DUONG_DAN_KERAS) if os.path.exists(DUONG_DAN_KERAS) else None
    return m_keras

model_3I_Keras = tai_mo_hinh()

if model_3I_Keras is None:
    st.error(f"Khong tim thay {DUONG_DAN_KERAS}")
    st.stop()

if "raw_boxes" not in st.session_state:
    st.session_state.raw_boxes = []
if "ai_names" not in st.session_state:
    st.session_state.ai_names = []
if "ai_confs" not in st.session_state:
    st.session_state.ai_confs = []
if "da_quet" not in st.session_state:
    st.session_state.da_quet = False

def can_bang_sang_CLAHE(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def tinh_IoU(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / float(union_area) if union_area > 0 else 0

def phan_loai_bang_CNN_full(img_rgb, box):
    x, y, w, h = box
    crop_img = img_rgb[y:y+h, x:x+w]
    
    if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
        return None

    anh_chuan = cv2.resize(crop_img, (280, 280))
    anh_mang = img_to_array(anh_chuan) / 255.0
    input_img = np.expand_dims(anh_mang, axis=0)
    
    pred_keras = model_3I_Keras.predict(input_img, verbose=0)
    return pred_keras[0]

# BỎ HẲN REMBG, CHỈ DÙNG OPENCV LỌC MÀU ĐỂ XÓA TÀNG HÌNH KHAY INOX
def tach_vat_the_bang_HSV_OpenCV(img_pil):
    img_rgb = np.array(img_pil.convert("RGB"))
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Khay inox là kim loại xám/trắng nên độ bão hòa màu (Saturation) rất thấp.
    # Các loại bánh (Vàng, Xanh, Nâu) có độ bão hòa cao.
    s_channel = hsv[:, :, 1]
    
    # Ngưỡng 65: Bỏ qua mọi thứ có màu nhạt/xám/bạc (bao gồm cả bóng đen và khay)
    _, mask = cv2.threshold(s_channel, 65, 255, cv2.THRESH_BINARY)
    
    # Xử lý hình thái học để gom khối bánh và xóa rác
    loi_loc_nho = np.ones((5, 5), np.uint8)
    loi_loc_to = np.ones((9, 9), np.uint8)
    
    # 1. Bào mòn mạnh để cắt đứt các vệt sáng/bóng râm nối giữa các bánh
    mask = cv2.erode(mask, loi_loc_nho, iterations=2)
    # 2. Lấp đầy các lỗ hổng trên mặt bánh (do những chỗ bánh bị cháy đen nhạt màu)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, loi_loc_to, iterations=3)
    # 3. Phình to ra một xíu để trả lại viền bánh nguyên vẹn
    mask = cv2.dilate(mask, loi_loc_nho, iterations=1)
    
    duong_bao, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_boxes = []
    h_img, w_img = img_rgb.shape[:2]
    
    for c in duong_bao:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        
        # Chỉ lấy các vật thể có diện tích từ 1.5% đến 30% bức ảnh (chuẩn 1 cái bánh)
        # Khung sẽ không bao giờ ôm cả cái khay to nữa
        if w >= 40 and h >= 40 and (h_img * w_img * 0.015) <= area <= (h_img * w_img * 0.3):
            # Nới lỏng nhẹ 5 pixel để khung ôm đẹp mép bánh
            pad = 5
            x_new = max(0, x - pad)
            y_new = max(0, y - pad)
            w_new = w + pad*2
            h_new = h + pad*2
            
            raw_boxes.append({"box": (int(x_new), int(y_new), int(w_new), int(h_new)), "area": area})
            
    raw_boxes.sort(key=lambda b: b["area"], reverse=True)
    
    # Xóa các khung đè lên nhau
    accepted_boxes = []
    for raw in raw_boxes:
        box = raw["box"]
        is_duplicate = False
        for acc in accepted_boxes:
            if tinh_IoU(box, acc) > 0.45: 
                is_duplicate = True
                break
        if not is_duplicate and len(accepted_boxes) < SO_BANH_TOI_DA:
            accepted_boxes.append(box)
            
    return accepted_boxes

col_cam, col_bill = st.columns([0.65, 0.35])

with col_cam:
    st.markdown('<div class="camera-box">', unsafe_allow_html=True)
    st.markdown('<h3>📸 Camera Quét Mâm Bánh Tự Động <span class="live-indicator"></span></h3>', unsafe_allow_html=True)
    
    tab_camera, tab_upload = st.tabs(["🔴 Mở Camera POS", "📂 Tải Ảnh Khay Bánh"])
    with tab_camera:
        picture = st.camera_input("Đưa mâm bánh vào khung hình")
        if picture is not None: 
            st.session_state.anh_giu_lai = Image.open(picture)
    with tab_upload:
        uploaded_file = st.file_uploader("Kéo thả ảnh mâm bánh", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.session_state.anh_giu_lai = Image.open(uploaded_file)
            st.image(st.session_state.anh_giu_lai, use_container_width=True)
            
    nut_quet = st.button("🚀 BẮT ĐẦU QUÉT HÓA ĐƠN", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

if nut_quet and ("anh_giu_lai" in st.session_state):
    with st.spinner("AI đang khử nhiễu sáng và lập hóa đơn..."):
        img_pil = st.session_state.anh_giu_lai.convert("RGB")
        img_rgb_goc = np.array(img_pil)
        
        # ĐỔI SANG THUẬT TOÁN BẮT MÀU ĐỂ XÓA KHAY INOX
        # Gọi hàm thuật toán lọc màu mới tại đây
        st.session_state.raw_boxes = tach_vat_the_bang_HSV_OpenCV(img_pil)
        
        img_rgb_can_bang = can_bang_sang_CLAHE(img_rgb_goc)
        
        st.session_state.ai_names = []
        st.session_state.ai_confs = []
        
        danh_sach_nhan_da_dung = []
        
        so_luong_hop = len(st.session_state.raw_boxes)
        i_hop = 0
        while i_hop < so_luong_hop:
            box_item = st.session_state.raw_boxes[i_hop]
            mang_du_doan = phan_loai_bang_CNN_full(img_rgb_can_bang, box_item)
            
            if mang_du_doan is None:
                st.session_state.ai_names.append("Khong xac dinh")
                st.session_state.ai_confs.append(0.0)
                i_hop += 1
                continue
                
            idx_k = -1
            max_val = -1.0
            
            idx_chay = 0
            while idx_chay < 10:
                if idx_chay not in danh_sach_nhan_da_dung:
                    if mang_du_doan[idx_chay] > max_val:
                        max_val = mang_du_doan[idx_chay]
                        idx_k = idx_chay
                idx_chay += 1
                
            if idx_k != -1:
                danh_sach_nhan_da_dung.append(idx_k)
                conf = float(max_val)
                if conf > NGUONG_TIN_CAY:
                    st.session_state.ai_names.append(DANH_SACH_MON[idx_k])
                    st.session_state.ai_confs.append(conf)
                else:
                    st.session_state.ai_names.append("Khong xac dinh")
                    st.session_state.ai_confs.append(conf)
            else:
                st.session_state.ai_names.append("Khong xac dinh")
                st.session_state.ai_confs.append(0.0)
                
            i_hop += 1
            
        st.session_state.da_quet = True

with col_bill:
    st.markdown('<div class="bill-panel">', unsafe_allow_html=True)
    st.subheader("🧾 HÓA ĐƠN BÁN LẺ")
    
    if st.session_state.da_quet and ("anh_giu_lai" in st.session_state):
        st.markdown(f"**Thời gian:** {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        
        img_pil = st.session_state.anh_giu_lai.convert("RGB")
        img_rgb = np.array(img_pil)
        img_output = img_rgb.copy()
        
        danh_sach_hop = st.session_state.raw_boxes
        tong_so_hop = len(danh_sach_hop)
        
        invoice_items = []
        tong_tien = 0
        
        chi_so_hop = 0
        while chi_so_hop < tong_so_hop:
            box = danh_sach_hop[chi_so_hop]
            x, y, w, h = box
            
            ten_mon_ai = st.session_state.ai_names[chi_so_hop]
            conf_ai = st.session_state.ai_confs[chi_so_hop]
            
            if ten_mon_ai != "Khong xac dinh":
                gia_mon = GIA_TIEN[ten_mon_ai]
                tong_tien += gia_mon
                invoice_items.append({"Tên Món": ten_mon_ai, "Độ Tự Tin": f"{conf_ai*100:.1f}%", "Giá": f"{gia_mon:,} đ"})
            
                color_box = (255, 229, 0)
                cv2.rectangle(img_output, (x, y), (x+w, y+h), color_box, 4)
                
                label = f"{ten_mon_ai} ({conf_ai*100:.0f}%)"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.rectangle(img_output, (x, max(0, y - th - 12)), (x + tw, y), color_box, -1)
                cv2.putText(img_output, label, (x, max(20, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            
            chi_so_hop += 1
            
        if len(invoice_items) > 0:
            df = pd.DataFrame(invoice_items)
            df.index += 1
            st.table(df)
            
            st.markdown(f'<div class="total-bill">TỔNG CỘNG:<br>{tong_tien:,} VNĐ</div>', unsafe_allow_html=True)
            
            qr_content = f"Thanh toan Tiem Banh 3I - Tong tien: {tong_tien} VND"
            qr = qrcode.QRCode(box_size=10, border=1)
            qr.add_data(qr_content)
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color="black", back_color="#ffffff")
            
            col_qr1, col_qr2, col_qr3 = st.columns([1, 3, 1])
            with col_qr2:
                st.image(qr_img.get_image(), width=250, use_container_width=False)
                st.markdown("<p style='text-align: center; color: #64748B; font-size: 16px; font-weight: bold;'>Quét mã QR để thanh toán</p>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Trạm POS không nhận diện được món ăn nào.")
            
        with col_cam:
            st.markdown("### 🖼️ Ảnh Trực Tiếp Từ Trạm POS")
            st.image(img_output, use_container_width=True)
    else:
        st.info("Vui lòng chụp hoặc tải ảnh lên bên trái, sau đó bấm Quét AI.")
        
    st.markdown('</div>', unsafe_allow_html=True)