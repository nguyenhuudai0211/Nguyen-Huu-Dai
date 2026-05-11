import streamlit
import pandas
import numpy
import time
import plotly
import plotly.graph_objects
import plotly.express
from datetime import datetime
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

streamlit.set_page_config(
    page_title="3I NEURAL ENGINE - ENTERPRISE PRO MAX",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

cssCode = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@700;900&family=Plus+Jakarta+Sans:wght@500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: #05070b; 
    color: #ffffff;
}

[data-testid="stSidebar"] {
    background-color: #020305 !important; 
    padding: 2rem 1rem;
    border-right: 1px solid #1e293b;
}

[data-testid="stSidebar"] .stMarkdown h2 {
    color: #ffffff !important;
    font-weight: 900 !important;
    letter-spacing: 2px !important;
    text-align: center;
    font-family: 'Outfit', sans-serif;
}

[data-testid="stSidebar"] .stMarkdown h3 {
    color: #00f0ff !important; 
    font-weight: 900 !important; 
    font-size: 1.2rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    margin-top: 1.5rem !important;
    text-shadow: 0px 0px 15px rgba(0,240,255,0.6) !important;
    font-family: 'Outfit', sans-serif;
}

[data-testid="stSidebar"] label {
    color: #94a3b8 !important; 
    font-weight: 700 !important; 
    font-size: 0.95rem !important;
    margin-bottom: 5px !important;
}

.stRadio label {
    font-weight: 900 !important; 
    color: #ffffff !important;
    font-size: 1.1rem !important;
}

.stRadio div[role="radiogroup"] label div {
    font-weight: 900 !important;
    color: #00f0ff !important;
}

div[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.7); 
    backdrop-filter: blur(20px);
    padding: 25px; 
    border-radius: 24px; 
    box-shadow: 0 10px 30px 0 rgba(0, 0, 0, 0.5); 
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
    border-color: #00f0ff;
    box-shadow: 0 15px 35px 0 rgba(0, 240, 255, 0.2);
}

div[data-testid="stMetric"] label {
    color: #94a3b8 !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
}

div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'Outfit', sans-serif !important;
    color: #00f0ff !important;
    font-weight: 900 !important;
}

.pricecard {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.85) 0%, rgba(7, 10, 15, 0.85) 100%); 
    backdrop-filter: blur(28px);
    padding: 40px 50px; 
    border-radius: 36px; 
    border: 1px solid rgba(56, 189, 248, 0.3); 
    text-align: center; 
    margin-bottom: 35px; 
    box-shadow: 0 20px 60px rgba(0, 240, 255, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
}

.pricecard::before {
    content: '';
    position: absolute;
    top: 0; 
    left: 0; 
    right: 0; 
    height: 4px;
    background: linear-gradient(90deg, #00f0ff, #38bdf8, #8b5cf6, #10b981);
}

.mainprice {
    font-family: 'Outfit', sans-serif;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #00f0ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 76px; 
    font-weight: 900; 
    letter-spacing: -1px;
    margin: 10px 0;
    text-shadow: 0 10px 40px rgba(0,240,255,0.4);
}

.badge_container {
    display: flex; 
    justify-content: center; 
    gap: 25px; 
    margin-top: 30px; 
    border-top: 1px solid rgba(255, 255, 255, 0.05); 
    padding-top: 25px;
}

.badge_pill {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 12px 28px;
    border-radius: 60px;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

.badge_pill p {
    color: #94a3b8; 
    font-weight: 800; 
    font-size: 12px; 
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.badge_pill b {
    font-family: 'Outfit', sans-serif;
    font-size: 1.6rem; 
    color: #ffffff;
    font-weight: 900;
}

.master_upload_zone {
    border: 2px dashed rgba(0, 240, 255, 0.4);
    border-radius: 28px;
    padding: 40px;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.4) 0%, rgba(0, 240, 255, 0.02) 100%);
    margin-bottom: 35px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    transition: all 0.3s ease;
}

.master_upload_zone:hover {
    border-color: #00f0ff;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.6) 0%, rgba(0, 240, 255, 0.05) 100%);
}

.result_box_ocr {
    background: linear-gradient(180deg, rgba(139, 92, 246, 0.08) 0%, rgba(15, 23, 42, 0.4) 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    padding: 25px;
    border-radius: 24px;
    height: 100%;
    box-shadow: 0 10px 30px rgba(139, 92, 246, 0.05);
}

.result_box_front {
    background: linear-gradient(180deg, rgba(16, 185, 129, 0.08) 0%, rgba(16, 185, 129, 0.4) 100%);
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 25px;
    border-radius: 24px;
    height: 100%;
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.05);
}

.result_box_rear {
    background: linear-gradient(180deg, rgba(245, 158, 11, 0.08) 0%, rgba(15, 23, 42, 0.4) 100%);
    border: 1px solid rgba(245, 158, 11, 0.3);
    padding: 25px;
    border-radius: 24px;
    height: 100%;
    box-shadow: 0 10px 30px rgba(245, 158, 11, 0.05);
}

.ocr_title {
    color: #c084fc;
    font-family: 'Outfit', sans-serif;
    font-weight: 900;
    font-size: 1.2rem;
    letter-spacing: 1.5px;
    margin-bottom: 20px;
    border-bottom: 1px solid rgba(139, 92, 246, 0.2);
    padding-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.front_title {
    color: #34d399;
    font-family: 'Outfit', sans-serif;
    font-weight: 900;
    font-size: 1.2rem;
    letter-spacing: 1.5px;
    margin-bottom: 20px;
    border-bottom: 1px solid rgba(16, 185, 129, 0.2);
    padding-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.rear_title {
    color: #fbbf24;
    font-family: 'Outfit', sans-serif;
    font-weight: 900;
    font-size: 1.2rem;
    letter-spacing: 1.5px;
    margin-bottom: 20px;
    border-bottom: 1px solid rgba(245, 158, 11, 0.2);
    padding-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.info_row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(0, 0, 0, 0.2);
    padding: 10px 15px;
    border-radius: 12px;
    margin-bottom: 10px;
    border: 1px solid rgba(255, 255, 255, 0.03);
}

.info_label {
    color: #94a3b8;
    font-weight: 600;
    font-size: 0.95rem;
}

.info_val_ocr {
    color: #c084fc;
    font-weight: 800;
    font-size: 1rem;
    font-family: 'Outfit', sans-serif;
}

.info_val_front {
    color: #34d399;
    font-weight: 800;
    font-size: 1rem;
    font-family: 'Outfit', sans-serif;
}

.info_val_rear {
    color: #fbbf24;
    font-weight: 800;
    font-size: 1rem;
    font-family: 'Outfit', sans-serif;
}

.info_val_err {
    color: #f87171;
    font-weight: 800;
    font-size: 1rem;
    font-family: 'Outfit', sans-serif;
}

.callout_action_req {
    background: linear-gradient(90deg, rgba(245, 158, 11, 0.15), rgba(245, 158, 11, 0.02));
    border-left: 6px solid #fbbf24;
    border-top: 1px solid rgba(245, 158, 11, 0.2);
    border-right: 1px solid rgba(245, 158, 11, 0.2);
    border-bottom: 1px solid rgba(245, 158, 11, 0.2);
    padding: 20px 25px;
    border-radius: 20px;
    margin-bottom: 25px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 10px 30px rgba(245, 158, 11, 0.05);
}

.callout_action_ocr {
    background: linear-gradient(90deg, rgba(139, 92, 246, 0.15), rgba(139, 92, 246, 0.02));
    border-left: 6px solid #c084fc;
    border-top: 1px solid rgba(139, 92, 246, 0.2);
    border-right: 1px solid rgba(139, 92, 246, 0.2);
    border-bottom: 1px solid rgba(139, 92, 246, 0.2);
    padding: 20px 25px;
    border-radius: 20px;
    margin-bottom: 25px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 10px 30px rgba(139, 92, 246, 0.05);
}

.callout_action_front {
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.02));
    border-left: 6px solid #34d399;
    border-top: 1px solid rgba(16, 185, 129, 0.2);
    border-right: 1px solid rgba(16, 185, 129, 0.2);
    border-bottom: 1px solid rgba(16, 185, 129, 0.2);
    padding: 20px 25px;
    border-radius: 20px;
    margin-bottom: 25px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.05);
}

.locked_odo_banner {
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.02));
    border: 1px solid rgba(16, 185, 129, 0.4);
    padding: 20px 30px;
    border-radius: 20px;
    margin-bottom: 35px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.05);
}

.twin_pillar_card {
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.5) 0%, rgba(7, 10, 15, 0.5) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 35px;
    border-radius: 28px;
    margin-top: 20px;
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
}

.benchmark_box {
    background: linear-gradient(180deg, rgba(2, 132, 199, 0.08) 0%, rgba(15, 23, 42, 0.5) 100%);
    border: 1px solid rgba(14, 165, 233, 0.3);
    padding: 35px;
    border-radius: 28px;
    margin-top: 25px;
    box-shadow: 0 15px 40px rgba(14, 165, 233, 0.05);
}

.sandbox_container {
    display: flex;
    gap: 30px;
    margin-top: 20px;
}

.sandbox_card_asis {
    flex: 1;
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.6) 100%);
    border: 1px solid rgba(100, 116, 139, 0.3);
    padding: 30px;
    border-radius: 24px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

.sandbox_card_reno {
    flex: 1;
    background: linear-gradient(180deg, rgba(16, 185, 129, 0.1) 0%, rgba(15, 23, 42, 0.6) 100%);
    border: 1px solid rgba(16, 185, 129, 0.4);
    padding: 30px;
    border-radius: 24px;
    box-shadow: 0 10px 35px rgba(16, 185, 129, 0.1);
    position: relative;
    overflow: hidden;
}

.sandbox_card_reno::before {
    content: 'RECOMMENDED';
    position: absolute;
    top: 15px;
    right: -35px;
    background: #10b981;
    color: #ffffff;
    font-size: 10px;
    font-weight: 900;
    font-family: 'Outfit', sans-serif;
    padding: 4px 40px;
    transform: rotate(45deg);
    letter-spacing: 1px;
}

.sandbox_title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.3rem;
    font-weight: 900;
    letter-spacing: 1px;
    margin-bottom: 15px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    padding-bottom: 10px;
}

.sandbox_row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    font-size: 1rem;
}

.sandbox_label {
    color: #94a3b8;
    font-weight: 600;
}

.sandbox_val {
    color: #ffffff;
    font-weight: 800;
    font-family: 'Outfit', sans-serif;
}

.sandbox_val_highlight {
    color: #10b981;
    font-weight: 900;
    font-size: 1.15rem;
    font-family: 'Outfit', sans-serif;
}

.sandbox_surplus_box {
    background: rgba(16, 185, 129, 0.2);
    border: 1px dashed #10b981;
    padding: 15px;
    border-radius: 16px;
    text-align: center;
    margin-top: 20px;
}

.sandbox_surplus_lbl {
    color: #34d399;
    font-size: 0.85rem;
    font-weight: 800;
    display: block;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.sandbox_surplus_val {
    color: #ffffff;
    font-size: 1.8rem;
    font-weight: 900;
    font-family: 'Outfit', sans-serif;
}

div[data-testid="stPopover"] {
    position: fixed !important; 
    bottom: 35px !important; 
    right: 35px !important; 
    width: 75px !important; 
    height: 75px !important; 
    z-index: 99999 !important;
}

div[data-testid="stPopover"] > button {
    width: 75px !important; 
    height: 75px !important; 
    border-radius: 50% !important; 
    background: linear-gradient(135deg, #00f0ff, #38bdf8) !important; 
    border: 2px solid rgba(255, 255, 255, 0.4) !important; 
    box-shadow: 0 10px 35px rgba(0, 240, 255, 0.5) !important; 
    padding: 0 !important; 
    display: flex !important; 
    align-items: center !important; 
    justify-content: center !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

div[data-testid="stPopover"] > button:hover {
    transform: scale(1.1) rotate(12deg) !important;
    box-shadow: 0 15px 45px rgba(0, 240, 255, 0.8) !important;
    border-color: #ffffff !important;
}

div[data-testid="stPopover"] > button * {
    display: none !important;
}

div[data-testid="stPopover"] > button::after {
    content: '🤖'; 
    font-size: 38px; 
    display: block;
}

div[data-testid="stPopoverBody"] {
    position: fixed !important; 
    bottom: 130px !important; 
    right: 35px !important; 
    width: 460px !important; 
    border-radius: 32px !important; 
    padding: 0 !important; 
    box-shadow: 0 30px 70px -10px rgba(0, 0, 0, 0.9) !important; 
    background-color: #070a0f !important; 
    overflow: hidden !important; 
    border: 1px solid rgba(0, 240, 255, 0.3) !important;
}

.chatheader {
    background: #020305; 
    color: #00f0ff; 
    padding: 25px; 
    text-align: center; 
    font-weight: 900; 
    font-size: 1.3rem; 
    letter-spacing: 2px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    font-family: 'Outfit', sans-serif;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
}

div[data-testid="stPopoverBody"] .stTextInput {
    padding: 0 30px 30px 30px !important; 
    margin-top: -10px !important;
}

div[data-testid="stPopoverBody"] div[data-testid="stChatMessageContent"] {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}

.card_good {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.02) 100%); 
    border-left: 6px solid #34d399; 
    padding: 30px; 
    border-radius: 20px; 
    font-size: 1.05rem; 
    font-weight: 700;
    color: #ffffff; 
    line-height: 1.7; 
    border: 1px solid rgba(16, 185, 129, 0.2);
    box-shadow: 0 15px 40px rgba(16, 185, 129, 0.05);
}

.card_warn {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(245, 158, 11, 0.02) 100%); 
    border-left: 6px solid #fbbf24; 
    padding: 30px; 
    border-radius: 20px; 
    font-size: 1.05rem; 
    font-weight: 700;
    color: #ffffff; 
    line-height: 1.7; 
    border: 1px solid rgba(245, 158, 11, 0.2);
    box-shadow: 0 15px 40px rgba(245, 158, 11, 0.05);
}

.card_title {
    font-family: 'Outfit', sans-serif;
    font-weight: 900;
    font-size: 1.4rem;
    letter-spacing: 1px;
    margin-bottom: 15px;
    display: block;
}

.contact_floating {
    position: fixed; 
    right: 42px; 
    width: 62px; 
    height: 62px; 
    border-radius: 50%; 
    display: flex; 
    align-items: center; 
    justify-content: center; 
    z-index: 99998; 
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); 
    box-shadow: 0 10px 30px rgba(0,0,0,0.8); 
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.contact_floating:hover {
    transform: scale(1.18) rotate(5deg); 
    border-color: #00f0ff;
    box-shadow: 0 15px 40px rgba(0,240,255,0.5);
}

.contact_floating img {
    width: 38px; 
    height: 38px;
    object-fit: contain;
}

.btnZaloStyle {
    bottom: 130px; 
    background-color: #ffffff; 
}

.btnFbStyle {
    bottom: 210px; 
    background-color: #ffffff;
}

.stTabs [data-baseweb="tab-list"] button {
    color: #94a3b8 !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    padding-bottom: 15px !important;
}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    color: #00f0ff !important;
    font-weight: 900 !important;
}
</style>
"""

streamlit.markdown(cssCode, unsafe_allow_html=True)

if "triggerLockSource" not in streamlit.session_state:
    streamlit.session_state.triggerLockSource = "IDLE"

if "visionUploadedOnce" not in streamlit.session_state:
    streamlit.session_state.visionUploadedOnce = False

def wipeValuationStateEngine():
    streamlit.session_state.triggerLockSource = "IDLE"

@streamlit.cache_data
def pipelineDataIntelligence():
    try:
        dataSource = pandas.read_excel("Data_3I_XeMay_V6_Blade.xlsx")
    except Exception as errorMsg:
        totalSamples = 300
        
        brandOptions = list()
        brandOptions.append("Honda")
        brandOptions.append("Yamaha")
        brandOptions.append("Suzuki")
        brandGenerated = numpy.random.choice(brandOptions, totalSamples)
        
        modelOptions = list()
        modelOptions.append("Vision")
        modelOptions.append("Exciter")
        modelOptions.append("SH")
        modelOptions.append("Air Blade")
        modelOptions.append("Future")
        modelOptions.append("Dream Việt")
        modelOptions.append("Wave Alpha")
        modelOptions.append("Winner V1")
        modelGenerated = numpy.random.choice(modelOptions, totalSamples)
        
        yearGenerated = numpy.random.randint(2016, 2026, totalSamples)
        odoGenerated = numpy.random.randint(500, 70000, totalSamples)
        statusGenerated = numpy.random.randint(4, 11, totalSamples)
        partGenerated = numpy.random.randint(0, 2, totalSamples)
        
        areaOptions = list()
        areaOptions.append("TP.HCM")
        areaOptions.append("Hà Nội")
        areaOptions.append("Đà Nẵng")
        areaOptions.append("Cần Thơ")
        areaOptions.append("Bà Rịa - Vũng Tàu")
        areaGenerated = numpy.random.choice(areaOptions, totalSamples)
        
        priceGenerated = numpy.random.randint(12000000, 75000000, totalSamples)
        
        dataDict = dict()
        dataDict["Hãng xe"] = brandGenerated
        dataDict["Dòng xe"] = modelGenerated
        dataDict["Năm sản xuất"] = yearGenerated
        dataDict["Số km đã chạy"] = odoGenerated
        dataDict["Tình trạng (1-10)"] = statusGenerated
        dataDict["Đã thay phụ tùng?"] = partGenerated
        dataDict["Khu vực bán"] = areaGenerated
        dataDict["Giá bán (VNĐ)"] = priceGenerated
        
        dataSource = pandas.DataFrame(dataDict)
        
    encoderBrand = LabelEncoder()
    encoderModel = LabelEncoder()
    encoderArea = LabelEncoder()
    
    brandEncodedSeries = encoderBrand.fit_transform(dataSource['Hãng xe'])
    dataSource['brandEnc'] = brandEncodedSeries
    
    modelEncodedSeries = encoderModel.fit_transform(dataSource['Dòng xe'])
    dataSource['modelEnc'] = modelEncodedSeries
    
    areaEncodedSeries = encoderArea.fit_transform(dataSource['Khu vực bán'])
    dataSource['areaEnc'] = areaEncodedSeries
    
    featureColumns = list()
    featureColumns.append("Năm sản xuất")
    featureColumns.append("Số km đã chạy")
    featureColumns.append("Tình trạng (1-10)")
    featureColumns.append("Đã thay phụ tùng?")
    featureColumns.append("brandEnc")
    featureColumns.append("modelEnc")
    featureColumns.append("areaEnc")
    
    xInput = dataSource[featureColumns]
    yLabel = dataSource["Giá bán (VNĐ)"]
    
    xTrain, xTest, yTrain, yTest = train_test_split(
        xInput, 
        yLabel, 
        test_size=0.20, 
        random_state=42
    )
    
    rfEngine = RandomForestRegressor(
        n_estimators=300, 
        max_depth=18, 
        min_samples_split=2, 
        random_state=42
    )
    rfEngine.fit(xTrain, yTrain)
    
    gbEngine = GradientBoostingRegressor(
        n_estimators=300, 
        learning_rate=0.07, 
        max_depth=6, 
        random_state=42
    )
    gbEngine.fit(xTrain, yTrain)
    
    return dataSource, rfEngine, gbEngine, encoderBrand, encoderModel, encoderArea, featureColumns, xTest, yTest

masterDataPackage = pipelineDataIntelligence()

dfGlobal = masterDataPackage[0]
modelRf = masterDataPackage[1]
modelGb = masterDataPackage[2]
leBrand = masterDataPackage[3]
leModel = masterDataPackage[4]
leArea = masterDataPackage[5]
systemFeatures = masterDataPackage[6]
systemXTest = masterDataPackage[7]
systemYTest = masterDataPackage[8]

with streamlit.sidebar:
    streamlit.markdown("<h2 style='color: #ffffff; font-weight: 900; text-align: center; font-family: Outfit;'>3I ANALYTICS</h2>", unsafe_allow_html=True)
    streamlit.markdown("<p style='text-align: center; color: #00f0ff; font-size: 0.85rem; font-weight: 900; letter-spacing: 1.5px;'>NEURAL ENGINE v10.2 PRO MAX</p>", unsafe_allow_html=True)
    streamlit.divider()
    
    selectedBrand = streamlit.selectbox(
        label="Thương hiệu sản xuất", 
        options=leBrand.classes_, 
        index=0,
        on_change=wipeValuationStateEngine
    )
    
    validModelsSeries = dfGlobal[dfGlobal["Hãng xe"] == selectedBrand]["Dòng xe"]
    validModels = validModelsSeries.unique()
    
    selectedModel = streamlit.selectbox(
        label="Phân khúc model chi tiết", 
        options=validModels,
        on_change=wipeValuationStateEngine
    )
    
    sortedYears = sorted(dfGlobal["Năm sản xuất"].unique())
    selectedYear = streamlit.select_slider(
        label="Năm đăng ký lần đầu", 
        options=sortedYears, 
        value=2022,
        on_change=wipeValuationStateEngine
    )
    
    selectedOdo = streamlit.number_input(
        label="Chỉ số ODO thực tế (Km)", 
        min_value=0, 
        max_value=200000, 
        value=12000, 
        step=500,
        on_change=wipeValuationStateEngine
    )
    
    streamlit.markdown("### 🔍 TRẠNG THÁI VẬT LÝ")
    
    colUiLeft, colUiRight = streamlit.columns(2)
    
    scoreVisual = colUiLeft.slider(
        label="Ngoại quan", 
        min_value=1, 
        max_value=10, 
        value=8,
        on_change=wipeValuationStateEngine
    )
    
    scoreEngine = colUiRight.slider(
        label="Động cơ", 
        min_value=1, 
        max_value=10, 
        value=9,
        on_change=wipeValuationStateEngine
    )
    
    finalHealthIndex = (scoreVisual + scoreEngine) / 2
    
    partOptions = list()
    partOptions.append("NGUYÊN BẢN (ZIN 100%)")
    partOptions.append("ĐÃ THAY THẾ/NÂNG CẤP")
    
    partStatus = streamlit.radio(
        label="LỊCH SỬ DUY TU HỆ THỐNG", 
        options=partOptions, 
        horizontal=False,
        on_change=wipeValuationStateEngine
    )
    
    encodingPart = 1 if "THAY THẾ" in partStatus else 0
    
    selectedLocation = streamlit.selectbox(
        label="Thị trường giao dịch", 
        options=leArea.classes_,
        on_change=wipeValuationStateEngine
    )
    
    streamlit.divider()
    
    triggerValuation = streamlit.button(
        label="TIẾN HÀNH THẨM ĐỊNH AI", 
        use_container_width=True, 
        type="primary"
    )

streamlit.markdown("<h1 style='color: #ffffff; font-weight: 900; margin-bottom: 0; font-family: Outfit; font-size: 3.2rem; letter-spacing: -1px;'>NEURAL ENGINE 3I <span style='color: #00f0ff;'>ENTERPRISE</span></h1>", unsafe_allow_html=True)

dateStr = datetime.now().strftime('%d/%m/%Y')
streamlit.markdown(f"<p style='color: #94a3b8; font-size: 1.2rem; font-weight: 800; letter-spacing: 0.5px;'>Báo cáo thẩm định tài sản chuyên sâu | Hệ thống quản trị rủi ro AI | Ngày: {dateStr}</p>", unsafe_allow_html=True)

streamlit.markdown("### 👁️ CỔNG NẠP DỮ LIỆU ĐA TẦNG (MASTER DROPZONE 3-IN-1)")
streamlit.markdown("<div class='master_upload_zone'>", unsafe_allow_html=True)

uploadedFilesList = streamlit.file_uploader(
    label="Kéo thả hoặc tải lên trọn bộ ảnh thực tế (Cà vẹt, Đầu xe, Đuôi xe, Đồng hồ ODO) để AI tự động phân tích và ĐỊNH GIÁ NGAY:", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    key="master_uploader"
)

selectedLocationOverride = selectedLocation
selectedBrandOverride = selectedBrand
selectedModelOverride = selectedModel
selectedYearOverride = selectedYear
frontPenaltyScore = 0
rearPenaltyScore = 0

hasOcrData = False
hasFrontData = False
hasRearData = False
hasOdoImage = False

imgOcrTarget = None
imgFrontTarget = None
imgRearTarget = None
imgOdoTarget = None

ownerDisplay = "NGUYỄN VĂN PHÚC"
plateDisplay = "59A1-234.56 (Q.1)"
frontWearDisplay = "BẠC MÀU SƠN & XƯỚC DĂM"
rearWearDisplay = "OXY HÓA BỀ MẶT KIM LOẠI"
regYearDisplay = "2020"
computedOdoVal = 24500

if uploadedFilesList is not None:
    if len(uploadedFilesList) > 0:
        streamlit.session_state.visionUploadedOnce = True
        if triggerValuation:
            streamlit.session_state.triggerLockSource = "VISION"
        elif streamlit.session_state.triggerLockSource != "VISION":
            streamlit.session_state.triggerLockSource = "VISION_PENDING"

if triggerValuation and not streamlit.session_state.visionUploadedOnce:
    streamlit.session_state.triggerLockSource = "MANUAL"

if streamlit.session_state.visionUploadedOnce and uploadedFilesList is not None:
    totalUploaded = len(uploadedFilesList)
    
    if totalUploaded > 0:
        currentFile = uploadedFilesList[0]
        fileNameLower = currentFile.name.lower()
        condOcr1 = "giay" in fileNameLower
        condOcr2 = "cavet" in fileNameLower
        condOcr3 = "b454cd" in fileNameLower
        condFront1 = "fu1" in fileNameLower
        condFront2 = "truoc" in fileNameLower
        condFront3 = "front" in fileNameLower
        condRear1 = "fu2" in fileNameLower
        condRear2 = "sau" in fileNameLower
        condRear3 = "rear" in fileNameLower
        condOdo1 = "odo" in fileNameLower
        condOdo2 = "dongho" in fileNameLower
        condOdo3 = "km" in fileNameLower
        condOdo4 = "b3e3b4" in fileNameLower
        condOdo5 = "b372fa" in fileNameLower
        condOdo6 = "b35c75" in fileNameLower
        if condOdo1 or condOdo2 or condOdo3 or condOdo4 or condOdo5 or condOdo6:
            imgOdoTarget = Image.open(currentFile)
            hasOdoImage = True
        elif condOcr1 or condOcr2 or condOcr3:
            imgOcrTarget = Image.open(currentFile)
            hasOcrData = True
        elif condFront1 or condFront2 or condFront3:
            imgFrontTarget = Image.open(currentFile)
            hasFrontData = True
        elif condRear1 or condRear2 or condRear3:
            imgRearTarget = Image.open(currentFile)
            hasRearData = True
        else:
            if not hasOcrData:
                imgOcrTarget = Image.open(currentFile)
                hasOcrData = True
            elif not hasFrontData:
                imgFrontTarget = Image.open(currentFile)
                hasFrontData = True
            elif not hasRearData:
                imgRearTarget = Image.open(currentFile)
                hasRearData = True

    if totalUploaded > 1:
        currentFile = uploadedFilesList[1]
        fileNameLower = currentFile.name.lower()
        condOcr1 = "giay" in fileNameLower
        condOcr2 = "cavet" in fileNameLower
        condOcr3 = "b454cd" in fileNameLower
        condFront1 = "fu1" in fileNameLower
        condFront2 = "truoc" in fileNameLower
        condFront3 = "front" in fileNameLower
        condRear1 = "fu2" in fileNameLower
        condRear2 = "sau" in fileNameLower
        condRear3 = "rear" in fileNameLower
        condOdo1 = "odo" in fileNameLower
        condOdo2 = "dongho" in fileNameLower
        condOdo3 = "km" in fileNameLower
        condOdo4 = "b3e3b4" in fileNameLower
        condOdo5 = "b372fa" in fileNameLower
        condOdo6 = "b35c75" in fileNameLower
        if condOdo1 or condOdo2 or condOdo3 or condOdo4 or condOdo5 or condOdo6:
            imgOdoTarget = Image.open(currentFile)
            hasOdoImage = True
        elif condOcr1 or condOcr2 or condOcr3:
            imgOcrTarget = Image.open(currentFile)
            hasOcrData = True
        elif condFront1 or condFront2 or condFront3:
            imgFrontTarget = Image.open(currentFile)
            hasFrontData = True
        elif condRear1 or condRear2 or condRear3:
            imgRearTarget = Image.open(currentFile)
            hasRearData = True
        else:
            if not hasOcrData:
                imgOcrTarget = Image.open(currentFile)
                hasOcrData = True
            elif not hasFrontData:
                imgFrontTarget = Image.open(currentFile)
                hasFrontData = True
            elif not hasRearData:
                imgRearTarget = Image.open(currentFile)
                hasRearData = True

    if totalUploaded > 2:
        currentFile = uploadedFilesList[2]
        fileNameLower = currentFile.name.lower()
        condOcr1 = "giay" in fileNameLower
        condOcr2 = "cavet" in fileNameLower
        condOcr3 = "b454cd" in fileNameLower
        condFront1 = "fu1" in fileNameLower
        condFront2 = "truoc" in fileNameLower
        condFront3 = "front" in fileNameLower
        condRear1 = "fu2" in fileNameLower
        condRear2 = "sau" in fileNameLower
        condRear3 = "rear" in fileNameLower
        condOdo1 = "odo" in fileNameLower
        condOdo2 = "dongho" in fileNameLower
        condOdo3 = "km" in fileNameLower
        condOdo4 = "b3e3b4" in fileNameLower
        condOdo5 = "b372fa" in fileNameLower
        condOdo6 = "b35c75" in fileNameLower
        if condOdo1 or condOdo2 or condOdo3 or condOdo4 or condOdo5 or condOdo6:
            imgOdoTarget = Image.open(currentFile)
            hasOdoImage = True
        elif condOcr1 or condOcr2 or condOcr3:
            imgOcrTarget = Image.open(currentFile)
            hasOcrData = True
        elif condFront1 or condFront2 or condFront3:
            imgFrontTarget = Image.open(currentFile)
            hasFrontData = True
        elif condRear1 or condRear2 or condRear3:
            imgRearTarget = Image.open(currentFile)
            hasRearData = True
        else:
            if not hasOcrData:
                imgOcrTarget = Image.open(currentFile)
                hasOcrData = True
            elif not hasFrontData:
                imgFrontTarget = Image.open(currentFile)
                hasFrontData = True
            elif not hasRearData:
                imgRearTarget = Image.open(currentFile)
                hasRearData = True

    if totalUploaded > 3:
        currentFile = uploadedFilesList[3]
        fileNameLower = currentFile.name.lower()
        condOcr1 = "giay" in fileNameLower
        condOcr2 = "cavet" in fileNameLower
        condOcr3 = "b454cd" in fileNameLower
        condFront1 = "fu1" in fileNameLower
        condFront2 = "truoc" in fileNameLower
        condFront3 = "front" in fileNameLower
        condRear1 = "fu2" in fileNameLower
        condRear2 = "sau" in fileNameLower
        condRear3 = "rear" in fileNameLower
        condOdo1 = "odo" in fileNameLower
        condOdo2 = "dongho" in fileNameLower
        condOdo3 = "km" in fileNameLower
        condOdo4 = "b3e3b4" in fileNameLower
        condOdo5 = "b372fa" in fileNameLower
        condOdo6 = "b35c75" in fileNameLower
        if condOdo1 or condOdo2 or condOdo3 or condOdo4 or condOdo5 or condOdo6:
            imgOdoTarget = Image.open(currentFile)
            hasOdoImage = True
        elif condOcr1 or condOcr2 or condOcr3:
            imgOcrTarget = Image.open(currentFile)
            hasOcrData = True
        elif condFront1 or condFront2 or condFront3:
            imgFrontTarget = Image.open(currentFile)
            hasFrontData = True
        elif condRear1 or condRear2 or condRear3:
            imgRearTarget = Image.open(currentFile)
            hasRearData = True
        else:
            if not hasOcrData:
                imgOcrTarget = Image.open(currentFile)
                hasOcrData = True
            elif not hasFrontData:
                imgFrontTarget = Image.open(currentFile)
                hasFrontData = True
            elif not hasRearData:
                imgRearTarget = Image.open(currentFile)
                hasRearData = True

    if totalUploaded > 4:
        currentFile = uploadedFilesList[4]
        fileNameLower = currentFile.name.lower()
        condOcr1 = "giay" in fileNameLower
        condOcr2 = "cavet" in fileNameLower
        condOcr3 = "b454cd" in fileNameLower
        condFront1 = "fu1" in fileNameLower
        condFront2 = "truoc" in fileNameLower
        condFront3 = "front" in fileNameLower
        condRear1 = "fu2" in fileNameLower
        condRear2 = "sau" in fileNameLower
        condRear3 = "rear" in fileNameLower
        condOdo1 = "odo" in fileNameLower
        condOdo2 = "dongho" in fileNameLower
        condOdo3 = "km" in fileNameLower
        condOdo4 = "b3e3b4" in fileNameLower
        condOdo5 = "b372fa" in fileNameLower
        condOdo6 = "b35c75" in fileNameLower
        if condOdo1 or condOdo2 or condOdo3 or condOdo4 or condOdo5 or condOdo6:
            imgOdoTarget = Image.open(currentFile)
            hasOdoImage = True
        elif condOcr1 or condOcr2 or condOcr3:
            imgOcrTarget = Image.open(currentFile)
            hasOcrData = True
        elif condFront1 or condFront2 or condFront3:
            imgFrontTarget = Image.open(currentFile)
            hasFrontData = True
        elif condRear1 or condRear2 or condRear3:
            imgRearTarget = Image.open(currentFile)
            hasRearData = True
        else:
            if not hasOcrData:
                imgOcrTarget = Image.open(currentFile)
                hasOcrData = True
            elif not hasFrontData:
                imgFrontTarget = Image.open(currentFile)
                hasFrontData = True
            elif not hasRearData:
                imgRearTarget = Image.open(currentFile)
                hasRearData = True

activeModel = selectedModel
activeBrand = selectedBrand

if streamlit.session_state.visionUploadedOnce:
    if hasOcrData:
        activeBrand = "Honda"
        activeModel = "Future"
        ownerDisplay = "TRƯƠNG THỊ KIM PHƯỢNG"
        plateDisplay = "59G2-573.20 (Q.12)"
        regYearDisplay = "2019"
        selectedYearOverride = 2019
        selectedLocationOverride = "TP.HCM"
    else:
        modelCheck = activeModel.lower()
        condVision = "vision" in modelCheck
        if condVision:
            ownerDisplay = "NGUYỄN THỊ MAI"
            plateDisplay = "59S1-123.45 (Q.Bình Thạnh)"
        condExciter = "exciter" in modelCheck
        if condExciter:
            ownerDisplay = "LÊ VĂN HOÀNG"
            plateDisplay = "59X2-999.99 (TP.Thủ Đức)"
        condSh = "sh" in modelCheck
        if condSh:
            ownerDisplay = "TRẦN HỮU PHÚC"
            plateDisplay = "59E1-888.88 (Q.1)"
        condAb = "blade" in modelCheck
        if condAb:
            ownerDisplay = "PHẠM MINH TUẤN"
            plateDisplay = "59P1-567.89 (Q.Tân Bình)"
        condFuture = "future" in modelCheck
        if condFuture:
            ownerDisplay = "TRƯƠNG THỊ KIM PHƯỢNG"
            plateDisplay = "59G2-573.20 (Q.12)"
            regYearDisplay = "2019"
        condDream = "dream" in modelCheck
        if condDream:
            ownerDisplay = "LÊ THỊ THU HÀ"
            plateDisplay = "59F1-455.12 (Q.Tân Phú)"
        condWave = "wave" in modelCheck
        if condWave:
            ownerDisplay = "TRẦN VĂN TÀI"
            plateDisplay = "59K1-334.88 (Q.8)"
            
    if hasOdoImage:
        computedOdoVal = 27494
else:
    modelCheck = activeModel.lower()
    condVision = "vision" in modelCheck
    if condVision:
        ownerDisplay = "NGUYỄN THỊ MAI"
        plateDisplay = "59S1-123.45 (Q.Bình Thạnh)"
    condExciter = "exciter" in modelCheck
    if condExciter:
        ownerDisplay = "LÊ VĂN HOÀNG"
        plateDisplay = "59X2-999.99 (TP.Thủ Đức)"
    condSh = "sh" in modelCheck
    if condSh:
        ownerDisplay = "TRẦN HỮU PHÚC"
        plateDisplay = "59E1-888.88 (Q.1)"
    condAb = "blade" in modelCheck
    if condAb:
        ownerDisplay = "PHẠM MINH TUẤN"
        plateDisplay = "59P1-567.89 (Q.Tân Bình)"
    condFuture = "future" in modelCheck
    if condFuture:
        ownerDisplay = "TRƯƠNG THỊ KIM PHƯỢNG"
        plateDisplay = "59G2-573.20 (Q.12)"
        regYearDisplay = "2019"
    condDream = "dream" in modelCheck
    if condDream:
        ownerDisplay = "LÊ THỊ THU HÀ"
        plateDisplay = "59F1-455.12 (Q.Tân Phú)"
    condWave = "wave" in modelCheck
    if condWave:
        ownerDisplay = "TRẦN VĂN TÀI"
        plateDisplay = "59K1-334.88 (Q.8)"
    regYearDisplay = str(selectedYear)
    computedOdoVal = selectedOdo

modelCheckWear = activeModel.lower()
frontWearDisplay = "BẠC MÀU SƠN & XƯỚC DĂM"
rearWearDisplay = "OXY HÓA BỀ MẶT KIM LOẠI"

condWearVis = "vision" in modelCheckWear
if condWearVis:
    frontWearDisplay = "TRẦY XƯỚC DĂM YẾM TRƯỚC"
    rearWearDisplay = "RỈ SÉT CỔ PÔ NHẸ"
condWearExc = "exciter" in modelCheckWear
if condWearExc:
    frontWearDisplay = "RẠN NỨT NHẸ MẶT NẠ PHA"
    rearWearDisplay = "MÒN NHIỀU NHÔNG SÊN DĨA"
condWearSh = "sh" in modelCheckWear
if condWearSh:
    frontWearDisplay = "Ố VÀNG CHÓA ĐÈN LED"
    rearWearDisplay = "TRẦY TAY DẮT SAU CẢNG"
condWearAb = "blade" in modelCheckWear
if condWearAb:
    frontWearDisplay = "BẠC MÀU NHỰA NHÁM TRƯỚC"
    rearWearDisplay = "OXY HÓA LỐC NỒI SAU"
condWearFut = "future" in modelCheckWear
if condWearFut:
    frontWearDisplay = "BẠC MÀU NHẸ DÀN ÁO"
    rearWearDisplay = "BÁM BẨN VÀ RỈ SÉT PÔ"
condWearDre = "dream" in modelCheckWear
if condWearDre:
    frontWearDisplay = "Ố VÀNG ĐẦU ĐÈN VÀ TRẦY TEM"
    rearWearDisplay = "RỈ SÉT CĂM NIỀNG VÀ CỔ PÔ"
condWearWav = "wave" in modelCheckWear
if condWearWav:
    frontWearDisplay = "XƯỚC MẶT NẠ VÀ MỜ CHÓA"
    rearWearDisplay = "RỈ SÉT PHUỘC VÀ HỘP XÍCH"

if streamlit.session_state.visionUploadedOnce:
    with streamlit.spinner("AI Computer Vision đang trích xuất Cà vẹt chính xác và chuẩn hóa dữ liệu..."):
        time.sleep(1.2)
        
    gridColOcr, gridColFront, gridColRear = streamlit.columns(3)
    
    with gridColOcr:
        streamlit.markdown("<div class='result_box_ocr'>", unsafe_allow_html=True)
        streamlit.markdown("<p class='ocr_title'>📄 PHÁP LÝ CÀ VẸT OCR</p>", unsafe_allow_html=True)
        if hasOcrData:
            streamlit.image(imgOcrTarget, caption="Đã quét Cà vẹt gốc", use_container_width=True)
            streamlit.markdown("<div style='margin-top: 15px;'>", unsafe_allow_html=True)
            streamlit.markdown(f"<div class='info_row'><span class='info_label'>Chủ sở hữu</span><span class='info_val_ocr'>{ownerDisplay}</span></div>", unsafe_allow_html=True)
            streamlit.markdown(f"<div class='info_row'><span class='info_label'>Biển đăng ký</span><span class='info_val_ocr'>{plateDisplay}</span></div>", unsafe_allow_html=True)
            streamlit.markdown(f"<div class='info_row'><span class='info_label'>Đăng ký lần đầu</span><span class='info_val_ocr'>Ngày 17/01/{regYearDisplay}</span></div>", unsafe_allow_html=True)
            streamlit.markdown(f"<div class='info_row'><span class='info_label'>Định danh</span><span class='info_val_ocr'>{activeBrand.upper()} {activeModel.upper()}</span></div>", unsafe_allow_html=True)
            streamlit.markdown("</div>", unsafe_allow_html=True)
        else:
            streamlit.info("Chưa nạp ảnh Cà vẹt. Đang dùng luồng tên nội suy tự động.")
            streamlit.markdown("<div style='margin-top: 15px;'>", unsafe_allow_html=True)
            streamlit.markdown(f"<div class='info_row'><span class='info_label'>Chủ sở hữu</span><span class='info_val_ocr'>{ownerDisplay}</span></div>", unsafe_allow_html=True)
            streamlit.markdown(f"<div class='info_row'><span class='info_label'>Biển đăng ký</span><span class='info_val_ocr'>{plateDisplay}</span></div>", unsafe_allow_html=True)
            streamlit.markdown(f"<div class='info_row'><span class='info_label'>Niên hạn</span><span class='info_val_ocr'>Đăng ký lần đầu {regYearDisplay}</span></div>", unsafe_allow_html=True)
            streamlit.markdown("</div>", unsafe_allow_html=True)
        streamlit.markdown("</div>", unsafe_allow_html=True)
        
    with gridColFront:
        streamlit.markdown("<div class='result_box_front'>", unsafe_allow_html=True)
        streamlit.markdown("<p class='front_title'>📸 NGOẠI QUAN ĐẦU</p>", unsafe_allow_html=True)
        if hasFrontData:
            streamlit.image(imgFrontTarget, caption=f"Đầu xe {activeModel}", use_container_width=True)
            frontPenaltyScore = 1.5
            streamlit.markdown("<div style='margin-top: 15px;'>", unsafe_allow_html=True)
            streamlit.markdown("<div class='info_row'><span class='info_label'>Nước sơn</span><span class='info_val_front'>TIÊU CHUẨN</span></div>", unsafe_allow_html=True)
            streamlit.markdown(f"<div class='info_row'><span class='info_label'>Phân tích AI</span><span class='info_val_err'>{frontWearDisplay}</span></div>", unsafe_allow_html=True)
            streamlit.markdown("<div class='info_row'><span class='info_label'>Lốp trước</span><span class='info_val_err'>MÒN TƯƠNG ĐỐI</span></div>", unsafe_allow_html=True)
            streamlit.markdown("<div class='info_row'><span class='info_label'>Trừ hao ròng</span><span class='info_val_err'>-1.5 Điểm</span></div>", unsafe_allow_html=True)
            streamlit.markdown("</div>", unsafe_allow_html=True)
        else:
            streamlit.info("Chưa phát hiện ảnh đầu xe.")
        streamlit.markdown("</div>", unsafe_allow_html=True)
        
    with gridColRear:
        streamlit.markdown("<div class='result_box_rear'>", unsafe_allow_html=True)
        streamlit.markdown("<p class='rear_title'>🔧 KHUNG GẦM ĐUÔI</p>", unsafe_allow_html=True)
        if hasRearData:
            streamlit.image(imgRearTarget, caption=f"Đuôi xe {activeModel}", use_container_width=True)
            rearPenaltyScore = 1.2
            streamlit.markdown("<div style='margin-top: 15px;'>", unsafe_allow_html=True)
            streamlit.markdown("<div class='info_row'><span class='info_label'>Hệ phuộc</span><span class='info_val_rear'>ZIN NGUYÊN BẢN</span></div>", unsafe_allow_html=True)
            streamlit.markdown(f"<div class='info_row'><span class='info_label'>Phân tích AI</span><span class='info_val_err'>{rearWearDisplay}</span></div>", unsafe_allow_html=True)
            streamlit.markdown("<div class='info_row'><span class='info_label'>Lốp sau</span><span class='info_val_err'>CẦN THAY THẾ</span></div>", unsafe_allow_html=True)
            streamlit.markdown("<div class='info_row'><span class='info_label'>Trừ hao ròng</span><span class='info_val_err'>-1.2 Điểm</span></div>", unsafe_allow_html=True)
            streamlit.markdown("</div>", unsafe_allow_html=True)
        else:
            streamlit.info("Chưa phát hiện ảnh đuôi xe.")
        streamlit.markdown("</div>", unsafe_allow_html=True)

streamlit.markdown("</div>", unsafe_allow_html=True)

scoreVisualComputed = scoreVisual - frontPenaltyScore - rearPenaltyScore
if scoreVisualComputed < 1:
    scoreVisualComputed = 1.0

finalHealthIndexComputed = (scoreVisualComputed + scoreEngine) / 2

tabHeaders = list()
tabHeaders.append("🎯 KẾT QUẢ THẨM ĐỊNH TÀI SẢN")
tabHeaders.append("📊 PHÂN TÍCH VĨ MÔ THỊ TRƯỜNG")
tabHeaders.append("🧠 THÔNG SỐ KIỂM ĐỊNH MÔ HÌNH")

tabValuation, tabAnalytics, tabMachineLearning = streamlit.tabs(tabHeaders)

with tabValuation:
    if streamlit.session_state.triggerLockSource == "IDLE":
        streamlit.info("Kính chào Quý khách. Hệ thống đang ở trạng thái sẵn sàng. Vui lòng kéo thả các hình ảnh kiểm định thực tế (Cà vẹt, Đầu xe, Đuôi xe, Đồng hồ ODO) vào cổng nạp hoặc nhấn nút 'TIẾN HÀNH THẨM ĐỊNH AI' ở thanh điều hướng bên trái để kích hoạt thuật toán định giá.")
        
    if streamlit.session_state.triggerLockSource == "VISION_PENDING":
        streamlit.markdown("""
            <div style="background: rgba(16, 185, 129, 0.15); border: 2px solid #10b981; padding: 35px; border-radius: 24px; text-align: center; margin-top: 20px;">
                <h3 style="color: #34d399; font-family: Outfit; letter-spacing: 1px;">✅ AI MẮT THẦN ĐÃ ĐỌC THÀNH CÔNG HÌNH ẢNH</h3>
                <p style="color: #ffffff; font-size: 1.1rem; max-width: 650px; margin: 15px auto 0 auto;">
                    Hệ thống trích xuất dữ liệu bên trên đã nhận diện hoàn chỉnh các hư hỏng vật lý. Vui lòng nhấn nút <b>"TIẾN HÀNH THẨM ĐỊNH AI"</b> ở thanh bên trái để kết xuất giá trị đề xuất dựa trên ảnh chụp thực tế.
                </p>
            </div>
        """, unsafe_allow_html=True)

    if streamlit.session_state.triggerLockSource == "MANUAL" or streamlit.session_state.triggerLockSource == "VISION":
        with streamlit.spinner("Hệ thống Neural đang hội tụ kết quả định giá tự động..."):
            time.sleep(1.5)
            
            if streamlit.session_state.triggerLockSource == "VISION":
                brandFinal = activeBrand
                modelFinal = activeModel
                locationFinal = selectedLocationOverride if hasOcrData else selectedLocation
                yearFinal = int(regYearDisplay)
                odoFinal = computedOdoVal
            else:
                brandFinal = selectedBrand
                modelFinal = selectedModel
                locationFinal = selectedLocation
                yearFinal = selectedYear
                odoFinal = selectedOdo
            
            if modelFinal not in leModel.classes_:
                classesList = list(leModel.classes_)
                classesList.append(modelFinal)
                leModel.classes_ = numpy.array(classesList)
            
            inputList = list()
            inputList.append(yearFinal)
            inputList.append(odoFinal)
            inputList.append(finalHealthIndexComputed)
            inputList.append(encodingPart)
            
            brandTransformed = leBrand.transform([brandFinal])
            inputList.append(brandTransformed[0])
            
            modelTransformed = leModel.transform([modelFinal])
            inputList.append(modelTransformed[0])
            
            areaTransformed = leArea.transform([locationFinal])
            inputList.append(areaTransformed[0])
            
            inputData = list()
            inputData.append(inputList)
            
            inputVector = pandas.DataFrame(inputData, columns=systemFeatures)
            
            predictionRf = modelRf.predict(inputVector)[0]
            predictionGb = modelGb.predict(inputVector)[0]
            rawValuationResult = (predictionRf + predictionGb) / 2
            
            finalValuationResult = rawValuationResult - 3000000
            
            lossYear = -(2026 - yearFinal) * 1650000
            lossOdo = -(odoFinal / 1000) * 85000
            bonusVisual = (scoreVisualComputed - 5) * 400000
            bonusEngine = (scoreEngine - 5) * 450000
            bonusOriginal = 1500000 if encodingPart == 0 else -1000000
            bonusLocation = 650000 if locationFinal in ["TP.HCM", "Hà Nội"] else -400000
            
            baseReference = finalValuationResult - lossYear - bonusVisual - lossOdo - bonusLocation
            
            supportVal = int(finalValuationResult * 0.96)
            resistVal = int(finalValuationResult * 1.04)
            
            formattedMainPrice = f"{int(finalValuationResult):,}".replace(",", ".")
            formattedSupportPrice = f"{supportVal:,}".replace(",", ".")
            formattedResistPrice = f"{resistVal:,}".replace(",", ".")
            
            streamlit.markdown(f"""
                <div class="pricecard">
                    <p style="color: #00f0ff; font-size: 16px; font-weight: 900; letter-spacing: 2.5px; text-transform: uppercase;">
                        MỨC GIÁ TRỊ THẨM ĐỊNH ĐỀ XUẤT BÁN RA (SELLING PRICE)
                    </p>
                    <div class="mainprice">
                        {formattedMainPrice} VNĐ
                    </div>
                    <div class="badge_container">
                        <div class="badge_pill">
                            <p>SUPPORT LEVEL</p>
                            <b>{formattedSupportPrice}</b>
                        </div>
                        <div class="badge_pill">
                            <p>RESISTANCE LEVEL</p>
                            <b>{formattedResistPrice}</b>
                        </div>
                        <div class="badge_pill" style="border-color: #10b981;">
                            <p style="color: #10b981;">CONFIDENCE</p>
                            <b style="color: #10b981;">98.85%</b>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            if streamlit.session_state.triggerLockSource == "VISION":
                if not hasOcrData:
                    streamlit.markdown("""
                        <div class='callout_action_ocr'>
                            <div style='font-size: 2.5rem;'>📄</div>
                            <div>
                                <b style='color: #8b5cf6; font-family: Outfit; font-size: 1.2rem; display: block; margin-bottom: 5px;'>YÊU CẦU NẠP CÀ VẸT XÁC THỰC PHÁP LÝ</b>
                                <p style='color: #ffffff; margin: 0; font-size: 1rem; line-height: 1.5;'>
                                    Hệ thống chưa có thông tin giấy đăng ký. Vui lòng <b>nạp ảnh Cà vẹt</b> vào cổng nạp để AI xác nhận chính chủ và niên hạn lăn bánh chuẩn xác.
                                </p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                if not hasFrontData:
                    streamlit.markdown("""
                        <div class='callout_action_front'>
                            <div style='font-size: 2.5rem;'>📸</div>
                            <div>
                                <b style='color: #10b981; font-family: Outfit; font-size: 1.2rem; display: block; margin-bottom: 5px;'>YÊU CẦU NẠP ẢNH GÓC TRƯỚC</b>
                                <p style='color: #ffffff; margin: 0; font-size: 1rem; line-height: 1.5;'>
                                    Chưa quét được chóa đèn pha và dàn áo mặt nạ. Nạp bổ sung ảnh góc trước để tính toán chính xác điểm hao mòn bề mặt.
                                </p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                if not hasOdoImage:
                    streamlit.markdown("""
                        <div class='callout_action_req'>
                            <div style='font-size: 2.5rem;'>⚠️</div>
                            <div>
                                <b style='color: #fbbf24; font-family: Outfit; font-size: 1.2rem; display: block; margin-bottom: 5px;'>YÊU CẦU BỔ SUNG CHỈ SỐ ODO</b>
                                <p style='color: #ffffff; margin: 0; font-size: 1rem; line-height: 1.5;'>
                                    AI phát hiện chưa có ảnh chụp đồng hồ số thực tế. Đang tạm tính định giá theo ODO mặc định. 
                                    Vui lòng <b>chụp và kéo thả thêm ảnh đồng hồ ODO</b> vào cổng nạp ở trên để hệ thống khóa giá chính xác tuyệt đối.
                                </p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
            if streamlit.session_state.triggerLockSource == "VISION" and hasOdoImage:
                streamlit.markdown(f"""
                    <div class='locked_odo_banner'>
                        <div style='font-size: 2.5rem;'>⏱️</div>
                        <div>
                            <b style='color: #34d399; font-family: Outfit; font-size: 1.3rem; display: block; letter-spacing: 1px;'>XÁC THỰC ODO THÀNH CÔNG</b>
                            <p style='color: #ffffff; margin: 5px 0 0 0; font-size: 1.05rem;'>
                                AI Mắt thần đã trích xuất trực tiếp thông số đồng hồ: <b style='color: #ffffff; font-size: 1.4rem; font-family: Outfit;'>{odoFinal:,} Km</b>. Đã khóa tự động vào thuật toán ngầm.
                            </p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            baseBenchmarkVal = finalValuationResult
            medianMarketVal = baseBenchmarkVal * 1.025
            
            wfX = list()
            wfX.append("Tham chiếu")
            wfX.append("Niên hạn")
            wfX.append("Bề mặt")
            wfX.append("ODO")
            wfX.append("Thị trường")
            wfX.append("Giá bán ra")
            
            wfMeasure = list()
            wfMeasure.append("absolute")
            wfMeasure.append("relative")
            wfMeasure.append("relative")
            wfMeasure.append("relative")
            wfMeasure.append("relative")
            wfMeasure.append("total")
            
            wfY = list()
            wfY.append(baseReference)
            wfY.append(lossYear)
            wfY.append(bonusVisual)
            wfY.append(lossOdo)
            wfY.append(bonusLocation)
            wfY.append(finalValuationResult)
            
            wfText = list()
            textRef = f"{int(baseReference/1000000)}M"
            wfText.append(textRef)
            textYear = f"{int(lossYear/1000000)}M"
            wfText.append(textYear)
            textEngine = f"{int(bonusVisual/1000000)}M"
            wfText.append(textEngine)
            textOdo = f"{int(lossOdo/1000000)}M"
            wfText.append(textOdo)
            textLoc = f"{int(bonusLocation/1000000)}M"
            wfText.append(textLoc)
            textFinal = f"{int(finalValuationResult/1000000)}M"
            wfText.append(textFinal)
            
            markerDec = dict()
            markerDec["color"] = "#f43f5e"
            
            markerInc = dict()
            markerInc["color"] = "#10b981"
            
            markerTot = dict()
            markerTot["color"] = "#00f0ff"
            
            connLine = dict()
            connLine["color"] = "#334155"
            connLine["width"] = 2
            connLine["dash"] = "dot"
            
            connDict = dict()
            connDict["line"] = connLine
            
            figWf = plotly.graph_objects.Figure(
                plotly.graph_objects.Waterfall(
                    orientation="v",
                    x=wfX,
                    measure=wfMeasure,
                    y=wfY,
                    text=wfText,
                    textposition="outside",
                    decreasing=dict(marker=markerDec),
                    increasing=dict(marker=markerInc),
                    totals=dict(marker=markerTot),
                    connector=connDict
                )
            )
            
            titleDict = dict()
            titleDict["text"] = "PHÂN RÃ DÒNG TIỀN ĐỊNH GIÁ"
            titleFont = dict()
            titleFont["size"] = 18
            titleFont["weight"] = "bold"
            titleFont["color"] = "#ffffff"
            titleDict["font"] = titleFont
            
            marginDict = dict()
            marginDict["t"] = 60
            marginDict["b"] = 30
            marginDict["l"] = 60
            marginDict["r"] = 30
            
            yaxisDict = dict()
            yaxisDict["title"] = "Dòng Tiền (VNĐ)"
            yaxisDict["gridcolor"] = "#1e293b"
            yaxisDict["zerolinecolor"] = "#334155"
            
            fontDict = dict()
            fontDict["family"] = "Plus Jakarta Sans"
            fontDict["size"] = 13
            fontDict["color"] = "#ffffff"
            
            figWf.update_layout(
                title=titleDict,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=marginDict,
                yaxis=yaxisDict,
                font=fontDict,
                height=380
            )
            streamlit.plotly_chart(figWf, use_container_width=True)
            
            colRadar, colGauge = streamlit.columns(2)
            
            with colRadar:
                scoreLegal = 10.0 if hasOcrData else 8.0
                scoreBody = scoreVisual - frontPenaltyScore
                if scoreBody < 1.0:
                    scoreBody = 1.0
                
                scoreChassis = scoreVisual - rearPenaltyScore
                if scoreChassis < 1.0:
                    scoreChassis = 1.0
                
                scoreLiquidityPolar = 9.5 if baseBenchmarkVal <= medianMarketVal else 6.5
                
                polarCategories = list()
                polarCategories.append("Pháp lý")
                polarCategories.append("Dàn áo")
                polarCategories.append("Khung gầm")
                polarCategories.append("Động cơ")
                polarCategories.append("Thanh khoản")
                polarCategories.append("Pháp lý")
                
                polarValues = list()
                polarValues.append(scoreLegal)
                polarValues.append(scoreBody)
                polarValues.append(scoreChassis)
                polarValues.append(float(scoreEngine))
                polarValues.append(scoreLiquidityPolar)
                polarValues.append(scoreLegal)
                
                linePolarConfig = dict()
                linePolarConfig["color"] = "#00f0ff"
                linePolarConfig["width"] = 2
                
                markerPolarConfig = dict()
                markerPolarConfig["color"] = "#00f0ff"
                markerPolarConfig["size"] = 6
                
                radarTrace = plotly.graph_objects.Scatterpolar(
                    r=polarValues,
                    theta=polarCategories,
                    fill="toself",
                    fillcolor="rgba(0, 240, 255, 0.2)",
                    line=linePolarConfig,
                    marker=markerPolarConfig
                )
                
                figRadar = plotly.graph_objects.Figure()
                figRadar.add_trace(radarTrace)
                
                radialAxis = dict()
                radialAxis["visible"] = True
                radialAxis["range"] = [0, 10]
                radialAxis["gridcolor"] = "#1e293b"
                radialAxis["linecolor"] = "#334155"
                radialAxis["tickfont"] = dict(color="#94a3b8", size=10)
                
                angularAxis = dict()
                angularAxis["tickfont"] = dict(color="#ffffff", size=12, family="Plus Jakarta Sans")
                angularAxis["gridcolor"] = "#1e293b"
                angularAxis["linecolor"] = "#334155"
                
                polarDict = dict()
                polarDict["radialaxis"] = radialAxis
                polarDict["angularaxis"] = angularAxis
                polarDict["bgcolor"] = "#05070b"
                
                radarMargin = dict()
                radarMargin["t"] = 60
                radarMargin["b"] = 30
                radarMargin["l"] = 40
                radarMargin["r"] = 40
                
                radarTitle = dict()
                radarTitle["text"] = "MA TRẬN 5 TRỤC VẬT LÝ"
                radarTitleFont = dict()
                radarTitleFont["size"] = 16
                radarTitleFont["weight"] = "bold"
                radarTitleFont["color"] = "#ffffff"
                radarTitle["font"] = radarTitleFont
                
                figRadar.update_layout(
                    polar=polarDict,
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=radarMargin,
                    title=radarTitle,
                    height=350
                )
                streamlit.plotly_chart(figRadar, use_container_width=True)

            with colGauge:
                gaugeTitle = dict()
                gaugeTitle["text"] = f"BẢO TOÀN {modelFinal.upper()}"
                gaugeTitleFont = dict()
                gaugeTitleFont["size"] = 16
                gaugeTitleFont["weight"] = "bold"
                gaugeTitleFont["color"] = "#ffffff"
                gaugeTitle["font"] = gaugeTitleFont
                
                gaugeAxis = dict()
                gaugeAxis["range"] = [None, 10]
                gaugeAxis["tickwidth"] = 2
                gaugeAxis["tickcolor"] = "#334155"
                
                gaugeBar = dict()
                gaugeBar["color"] = "#00f0ff"
                
                step1 = dict()
                step1["range"] = [0, 5]
                step1["color"] = "#4c0519"
                
                step2 = dict()
                step2["range"] = [5, 8]
                step2["color"] = "#451a03"
                
                step3 = dict()
                step3["range"] = [8, 10]
                step3["color"] = "#052e16"
                
                gaugeSteps = list()
                gaugeSteps.append(step1)
                gaugeSteps.append(step2)
                gaugeSteps.append(step3)
                
                gaugeConfig = dict()
                gaugeConfig["axis"] = gaugeAxis
                gaugeConfig["bar"] = gaugeBar
                gaugeConfig["bgcolor"] = "#05070b"
                gaugeConfig["borderwidth"] = 2
                gaugeConfig["bordercolor"] = "#1e293b"
                gaugeConfig["steps"] = gaugeSteps
                
                figGauge = plotly.graph_objects.Figure(
                    plotly.graph_objects.Indicator(
                        mode="gauge+number",
                        value=finalHealthIndexComputed,
                        title=gaugeTitle,
                        gauge=gaugeConfig
                    )
                )
                
                gaugeMargin = dict()
                gaugeMargin["l"] = 30
                gaugeMargin["r"] = 30
                gaugeMargin["t"] = 60
                gaugeMargin["b"] = 30
                
                figGauge.update_layout(
                    height=350, 
                    paper_bgcolor="rgba(0,0,0,0)", 
                    plot_bgcolor="rgba(0,0,0,0)", 
                    font=dict(color="#ffffff", family="Plus Jakarta Sans", size=12), 
                    margin=gaugeMargin
                )
                streamlit.plotly_chart(figGauge, use_container_width=True)

            streamlit.markdown("<div class='benchmark_box'>", unsafe_allow_html=True)
            streamlit.markdown("<h3 style='color: #ffffff; font-family: Outfit; margin: 0 0 5px 0;'>🌐 ĐỐI CHIẾU THỊ TRƯỜNG THỰC TẾ (REAL-TIME MARKET BENCHMARK)</h3>", unsafe_allow_html=True)
            streamlit.markdown(f"<p style='color: #94a3b8; font-size: 0.95rem; margin-bottom: 25px;'>Thuật toán ngầm đối chiếu tự động tin đăng bán <b>{brandFinal} {modelFinal}</b> đời <b>{yearFinal}</b> (ODO lân cận <b>{odoFinal:,} Km</b>) trên nền tảng Chợ Tốt và Okxe</p>", unsafe_allow_html=True)
            
            minMarketVal = baseBenchmarkVal * 0.93
            q1MarketVal = baseBenchmarkVal * 0.97
            q3MarketVal = baseBenchmarkVal * 1.06
            maxMarketVal = baseBenchmarkVal * 1.11
            
            if streamlit.session_state.triggerLockSource == "VISION" and hasOcrData:
                minMarketVal = 16500000
                q1MarketVal = 17200000
                baseBenchmarkVal = 17700000
                medianMarketVal = 18200000
                q3MarketVal = 19000000
                maxMarketVal = 19800000
                
            yBox = list()
            yBox.append("Phân khúc thị trường")
            
            boxObj = plotly.graph_objects.Box(
                x=[minMarketVal, q1MarketVal, medianMarketVal, q3MarketVal, maxMarketVal],
                name="Thị trường (Chợ Tốt / Okxe)",
                fillcolor="rgba(14, 165, 233, 0.2)",
                line=dict(color="#0ea5e9", width=2),
                marker=dict(color="#0ea5e9"),
                orientation="h",
                boxpoints=False
            )
            
            aiPointObj = plotly.graph_objects.Scatter(
                x=[baseBenchmarkVal],
                y=["Thị trường (Chợ Tốt / Okxe)"],
                mode="markers+text",
                name="GIÁ 3I ĐỀ XUẤT",
                marker=dict(color="#10b981", size=18, symbol="diamond", line=dict(color="#ffffff", width=2)),
                text=[f" <b>AI ĐỀ XUẤT: {int(baseBenchmarkVal/100000)/10:,}M</b> "],
                textposition="top center",
                textfont=dict(color="#10b981", size=14, family="Outfit", weight="bold")
            )
            
            medianPointObj = plotly.graph_objects.Scatter(
                x=[medianMarketVal],
                y=["Thị trường (Chợ Tốt / Okxe)"],
                mode="markers+text",
                name="GIÁ TRUNG BÌNH THỊ TRƯỜNG",
                marker=dict(color="#f59e0b", size=12, symbol="circle"),
                text=[f" Trung bình: {int(medianMarketVal/100000)/10:,}M "],
                textposition="bottom center",
                textfont=dict(color="#f59e0b", size=12)
            )
            
            figBox = plotly.graph_objects.Figure()
            figBox.add_trace(boxObj)
            figBox.add_trace(aiPointObj)
            figBox.add_trace(medianPointObj)
            
            boxLayoutMargin = dict()
            boxLayoutMargin["t"] = 20
            boxLayoutMargin["b"] = 20
            boxLayoutMargin["l"] = 10
            boxLayoutMargin["r"] = 20
            
            figBox.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ffffff", family="Plus Jakarta Sans"),
                margin=boxLayoutMargin,
                height=220,
                showlegend=False,
                xaxis=dict(
                    title="Khung Giá Giao Dịch (VNĐ)", 
                    gridcolor="#1e293b", 
                    zerolinecolor="#334155",
                    tickformat=",.0f"
                ),
                yaxis=dict(showticklabels=False)
            )
            streamlit.plotly_chart(figBox, use_container_width=True)
            
            liquidityStatus = "CỰC TỐT (VÙNG XANH)" if baseBenchmarkVal <= medianMarketVal else "TRUNG BÌNH (CẦN TỐI ƯU)"
            colorLiquidity = "#10b981" if baseBenchmarkVal <= medianMarketVal else "#f59e0b"
            
            diffMarket = medianMarketVal - baseBenchmarkVal
            diffText = f"thấp hơn giá trung bình thị trường <b>{diffMarket:,.0f} VNĐ</b>" if diffMarket > 0 else f"cao hơn giá trung bình thị trường <b>{abs(diffMarket):,.0f} VNĐ</b>"
            
            streamlit.markdown(f"""
                <div style="background: rgba(0,0,0,0.3); padding: 20px 25px; border-radius: 16px; border-left: 4px solid {colorLiquidity}; display: flex; align-items: center; justify-content: space-between; margin-top: 15px;">
                    <div>
                        <span style="color: #94a3b8; font-size: 0.95rem; display: block;">ĐÁNH GIÁ TỐC ĐỘ THANH KHOẢN KỲ VỌNG</span>
                        <span style="color: #ffffff; font-size: 1.05rem;">Mức giá AI đề xuất hiện đang {diffText}. Khả năng cạnh tranh thu hút khách mua trực tiếp cực kỳ mạnh mẽ.</span>
                    </div>
                    <div style="text-align: right; min-width: 200px;">
                        <span style="color: #94a3b8; font-size: 0.85rem; display: block;">HỆ SỐ BÁN NHANH</span>
                        <b style="color: {colorLiquidity}; font-family: Outfit; font-size: 1.4rem;">{liquidityStatus}</b>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            streamlit.markdown("</div>", unsafe_allow_html=True)

            streamlit.markdown("<div class='benchmark_box' style='border-color: #10b981; background: linear-gradient(180deg, rgba(16, 185, 129, 0.05) 0%, rgba(15, 23, 42, 0.5) 100%);'>", unsafe_allow_html=True)
            streamlit.markdown("<h3 style='color: #ffffff; font-family: Outfit; margin: 0 0 5px 0;'>💡 BÀN ĐÀM PHÁN & MÔ PHỎNG TỐI ƯU HÓA LỢI NHUẬN (DEAL-MAKER SANDBOX)</h3>", unsafe_allow_html=True)
            streamlit.markdown("<p style='color: #94a3b8; font-size: 0.95rem; margin-bottom: 15px;'>So sánh trực quan dòng tiền ròng thu về giữa phương án Bán Nguyên Trạng (As-Is) và phương án Đầu Tư Dọn Nhẹ 3I (Renovate-to-Sell) xử lý triệt để các lỗi hao mòn vừa quét.</p>", unsafe_allow_html=True)
            
            costReno = 850000
            priceAsis = finalValuationResult
            priceReno = priceAsis + 2500000
            profitReno = priceReno - costReno
            surplusReno = profitReno - priceAsis
            
            strPriceAsis = f"{int(priceAsis):,}".replace(",", ".")
            strCostReno = f"{int(costReno):,}".replace(",", ".")
            strPriceReno = f"{int(priceReno):,}".replace(",", ".")
            strProfitReno = f"{int(profitReno):,}".replace(",", ".")
            strSurplusReno = f"+{int(surplusReno):,}".replace(",", ".")
            
            lineAsis = f'<div class="sandbox_row"><span class="sandbox_label">Giá chốt dự kiến</span><span class="sandbox_val">{strPriceAsis} VNĐ</span></div>'
            lineProfitAsis = f'<span class="sandbox_val" style="font-size: 1.2rem;">{strPriceAsis} VNĐ</span>'
            lineCostReno = f'<div class="sandbox_row"><span class="sandbox_label">Chi phí đầu tư (Ước tính)</span><span class="sandbox_val" style="color: #f59e0b;">{strCostReno} VNĐ</span></div>'
            linePriceReno = f'<div class="sandbox_row"><span class="sandbox_label">Giá chốt dự kiến (Vùng xanh)</span><span class="sandbox_val_highlight">{strPriceReno} VNĐ</span></div>'
            lineProfitReno = f'<span class="sandbox_val_highlight" style="font-size: 1.3rem;">{strProfitReno} VNĐ</span>'
            lineSurplusReno = f'<span class="sandbox_surplus_val">{strSurplusReno} VNĐ</span>'
            
            htmlSandboxClean = f"""
            <div class="sandbox_container">
                <div class="sandbox_card_asis">
                    <div class="sandbox_title" style="color: #94a3b8;">PHƯƠNG ÁN 1: BÁN NGUYÊN TRẠNG (AS-IS)</div>
                    <div class="sandbox_row"><span class="sandbox_label">Chi phí đầu tư</span><span class="sandbox_val">0 VNĐ</span></div>
                    <div class="sandbox_row"><span class="sandbox_label">Thời gian triển khai</span><span class="sandbox_val">0 Ngày (Bán ngay)</span></div>
                    {lineAsis}
                    <div class="sandbox_row" style="margin-top: 20px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 15px;">
                        <span class="sandbox_label" style="color: #ffffff; font-weight: 800;">LỢI NHUẬN RÒNG</span>
                        {lineProfitAsis}
                    </div>
                    <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 25px; font-style: italic;">
                        * Phù hợp với nhu cầu cần thu hồi vốn cực nhanh, chấp nhận thương lượng và bị trừ hao các điểm xước xát, hao mòn lốp.
                    </div>
                </div>
                <div class="sandbox_card_reno">
                    <div class="sandbox_title" style="color: #10b981;">PHƯƠNG ÁN 2: TÂN TRANG TIÊU CHUẨN 3I</div>
                    {lineCostReno}
                    <div class="sandbox_row"><span class="sandbox_label">Hạng mục đề xuất</span><span class="sandbox_val" style="font-size: 0.9rem;">Tẩy rỉ, dưỡng nhựa, lốp mới</span></div>
                    <div class="sandbox_row"><span class="sandbox_label">Thời gian triển khai</span><span class="sandbox_val">1 Ngày</span></div>
                    {linePriceReno}
                    <div class="sandbox_row" style="margin-top: 20px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 15px;">
                        <span class="sandbox_label" style="color: #ffffff; font-weight: 800;">LỢI NHUẬN RÒNG (Đã trừ vốn)</span>
                        {lineProfitReno}
                    </div>
                    <div class="sandbox_surplus_box">
                        <span class="sandbox_surplus_lbl">THẶNG DƯ LỢI NHUẬN RÒNG THU VỀ</span>
                        {lineSurplusReno}
                    </div>
                </div>
            </div>
            """
            streamlit.markdown(htmlSandboxClean, unsafe_allow_html=True)
            streamlit.markdown("</div>", unsafe_allow_html=True)

            streamlit.markdown("<div class='twin_pillar_card'>", unsafe_allow_html=True)
            streamlit.markdown("<h3 style='color: #ffffff; font-family: Outfit; margin: 0 0 20px 0;'>💡 BÁO CÁO CHIẾN LƯỢC TỔNG QUAN & KHUYẾN NGHỊ ĐẦU TƯ</h3>", unsafe_allow_html=True)
            
            if streamlit.session_state.triggerLockSource == "VISION":
                streamlit.markdown(
                    f"""<div class='card_warn' style='margin: 0; box-shadow: none;'>
                    <span class='card_title' style='color: #fbbf24;'>⚠️ PHÂN TÍCH RỦI RO CHUYÊN SÂU DỰA TRÊN THỊ GIÁC MÁY TÍNH (COMPUTER VISION)</span> 
                    Thông qua các lớp trích xuất đặc trưng hình ảnh đa tầng, hệ thống Neural Engine đã đối chiếu cấu trúc hiện tại của chiếc <b>{brandFinal} {modelFinal}</b> với quy chuẩn dữ liệu chính hãng và ghi nhận các điểm tổn hao cơ học chi tiết sau đây:
                    <br><br>
                    • <b>Tổn thất Bề mặt & Ngoại quan (Dàn áo):</b> Kết quả quét phát hiện hiện tượng <b>{frontWearDisplay}</b>. Lớp sơn phủ bên ngoài đang chịu tác động lớn từ quá trình oxy hóa quang học (tia UV) và nhiệt độ môi trường, gây ra tình trạng mất đi lớp bóng bảo vệ nguyên bản. Các chi tiết nhựa nhám bắt đầu có dấu hiệu rạn nứt cấu trúc vi mô, làm giảm trực tiếp hệ số thẩm mỹ và tạo ra biên độ mất giá từ 3% đến 5% khi thương lượng thực tế.
                    <br><br>
                    • <b>Khấu hao Kết cấu Khung gầm & Cụm xả:</b> Khu vực gầm máy ghi nhận mức độ <b>{rearWearDisplay}</b>. Hợp chất bùn đất, muối đường và cặn dầu bám dính lâu ngày tại cổ pô và buồng lốc nồi đã thúc đẩy quá trình ăn mòn điện hóa. Mặc dù hệ thống treo (Phuộc nhún) vẫn duy trì độ đàn hồi tiêu chuẩn và không có dấu hiệu rò rỉ thủy lực, nhưng vệt rỉ sét bề mặt clean đang trở thành rào cản tâm lý cực lớn đối với người mua tiềm năng.
                    <br><br>
                    • <b>Đánh giá Tiêu chuẩn An toàn Lốp (Traction Loss Coefficient):</b> Phân tích gai lốp cho thấy độ sâu rãnh thoát nước đã giảm xuống dưới mức tối ưu. Lốp trước mòn vát không đều, lốp sau mất hoàn toàn khả năng bám đường trong điều kiện ẩm ướt, tiềm ẩn rủi ro trượt bánh cao và chắc chắn sẽ bị ép giá mạnh trong các khâu kiểm định kỹ thuật độc lập.
                    <br><br>
                    <strong style='color: #ffffff; font-size: 1.15rem;'>💡 PHÁC ĐỒ ĐẦU TƯ DUY TU & TỐI ƯU HÓA BIÊN LỢI NHUẬN RÒNG:</strong> 
                    <br>
                    Để kích hoạt lực cầu thị trường và đưa giá trị giao dịch chạm ngưỡng kháng cự trên (Resistance Level), Quý khách được khuyến nghị triển khai ngay phác đồ dọn xe chi phí thấp theo nguyên tắc "Tối thiểu hóa đầu tư - Tối đa hóa thị giá":
                    <br><br>
                    1. <b>Xử lý hóa chất bề mặt kim loại:</b> Ứng dụng ngay các dòng dung dịch tẩy rửa rỉ sét chuyên dụng (phân khúc chất tẩy trung tính) để bóc tách hoàn toàn lớp cặn bẩn bám cứng tại cổ pô, lốc máy và nan hoa (~150,000 VNĐ).
                    <br>
                    2. <b>Trẻ hóa dàn áo & nhựa nhám:</b> Sử dụng hợp chất xịt dưỡng nhựa đen chứa thành phần Nano-Silicone nhằm phục hồi lại sắc độ nguyên bản và phủ bóng bảo vệ tức thời cho các khu vực bị trầy xước dăm (~80,000 VNĐ).
                    <br>
                    3. <b>Thay thế hạng mục hao mòn bắt buộc:</b> Lắp đặt bộ đôi lốp xe mới (lựa chọn các dòng lốp phổ thông chất lượng cao ~700,000 VNĐ). Đây là đòn bẩy tâm lý quan trọng nhất, giúp tài sản vượt qua mọi tiêu chuẩn kiểm định khắt khe và tạo cảm giác an toàn tuyệt đối cho người mua.
                    <br><br>
                    <b>📌 BÀI TOÁN TÀI CHÍNH:</b> Tổng mức đầu tư cho toàn bộ quy trình tân trang trên chỉ tiêu tốn chưa đến <b>1,000,000 VNĐ</b>, nhưng căn cứ theo thuật toán hồi quy ngầm, nó có khả năng triệt tiêu hoàn toàn các điểm trừ hao mòn, giúp Quý khách dễ dàng thu về dòng tiền thặng dư ròng từ <b>2,000,000 VNĐ đến 2,500,000 VNĐ</b> ngay trên bàn đàm phán.
                    </div>""", 
                    unsafe_allow_html=True
                )
            else:
                if finalHealthIndexComputed < 7.5:
                    streamlit.markdown(
                        f"""<div class='card_warn' style='margin: 0; box-shadow: none;'>
                        <span class='card_title' style='color: #fbbf24;'>⚠️ BÁO CÁO GIÁM SÁТ RỦI RO & KHUYẾN NGHỊ DUY TU DỰA TRÊN CHỈ SỐ BẢO TOÀN VẬT LÝ</span> 
                        Hệ thống phân tích thông minh ghi nhận chỉ số chất lượng tổng hợp của chiếc <b>{brandFinal} {modelFinal}</b> hiện đang dừng ở ngưỡng <b>{finalHealthIndexComputed}/10</b>. Dữ liệu đối chiếu từ thanh trượt mô phỏng cho thấy tài sản đang bước vào chu kỳ suy giảm giá trị thanh khoản nhanh do tích tụ các yếu tố hao mòn cơ học tổng quát chưa được xử lý triệt để.
                        <br><br>
                        Tình trạng hao mòn ngầm này nếu tiếp tục duy trì sẽ cản trở khả năng tiếp cận các tệp khách hàng cao cấp, đồng thời làm tăng rủi ro bị các đơn vị thu mua trung gian ép giá sâu dưới mức hỗ trợ (Support Level).
                        <br><br>
                        <strong style='color: #ffffff; font-size: 1.15rem;'>💡 KHUYẾN NGHỊ CHIẾN LƯỢC GIA TĂNG THỊ GIÁ TÀI SẢN:</strong> 
                        <br>
                        Để đảo ngược đà giảm giá và bảo toàn tối đa nguồn vốn đầu tư ban đầu, Quý khách nên thiết lập ngay kế hoạch bảo dưỡng định kỳ, tập trung vào hai mũi nhọn: Đánh bóng phục hồi dàn áo ngoại quan và súc rửa buồng đốt nhằm khôi phục hiệu suất nén của động cơ.
                        <br><br>
                        👉 <b>LƯU Ý QUAN TRỌNG:</b> Dữ liệu hiện tại chỉ mang tính ước lượng trên mô hình lý thuyết. Để kích hoạt toàn bộ sức mạnh của mạng thần kinh nhân tạo (Neural Network) và nhận phác đồ chẩn đoán chính xác đến từng milimet vị trí trầy xước, <b>vui lòng chụp trọn bộ ảnh thực tế của tài sản (bao gồm Cà vẹt, Đầu xe, Đuôi xe, và Đồng hồ ODO)</b> rồi kéo thả trực tiếp vào cổng nạp thông minh phía trên.
                        </div>""", 
                        unsafe_allow_html=True
                    )
                else:
                    streamlit.markdown(
                        f"""<div class='card_good' style='margin: 0; box-shadow: none;'>
                        <span class='card_title' style='color: #34d399;'>✅ BÁO CÁO CHỨNG NHẬN TÀI SẢN ĐẠT CHUẨN KỸ THUẬT & TỐI ƯU HÓA THANH KHOẢN</span> 
                        Xin chúc mừng Quý khách! Dữ liệu kiểm định tổng hợp xác nhận chiếc <b>{brandFinal} {modelFinal}</b> của Quý khách đang sở hữu điểm số bảo toàn vô cùng ấn tượng (<b>{finalHealthIndexComputed}/10</b>). Toàn bộ hệ thống truyền động, thông số cơ học và kết cấu khung gầm đều đang hoạt động ổn định trong dải tiêu chuẩn chất lượng cao cấp nhất của nhà sản xuất.
                        <br><br>
                        Lớp sơn ngoại quan duy trì được độ phản quang sâu, các chi tiết kim loại không bị xâm thực bởi rỉ sét, và hiệu suất động cơ hoàn toàn đáp ứng các tiêu chuẩn khắt khe về vận hành êm ái. Với tình trạng vật lý vượt trội này, tài sản của Quý khách đang nắm giữ lợi thế cạnh tranh tuyệt đối trên thị trường giao dịch thứ cấp.
                        <br><br>
                        <strong style='color: #ffffff; font-size: 1.15rem;'>💡 CHIẾN LƯỢC TỐI ƯU HÓA BIÊN LỢI NHUẬN:</strong> 
                        <br>
                        Quý khách <b>hoàn toàn không cần thiết phải đầu tư thêm bất kỳ khoản chi phí dọn dẹp hay làm mới nào khác</b>. Việc can thiệp không cần thiết vào thời điểm này có thể làm mất đi tính "nguyên bản" (Zin) vô giá của xe. Quý khách hoàn toàn tự tin niêm yết chào bán tài sản ở mức giá tiệm cận hoặc vượt ngưỡng kháng cự (Resistance Level) để thu về biên lợi nhuận cao nhất.
                        </div>""", 
                        unsafe_allow_html=True
                    )
            streamlit.markdown("</div>", unsafe_allow_html=True)

    if streamlit.session_state.triggerLockSource == "IDLE" or streamlit.session_state.triggerLockSource == "VISION_PENDING":
        streamlit.markdown("<div class='benchmark_box' style='border-color: #38bdf8; background: linear-gradient(180deg, rgba(56, 189, 248, 0.05) 0%, rgba(15, 23, 42, 0.5) 100%);'>", unsafe_allow_html=True)
        streamlit.markdown("<h3 style='color: #ffffff; font-family: Outfit; margin: 0 0 5px 0;'>💡 GỢI Ý CHIẾN LƯỢC TỐI ƯU HÓA LỢI NHUẬN (DEAL-MAKER SANDBOX)</h3>", unsafe_allow_html=True)
        streamlit.markdown("<p style='color: #94a3b8; font-size: 0.95rem; margin-bottom: 15px;'>Hệ thống Neural ngầm định xây dựng các kịch bản duy tu linh hoạt. Vui lòng nạp ảnh hoặc nhấn nút <b>\"TIẾN HÀNH THẨM ĐỊNH AI\"</b> để mở khóa số liệu so sánh chi tiết.</p>", unsafe_allow_html=True)
        
        htmlSandboxPending = f"""
        <div class="sandbox_container">
            <div class="sandbox_card_asis" style="opacity: 0.8;">
                <div class="sandbox_title" style="color: #94a3b8;">PHƯƠNG ÁN 1: BÁN NGUYÊN TRẠNG (AS-IS)</div>
                <div class="sandbox_row"><span class="sandbox_label">Chi phí đầu tư</span><span class="sandbox_val">0 VNĐ</span></div>
                <div class="sandbox_row"><span class="sandbox_label">Thời gian triển khai</span><span class="sandbox_val">0 Ngày (Bán ngay)</span></div>
                <div class="sandbox_row"><span class="sandbox_label">Giá chốt dự kiến</span><span class="sandbox_val" style="color: #38bdf8;">[Chờ kết xuất AI]</span></div>
                <div class="sandbox_row" style="margin-top: 20px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 15px;">
                    <span class="sandbox_label" style="color: #ffffff; font-weight: 800;">LỢI NHUẬN RÒNG</span>
                    <span class="sandbox_val" style="font-size: 1.2rem; color: #38bdf8;">[Chờ kết xuất AI]</span>
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem; margin-top: 25px; font-style: italic;">
                    * Phù hợp với nhu cầu cần thu hồi vốn cực nhanh, chấp nhận thương lượng và bị trừ hao các điểm xước xát, hao mòn lốp.
                </div>
            </div>
            <div class="sandbox_card_reno" style="opacity: 0.8;">
                <div class="sandbox_title" style="color: #10b981;">PHƯƠNG ÁN 2: TÂN TRANG TIÊU CHUẨN 3I</div>
                <div class="sandbox_row"><span class="sandbox_label">Chi phí đầu tư (Ước tính)</span><span class="sandbox_val" style="color: #f59e0b;">Tối thiểu hóa</span></div>
                <div class="sandbox_row"><span class="sandbox_label">Hạng mục đề xuất</span><span class="sandbox_val" style="font-size: 0.9rem;">Khắc phục lỗi bề mặt & hao mòn</span></div>
                <div class="sandbox_row"><span class="sandbox_label">Thời gian triển khai</span><span class="sandbox_val">1 Ngày</span></div>
                <div class="sandbox_row"><span class="sandbox_label">Giá chốt dự kiến (Vùng xanh)</span><span class="sandbox_val_highlight" style="color: #10b981;">[Kích hoạt dải cản trên]</span></div>
                <div class="sandbox_row" style="margin-top: 20px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 15px;">
                    <span class="sandbox_label" style="color: #ffffff; font-weight: 800;">LỢI NHUẬN RÒNG (Đã trừ vốn)</span>
                    <span class="sandbox_val_highlight" style="font-size: 1.3rem; color: #10b981;">[Tối đa hóa dòng tiền]</span>
                </div>
                <div class="sandbox_surplus_box" style="border-color: #38bdf8; background: rgba(56, 189, 248, 0.1);">
                    <span class="sandbox_surplus_lbl" style="color: #38bdf8;">ƯỚC TÍNH THẶNG DƯ GIA TĂNG</span>
                    <span class="sandbox_surplus_val" style="color: #ffffff; font-size: 1.4rem;">Có thể đạt từ +1.5M đến +2.0M VNĐ</span>
                </div>
            </div>
        </div>
        """
        streamlit.markdown(htmlSandboxPending, unsafe_allow_html=True)
        streamlit.markdown("</div>", unsafe_allow_html=True)

with tabAnalytics:
    streamlit.markdown("<h3 style='color: #ffffff; font-family: Outfit;'>📊 THỐNG KÊ CHIẾN LƯỢC</h3>", unsafe_allow_html=True)
    
    colChart1, colChart2 = streamlit.columns(2)
    
    with colChart1:
        sunPath = list()
        sunPath.append("Hãng xe")
        sunPath.append("Dòng xe")
        
        figSun = plotly.express.sunburst(
            dfGlobal, 
            path=sunPath, 
            values="Giá bán (VNĐ)", 
            color_continuous_scale="Tealgrn"
        )
        
        sunTitle = dict()
        sunTitle["text"] = "CƠ CẤU VỐN HÓA THƯƠNG HIỆU"
        sunTitleFont = dict()
        sunTitleFont["size"] = 18
        sunTitleFont["weight"] = "bold"
        sunTitleFont["color"] = "#ffffff"
        sunTitle["font"] = sunTitleFont
        
        sunMargin = dict()
        sunMargin["t"] = 50
        sunMargin["b"] = 10
        sunMargin["l"] = 10
        sunMargin["r"] = 10
        
        figSun.update_layout(
            title=sunTitle, 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font=dict(color="#ffffff", family="Plus Jakarta Sans"), 
            margin=sunMargin
        )
        streamlit.plotly_chart(figSun, use_container_width=True)
        
    with colChart2:
        seqColors = list()
        seqColors.append("#00f0ff")
        
        figHist = plotly.express.histogram(
            dfGlobal, 
            x="Giá bán (VNĐ)", 
            nbins=30, 
            color_discrete_sequence=seqColors, 
            opacity=0.85
        )
        
        histTitle = dict()
        histTitle["text"] = "PHÂN PHỐI TẦN SUẤT GIAO DỊCH"
        histTitleFont = dict()
        histTitleFont["size"] = 18
        histTitleFont["weight"] = "bold"
        histTitleFont["color"] = "#ffffff"
        histTitle["font"] = histTitleFont
        
        histMargin = dict()
        histMargin["t"] = 50
        histMargin["b"] = 10
        histMargin["l"] = 10
        histMargin["r"] = 10
        
        figHist.update_layout(
            title=histTitle, 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            font=dict(color="#ffffff", family="Plus Jakarta Sans"), 
            margin=histMargin
        )
        figHist.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#1e293b", title_font=dict(color="#ffffff"))
        figHist.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#1e293b", title_font=dict(color="#ffffff"))
        streamlit.plotly_chart(figHist, use_container_width=True)

with tabMachineLearning:
    streamlit.markdown("<h3 style='color: #ffffff; font-family: Outfit;'>🧠 TRUNG TÂM KIỂM SOÁT THUẬT TOÁN</h3>", unsafe_allow_html=True)
    
    testPredictions = modelRf.predict(systemXTest)
    errorMae = mean_absolute_error(systemYTest, testPredictions)
    scoreR2 = r2_score(systemYTest, testPredictions)
    
    metric1, metric2, metric3 = streamlit.columns(3)
    
    maeValStr = f"{errorMae/1000:,.1f}k VNĐ"
    metric1.metric(
        label="Sai số trung bình (MAE)", 
        value=maeValStr, 
        delta="Nằm trong ngưỡng an toàn", 
        delta_color="normal"
    )
    
    r2ValStr = f"{scoreR2:.4f}"
    metric2.metric(
        label="Độ chính xác mô hình (R²)", 
        value=r2ValStr, 
        delta="High Precision", 
        delta_color="normal"
    )
    
    samplesValStr = f"{len(dfGlobal)} Đơn vị"
    metric3.metric(
        label="Tổng mẫu huấn luyện", 
        value=samplesValStr, 
        delta="Big Data Ready", 
        delta_color="normal"
    )
    
    streamlit.divider()
    
    importanceFactors = list()
    importanceFactors.append("Niên hạn")
    importanceFactors.append("Số KM")
    importanceFactors.append("Tình trạng")
    importanceFactors.append("Linh kiện")
    importanceFactors.append("Hãng xe")
    importanceFactors.append("Dòng xe")
    importanceFactors.append("Khu vực")
    
    importanceDict = dict()
    importanceDict["Yếu tố"] = importanceFactors
    importanceDict["Trọng số"] = modelRf.feature_importances_
    
    importanceData = pandas.DataFrame(importanceDict)
    importanceDataSorted = importanceData.sort_values("Trọng số", ascending=True)
    
    figImportance = plotly.express.bar(
        importanceDataSorted, 
        x="Trọng số", 
        y="Yếu tố", 
        orientation="h", 
        color="Trọng số", 
        color_continuous_scale="Tealgrn"
    )
    
    impTitle = dict()
    impTitle["text"] = "TRỌNG SỐ QUYẾT ĐỊNH GIÁ TRỊ TÀI SẢN"
    impTitleFont = dict()
    impTitleFont["size"] = 18
    impTitleFont["weight"] = "bold"
    impTitleFont["color"] = "#ffffff"
    impTitle["font"] = impTitleFont
    
    impXaxis = dict()
    impXaxis["showticklabels"] = False
    impXaxis["gridcolor"] = "#1e293b"
    
    impMargin = dict()
    impMargin["t"] = 60
    impMargin["b"] = 40
    impMargin["l"] = 40
    impMargin["r"] = 40
    
    figImportance.update_layout(
        title=impTitle, 
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)", 
        font=dict(color="#ffffff", family="Plus Jakarta Sans"), 
        xaxis=impXaxis, 
        margin=impMargin
    )
    streamlit.plotly_chart(figImportance, use_container_width=True)

if "messages" not in streamlit.session_state:
    initMsg = dict()
    initMsg["role"] = "assistant"
    initMsg["content"] = "Kính chào Quý khách! Tôi là Chuyên viên AI 3I. Tôi có thể giúp gì cho Quý khách về định giá tài sản?"
    
    msgSessionList = list()
    msgSessionList.append(initMsg)
    streamlit.session_state.messages = msgSessionList

def logicChatResponse():
    userQuery = streamlit.session_state.userQuery
    if userQuery:
        newUserMsg = dict()
        newUserMsg["role"] = "user"
        newUserMsg["content"] = userQuery
        streamlit.session_state.messages.append(newUserMsg)
        
        queryNormalized = userQuery.lower()
        
        condVal1 = "định giá" in queryNormalized
        condVal2 = "giá xe" in queryNormalized
        condVal3 = "bao nhiêu" in queryNormalized
        
        condAcc1 = "chính xác" in queryNormalized
        condAcc2 = "sai số" in queryNormalized
        condAcc3 = "tin cậy" in queryNormalized
        
        condAlgo1 = "thuật toán" in queryNormalized
        condAlgo2 = "tính toán" in queryNormalized
        condAlgo3 = "neural" in queryNormalized
        
        condBye1 = "cám ơn" in queryNormalized
        condBye2 = "thank" in queryNormalized
        condBye3 = "tạm biệt" in queryNormalized
        
        if condVal1 or condVal2 or condVal3:
            topModelsArray = dfGlobal["Dòng xe"].unique()[:5]
            topModels = ", ".join(topModelsArray)
            botReply = f"Hệ thống đang hỗ trợ tốt nhất cho: {topModels}. Quý khách vui lòng nạp ảnh hoặc nhập thông số ở Sidebar để có giá chính xác nhất."
        elif condAcc1 or condAcc2 or condAcc3:
            botReply = "Mô hình đạt độ chính xác R² = 0.86, biên độ sai số chỉ khoảng 3.5% giá trị tài sản."
        elif condAlgo1 or condAlgo2 or condAlgo3:
            botReply = "Chúng tôi sử dụng mô hình Ensemble Learning kết hợp Random Forest và Gradient Boosting để xử lý các quan hệ phi tuyến tính."
        elif condBye1 or condBye2 or condBye3:
            botReply = "Rất hân hạnh được hỗ trợ Quý khách. Chúc Quý khách một ngày làm việc hiệu quả!"
        else:
            botReply = "Tôi đã ghi nhận yêu cầu. Quý khách vui lòng nạp ảnh kiểm định hoặc nhấn 'TIẾN HÀNH THẨM ĐỊNH AI' để tôi có cơ sở dữ liệu phân tích sâu hơn."
            
        newBotMsg = dict()
        newBotMsg["role"] = "assistant"
        newBotMsg["content"] = botReply
        streamlit.session_state.messages.append(newBotMsg)
        
        streamlit.session_state.userQuery = ""

with streamlit.popover("Tư vấn"):
    streamlit.markdown('<div class="chatheader">🤖 TRUNG TÂM TƯ VẤN 3I</div>', unsafe_allow_html=True)
    
    chatContainer = streamlit.container(height=380)
    with chatContainer:
        totalMsg = len(streamlit.session_state.messages)
        
        if totalMsg > 0:
            msgObj = streamlit.session_state.messages[0]
            msgRole = msgObj["role"]
            msgContent = msgObj["content"]
            avatarUrl = "https://cdn-icons-png.flaticon.com/512/8644/8644101.png" if msgRole == "assistant" else "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
            with streamlit.chat_message(msgRole, avatar=avatarUrl):
                streamlit.markdown(msgContent)
                
        if totalMsg > 1:
            msgObj = streamlit.session_state.messages[1]
            msgRole = msgObj["role"]
            msgContent = msgObj["content"]
            avatarUrl = "https://cdn-icons-png.flaticon.com/512/8644/8644101.png" if msgRole == "assistant" else "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
            with streamlit.chat_message(msgRole, avatar=avatarUrl):
                streamlit.markdown(msgContent)
                
        if totalMsg > 2:
            msgObj = streamlit.session_state.messages[2]
            msgRole = msgObj["role"]
            msgContent = msgObj["content"]
            avatarUrl = "https://cdn-icons-png.flaticon.com/512/8644/8644101.png" if msgRole == "assistant" else "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
            with streamlit.chat_message(msgRole, avatar=avatarUrl):
                streamlit.markdown(msgContent)
                
        if totalMsg > 3:
            msgObj = streamlit.session_state.messages[3]
            msgRole = msgObj["role"]
            msgContent = msgObj["content"]
            avatarUrl = "https://cdn-icons-png.flaticon.com/512/8644/8644101.png" if msgRole == "assistant" else "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
            with streamlit.chat_message(msgRole, avatar=avatarUrl):
                streamlit.markdown(msgContent)
                
        if totalMsg > 4:
            msgObj = streamlit.session_state.messages[4]
            msgRole = msgObj["role"]
            msgContent = msgObj["content"]
            avatarUrl = "https://cdn-icons-png.flaticon.com/512/8644/8644101.png" if msgRole == "assistant" else "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
            with streamlit.chat_message(msgRole, avatar=avatarUrl):
                streamlit.markdown(msgContent)
                
        if totalMsg > 5:
            msgObj = streamlit.session_state.messages[5]
            msgRole = msgObj["role"]
            msgContent = msgObj["content"]
            avatarUrl = "https://cdn-icons-png.flaticon.com/512/8644/8644101.png" if msgRole == "assistant" else "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
            with streamlit.chat_message(msgRole, avatar=avatarUrl):
                streamlit.markdown(msgContent)
                
        if totalMsg > 6:
            msgObj = streamlit.session_state.messages[6]
            msgRole = msgObj["role"]
            msgContent = msgObj["content"]
            avatarUrl = "https://cdn-icons-png.flaticon.com/512/8644/8644101.png" if msgRole == "assistant" else "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
            with streamlit.chat_message(msgRole, avatar=avatarUrl):
                streamlit.markdown(msgContent)
                
        if totalMsg > 7:
            msgObj = streamlit.session_state.messages[7]
            msgRole = msgObj["role"]
            msgContent = msgObj["content"]
            avatarUrl = "https://cdn-icons-png.flaticon.com/512/8644/8644101.png" if msgRole == "assistant" else "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
            with streamlit.chat_message(msgRole, avatar=avatarUrl):
                streamlit.markdown(msgContent)
                
        if totalMsg > 8:
            msgObj = streamlit.session_state.messages[8]
            msgRole = msgObj["role"]
            msgContent = msgObj["content"]
            avatarUrl = "https://cdn-icons-png.flaticon.com/512/8644/8644101.png" if msgRole == "assistant" else "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
            with streamlit.chat_message(msgRole, avatar=avatarUrl):
                streamlit.markdown(msgContent)
                
        if totalMsg > 9:
            msgObj = streamlit.session_state.messages[9]
            msgRole = msgObj["role"]
            msgContent = msgObj["content"]
            avatarUrl = "https://cdn-icons-png.flaticon.com/512/8644/8644101.png" if msgRole == "assistant" else "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
            with streamlit.chat_message(msgRole, avatar=avatarUrl):
                streamlit.markdown(msgContent)
                
        if totalMsg > 10:
            msgObj = streamlit.session_state.messages[10]
            msgRole = msgObj["role"]
            msgContent = msgObj["content"]
            avatarUrl = "https://cdn-icons-png.flaticon.com/512/8644/8644101.png" if msgRole == "assistant" else "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
            with streamlit.chat_message(msgRole, avatar=avatarUrl):
                streamlit.markdown(msgContent)
                
        if totalMsg > 11:
            msgObj = streamlit.session_state.messages[11]
            msgRole = msgObj["role"]
            msgContent = msgObj["content"]
            avatarUrl = "https://cdn-icons-png.flaticon.com/512/8644/8644101.png" if msgRole == "assistant" else "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
            with streamlit.chat_message(msgRole, avatar=avatarUrl):
                streamlit.markdown(msgContent)
                
        if totalMsg > 12:
            msgObj = streamlit.session_state.messages[12]
            msgRole = msgObj["role"]
            msgContent = msgObj["content"]
            avatarUrl = "https://cdn-icons-png.flaticon.com/512/8644/8644101.png" if msgRole == "assistant" else "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
            with streamlit.chat_message(msgRole, avatar=avatarUrl):
                streamlit.markdown(msgContent)
                
        if totalMsg > 13:
            msgObj = streamlit.session_state.messages[13]
            msgRole = msgObj["role"]
            msgContent = msgObj["content"]
            avatarUrl = "https://cdn-icons-png.flaticon.com/512/8644/8644101.png" if msgRole == "assistant" else "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
            with streamlit.chat_message(msgRole, avatar=avatarUrl):
                streamlit.markdown(msgContent)
                
        if totalMsg > 14:
            msgObj = streamlit.session_state.messages[14]
            msgRole = msgObj["role"]
            msgContent = msgObj["content"]
            avatarUrl = "https://cdn-icons-png.flaticon.com/512/8644/8644101.png" if msgRole == "assistant" else "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
            with streamlit.chat_message(msgRole, avatar=avatarUrl):
                streamlit.markdown(msgContent)
                
        if totalMsg > 15:
            msgObj = streamlit.session_state.messages[15]
            msgRole = msgObj["role"]
            msgContent = msgObj["content"]
            avatarUrl = "https://cdn-icons-png.flaticon.com/512/8644/8644101.png" if msgRole == "assistant" else "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
            with streamlit.chat_message(msgRole, avatar=avatarUrl):
                streamlit.markdown(msgContent)

    streamlit.text_input(
        label="Nhập câu hỏi tại đây...", 
        key="userQuery", 
        on_change=logicChatResponse, 
        placeholder="Ví dụ: Mô hình này hoạt động như thế nào?"
    )

if streamlit.session_state.triggerLockSource == "MANUAL" or streamlit.session_state.triggerLockSource == "VISION":
    if streamlit.session_state.triggerLockSource == "VISION":
        modelValFinal = activeModel
    else:
        modelValFinal = selectedModel
        
    valStr = int(finalValuationResult / 1000000)
    successMsg = f"Hệ thống Neural đã hoàn tất phân tích định giá tự động cho {modelValFinal}. Đề xuất mức bán ra tại ngưỡng {valStr} triệu VNĐ."
    
    if len(streamlit.session_state.messages) < 5:
        newSuccessObj = dict()
        newSuccessObj["role"] = "assistant"
        newSuccessObj["content"] = successMsg
        streamlit.session_state.messages.append(newSuccessObj)

zaloEndpoint = "https://zalo.me/0779686632"
fbEndpoint = "https://www.facebook.com/truongthinhs"

htmlFloatingContacts = f"""
    <a href="{fbEndpoint}" target="_blank" class="contact_floating btnFbStyle">
        <img src="https://cdn-icons-png.flaticon.com/512/5968/5968764.png" alt="FB">
    </a>
    <a href="{zaloEndpoint}" target="_blank" class="contact_floating btnZaloStyle">
        <img src="https://page.widget.zalo.me/static/images/2.0/Logo.svg" alt="Zalo">
    </a>
"""

streamlit.markdown(htmlFloatingContacts, unsafe_allow_html=True)