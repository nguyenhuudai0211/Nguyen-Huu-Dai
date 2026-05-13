import pandas as pd
import streamlit as st
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cấu hình giao diện
st.set_page_config(
    page_title="Dự đoán Tiểu đường - Perceptron", page_icon="🩺", layout="wide"
)


# 1. ĐỌC DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH (GIỮ NGUYÊN LOGIC CỦA BẠN)
@st.cache_resource
def train_perceptron_models():
    try:
        df = pd.read_csv("diabetes.csv")
    except Exception:
        st.error(
            "Không tìm thấy file 'diabetes.csv'. Vui lòng để file vào cùng thư mục."
        )
        return None, None, None, None, None, None, None, None

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Model 1
    model1 = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
    model1.fit(X_train, y_train.values.ravel())
    y_pred1 = model1.predict(X_test_std)
    acc1 = accuracy_score(y_test, y_pred1)

    # Model 2
    model2 = Perceptron(max_iter=1000, eta0=0.05, random_state=42)
    model2.fit(X_train_std, y_train.values.ravel())
    y_pred2 = model2.predict(X_test_std)
    acc2 = accuracy_score(y_test, y_pred2)

    mean_healthy = df[df["Outcome"] == 0].mean()
    mean_diabetic = df[df["Outcome"] == 1].mean()

    return model1, model2, sc, acc1, acc2, df, X.columns, mean_healthy, mean_diabetic


(
    model1,
    model2,
    sc,
    acc1,
    acc2,
    df,
    feature_names,
    mean_healthy,
    mean_diabetic,
) = train_perceptron_models()


# --- HÀM HỖ TRỢ ĐỒNG BỘ SLIDER VÀ NUMBER INPUT ---
def sync_input(field_name):
    """Đồng bộ giá trị từ number_input sang slider"""
    st.session_state[f"{field_name}_slider"] = st.session_state[
        f"{field_name}_num"
    ]


def sync_slider(field_name):
    """Đồng bộ giá trị từ slider sang number_input"""
    st.session_state[f"{field_name}_num"] = st.session_state[
        f"{field_name}_slider"
    ]


# Khởi tạo giá trị mặc định trong Session State nếu chưa có
defaults = {
    "preg": 1,
    "gluc": 120,
    "bp": 70,
    "skin": 20,
    "ins": 79,
    "bmi": 25.0,
    "dpf": 0.45,
    "age": 30,
}
for key, val in defaults.items():
    if f"{key}_slider" not in st.session_state:
        st.session_state[f"{key}_slider"] = val
    if f"{key}_num" not in st.session_state:
        st.session_state[f"{key}_num"] = val

# 2. GIAO DIỆN CHÍNH
st.title("🩺 Ứng dụng Dự đoán Tiểu đường bằng mô hình Perceptron")
st.markdown(
    "Hệ thống áp dụng thuật toán **Linear Perceptron** hỗ trợ nhập liệu linh hoạt (kéo thanh trượt hoặc gõ số trực tiếp)."
)
st.divider()

if model2 is not None:
    # --- BÊN TRÁI: SIDEBAR NHẬP LIỆU KÉP ---
    st.sidebar.header("⚙️ Nhập chỉ số kiểm tra")
    st.sidebar.caption("💡 Bạn có thể kéo thanh trượt hoặc gõ số trực tiếp:")

    # 1. Mang thai (Pregnancies)
    st.sidebar.markdown("**1. Số lần mang thai**")
    col1_p, col2_p = st.sidebar.columns([2, 1])
    with col1_p:
        st.slider(
            "Preg_slider",
            0,
            17,
            key="preg_slider",
            on_change=sync_slider,
            args=("preg",),
            label_visibility="collapsed",
        )
    with col2_p:
        st.number_input(
            "Preg_num",
            0,
            17,
            key="preg_num",
            on_change=sync_input,
            args=("preg",),
            label_visibility="collapsed",
        )

    # 2. Glucose
    st.sidebar.markdown("**2. Đường huyết (Glucose)**")
    col1_g, col2_g = st.sidebar.columns([2, 1])
    with col1_g:
        st.slider(
            "Gluc_slider",
            0,
            200,
            key="gluc_slider",
            on_change=sync_slider,
            args=("gluc",),
            label_visibility="collapsed",
        )
    with col2_g:
        st.number_input(
            "Gluc_num",
            0,
            200,
            key="gluc_num",
            on_change=sync_input,
            args=("gluc",),
            label_visibility="collapsed",
        )

    # 3. Huyết áp (Blood Pressure)
    st.sidebar.markdown("**3. Huyết áp (BP - mmHg)**")
    col1_bp, col2_bp = st.sidebar.columns([2, 1])
    with col1_bp:
        st.slider(
            "Bp_slider",
            0,
            122,
            key="bp_slider",
            on_change=sync_slider,
            args=("bp",),
            label_visibility="collapsed",
        )
    with col2_bp:
        st.number_input(
            "Bp_num",
            0,
            122,
            key="bp_num",
            on_change=sync_input,
            args=("bp",),
            label_visibility="collapsed",
        )

    # 4. Độ dày da (Skin Thickness)
    st.sidebar.markdown("**4. Độ dày da (mm)**")
    col1_sk, col2_sk = st.sidebar.columns([2, 1])
    with col1_sk:
        st.slider(
            "Skin_slider",
            0,
            99,
            key="skin_slider",
            on_change=sync_slider,
            args=("skin",),
            label_visibility="collapsed",
        )
    with col2_sk:
        st.number_input(
            "Skin_num",
            0,
            99,
            key="skin_num",
            on_change=sync_input,
            args=("skin",),
            label_visibility="collapsed",
        )

    # 5. Insulin
    st.sidebar.markdown("**5. Lượng Insulin**")
    col1_in, col2_in = st.sidebar.columns([2, 1])
    with col1_in:
        st.slider(
            "Ins_slider",
            0,
            846,
            key="ins_slider",
            on_change=sync_slider,
            args=("ins",),
            label_visibility="collapsed",
        )
    with col2_in:
        st.number_input(
            "Ins_num",
            0,
            846,
            key="ins_num",
            on_change=sync_input,
            args=("ins",),
            label_visibility="collapsed",
        )

    # 6. BMI
    st.sidebar.markdown("**6. Chỉ số BMI**")
    col1_bmi, col2_bmi = st.sidebar.columns([2, 1])
    with col1_bmi:
        st.slider(
            "Bmi_slider",
            0.0,
            68.0,
            step=0.1,
            key="bmi_slider",
            on_change=sync_slider,
            args=("bmi",),
            label_visibility="collapsed",
        )
    with col2_bmi:
        st.number_input(
            "Bmi_num",
            0.0,
            68.0,
            step=0.1,
            key="bmi_num",
            on_change=sync_input,
            args=("bmi",),
            label_visibility="collapsed",
        )

    # 7. DPF (Hệ số di truyền)
    st.sidebar.markdown("**7. Hệ số di truyền (DPF)**")
    col1_dpf, col2_dpf = st.sidebar.columns([2, 1])
    with col1_dpf:
        st.slider(
            "Dpf_slider",
            0.07,
            2.42,
            step=0.01,
            key="dpf_slider",
            on_change=sync_slider,
            args=("dpf",),
            label_visibility="collapsed",
        )
    with col2_dpf:
        st.number_input(
            "Dpf_num",
            0.07,
            2.42,
            step=0.01,
            key="dpf_num",
            on_change=sync_input,
            args=("dpf",),
            label_visibility="collapsed",
        )

    # 8. Tuổi (Age)
    st.sidebar.markdown("**8. Tuổi (Age)**")
    col1_age, col2_age = st.sidebar.columns([2, 1])
    with col1_age:
        st.slider(
            "Age_slider",
            21,
            81,
            key="age_slider",
            on_change=sync_slider,
            args=("age",),
            label_visibility="collapsed",
        )
    with col2_age:
        st.number_input(
            "Age_num",
            21,
            81,
            key="age_num",
            on_change=sync_input,
            args=("age",),
            label_visibility="collapsed",
        )

    # Thu thập dữ liệu cuối cùng từ trạng thái (dùng giá trị num hoặc slider đều giống nhau vì đã đồng bộ)
    input_data = pd.DataFrame(
        {
            "Pregnancies": [st.session_state["preg_num"]],
            "Glucose": [st.session_state["gluc_num"]],
            "BloodPressure": [st.session_state["bp_num"]],
            "SkinThickness": [st.session_state["skin_num"]],
            "Insulin": [st.session_state["ins_num"]],
            "BMI": [st.session_state["bmi_num"]],
            "DiabetesPedigreeFunction": [st.session_state["dpf_num"]],
            "Age": [st.session_state["age_num"]],
        }
    )

    st.subheader("📋 Bảng thông số hồ sơ đầu vào")
    st.dataframe(input_data, use_container_width=True)

    # NÚT DỰ ĐOÁN
    if st.button("🚀 Thực hiện Phân tích & Dự đoán", type="primary"):
        input_std = sc.transform(input_data)
        prediction = model2.predict(input_std)[0]
        decision_score = model2.decision_function(input_std)[0]

        st.divider()
        st.header("🎯 KẾT QUẢ CHẨN ĐOÁN (SỬ DỤNG MODEL 2)")

        col1, col2 = st.columns([2, 1])

        with col1:
            if prediction == 1:
                st.error("### ⚠️ PHÁT HIỆN NGUY CƠ MẮC BỆNH TIỂU ĐƯỜNG")
                st.write(
                    "Mô hình phân loại tuyến tính đánh giá các chỉ số của bạn nằm ở miền dương của đường biên quyết định. Bạn nên đến cơ sở y tế để được thăm khám chi tiết."
                )
            else:
                st.success("### ✅ CHỈ SỐ AN TOÀN (NGUY CƠ THẤP)")
                st.write(
                    "Các chỉ số hiện tại nằm ở miền âm của đường biên quyết định. Mô hình chẩn đoán bạn có rủi ro thấp."
                )

        with col2:
            st.metric(
                label="Điểm biên quyết định (Score)",
                value=f"{decision_score:.2f}",
                delta="Nguy cơ cao" if decision_score > 0 else "An toàn",
                delta_color="inverse",
            )
            st.caption(
                "*(Điểm > 0: Dự đoán mắc bệnh | Điểm < 0: Dự đoán không mắc)*"
            )

        st.divider()

        # --- PHẦN PHÂN TÍCH CHUYÊN SÂU ---
        st.subheader(
            "📊 1. Đánh giá & So sánh mô hình trên tập kiểm thử (Test Size = 30%)"
        )
        acc_df = pd.DataFrame(
            {
                "Mô hình cấu hình": [
                    "Model 1 (eta0=0.1, Fit data gốc, Predict data chuẩn hóa)",
                    "Model 2 (eta0=0.05, Fit & Predict hoàn toàn trên data chuẩn hóa)",
                ],
                "Độ chính xác (Accuracy)": [f"{acc1*100:.2f}%", f"{acc2*100:.2f}%"],
            }
        )
        st.dataframe(acc_df, use_container_width=True)

        st.divider()

        st.subheader("⚖️ 2. Trọng số học được của Perceptron (Model 2)")
        weights = model2.coef_[0]
        coef_df = pd.DataFrame(
            {"Chỉ số": feature_names, "Trọng số (Weight)": weights}
        ).sort_values(by="Trọng số (Weight)", ascending=False)

        col_w1, col_w2 = st.columns([1, 1])
        with col_w1:
            st.dataframe(coef_df, use_container_width=True)
        with col_w2:
            st.bar_chart(
                coef_df.set_index("Chỉ số"), horizontal=True, color="#ff5733"
            )

        st.divider()

        st.subheader("📈 3. Đối chiếu chi tiết với dữ liệu lâm sàng")
        comp_df = pd.DataFrame(
            {
                "Chỉ số": [
                    "Pregnancies",
                    "Glucose",
                    "BloodPressure",
                    "SkinThickness",
                    "Insulin",
                    "BMI",
                    "DPF",
                    "Age",
                ],
                "Của bạn": [
                    st.session_state["preg_num"],
                    st.session_state["gluc_num"],
                    st.session_state["bp_num"],
                    st.session_state["skin_num"],
                    st.session_state["ins_num"],
                    st.session_state["bmi_num"],
                    st.session_state["dpf_num"],
                    st.session_state["age_num"],
                ],
                "TB Khỏe mạnh": [
                    round(mean_healthy["Pregnancies"], 1),
                    round(mean_healthy["Glucose"], 1),
                    round(mean_healthy["BloodPressure"], 1),
                    round(mean_healthy["SkinThickness"], 1),
                    round(mean_healthy["Insulin"], 1),
                    round(mean_healthy["BMI"], 1),
                    round(mean_healthy["DiabetesPedigreeFunction"], 2),
                    round(mean_healthy["Age"], 1),
                ],
                "TB Mắc bệnh": [
                    round(mean_diabetic["Pregnancies"], 1),
                    round(mean_diabetic["Glucose"], 1),
                    round(mean_diabetic["BloodPressure"], 1),
                    round(mean_diabetic["SkinThickness"], 1),
                    round(mean_diabetic["Insulin"], 1),
                    round(mean_diabetic["BMI"], 1),
                    round(mean_diabetic["DiabetesPedigreeFunction"], 2),
                    round(mean_diabetic["Age"], 1),
                ],
            }
        )
        st.dataframe(comp_df, use_container_width=True)

    st.divider()
    with st.expander("📂 Xem toàn bộ tập dữ liệu gốc"):
        st.dataframe(df, use_container_width=True)