import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính & Chatbot 📊💬")

# ------------------------------------------------------------------
# Khởi tạo trạng thái phiên (Session State) cho Chatbot
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "data_context" not in st.session_state:
    st.session_state["data_context"] = None # Sẽ chứa df_processed dưới dạng markdown
if "df_processed" not in st.session_state:
    st.session_state["df_processed"] = None # Lưu trữ dataframe để dùng cho tính toán chỉ số

# ------------------------------------------------------------------
# Hàm gọi API Gemini cho Phân tích Báo cáo (Chức năng 5 - Giữ nguyên)
# ------------------------------------------------------------------
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        # LƯU Ý: Khởi tạo Client mỗi lần gọi hàm vì Streamlit chạy lại script
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# ------------------------------------------------------------------
# NEW: Hàm gọi API Gemini cho Chatbot (Hỗ trợ Hội thoại)
# ------------------------------------------------------------------
def chat_with_gemini(user_query, data_context, api_key, history):
    """Gửi tin nhắn chat đến Gemini API, cung cấp ngữ cảnh dữ liệu tài chính."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'

        # Xây dựng System Prompt để cung cấp ngữ cảnh (data_context)
        system_prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp và tận tâm.
        Dưới đây là dữ liệu tài chính (Bảng cân đối kế toán) đã được phân tích Tăng trưởng và Tỷ trọng của một doanh nghiệp:
        --- DỮ LIỆU TÀI CHÍNH ĐÃ XỬ LÝ ---
        {data_context}
        --- HẾT DỮ LIỆU ---
        
        Hãy trả lời các câu hỏi của người dùng dựa trên ngữ cảnh này. Nếu câu hỏi không liên quan đến tài chính, hãy lịch sự từ chối và đề nghị họ hỏi về báo cáo tài chính.
        Hãy duy trì cuộc trò chuyện ngắn gọn và hữu ích.
        """
        
        # Chuyển đổi lịch sử chat của Streamlit sang định dạng của Gemini API
        contents = [
            {"role": "user" if msg["role"] == "user" else "model", "parts": [{"text": msg["content"]}]}
            for msg in history
        ]
        
        # Thêm câu hỏi hiện tại của người dùng
        contents.append({"role": "user", "parts": [{"text": user_query}]})

        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            system_instruction={"parts": [{"text": system_prompt}]}
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        # Nếu không tìm thấy, cố gắng tìm kiếm với regex linh hoạt hơn (ví dụ: 'tổng tài sản')
        tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG.*TÀI SẢN', case=False, na=False)]
        if tong_tai_san_row.empty:
             raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df


# ------------------------------------------------------------------
# Cấu trúc UI bằng Tabs
# ------------------------------------------------------------------
tab_analysis, tab_chat = st.tabs(["📊 Phân Tích Báo Cáo", "💬 Hỏi Đáp AI (Chatbot)"])

with tab_analysis:
    # --- Chức năng 1: Tải File ---
    uploaded_file = st.file_uploader(
        "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
        type=['xlsx', 'xls']
    )

    if uploaded_file is not None:
        try:
            df_raw = pd.read_excel(uploaded_file)
            
            # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
            df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
            
            # Xử lý dữ liệu
            df_processed = process_financial_data(df_raw.copy())

            if df_processed is not None:
                
                # CẬP NHẬT CONTEXT cho Chatbot
                st.session_state["data_context"] = df_processed.to_markdown(index=False)
                st.session_state["df_processed"] = df_processed # Lưu trữ để dùng lại logic tính chỉ số
                
                # --- Chức năng 2 & 3: Hiển thị Kết quả ---
                st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
                st.dataframe(df_processed.style.format({
                    'Năm trước': '{:,.0f}',
                    'Năm sau': '{:,.0f}',
                    'Tốc độ tăng trưởng (%)': '{:.2f}%',
                    'Tỷ trọng Năm trước (%)': '{:.2f}%',
                    'Tỷ trọng Năm sau (%)': '{:.2f}%'
                }), use_container_width=True)
                
                # Biến dùng tạm cho chỉ số thanh toán
                thanh_toan_hien_hanh_N = "N/A"
                thanh_toan_hien_hanh_N_1 = "N/A"

                # --- Chức năng 4: Tính Chỉ số Tài chính ---
                st.subheader("4. Các Chỉ số Tài chính Cơ bản")
                
                try:
                    # Lấy Tài sản ngắn hạn
                    tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                    tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                    # Lấy Nợ ngắn hạn
                    no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]    
                    no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                    # Xử lý chia cho 0
                    if no_ngan_han_N != 0:
                         thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                    if no_ngan_han_N_1 != 0:
                         thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                            value=f"{thanh_toan_hien_hanh_N_1:.2f} lần" if isinstance(thanh_toan_hien_hanh_N_1, float) else thanh_toan_hien_hanh_N_1
                        )
                    with col2:
                        st.metric(
                            label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                            value=f"{thanh_toan_hien_hanh_N:.2f} lần" if isinstance(thanh_toan_hien_hanh_N, float) else thanh_toan_hien_hanh_N,
                            delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}" if isinstance(thanh_toan_hien_hanh_N, float) and isinstance(thanh_toan_hien_hanh_N_1, float) else None
                        )
                        
                except IndexError:
                    st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số. Vui lòng kiểm tra lại file.")
                    thanh_toan_hien_hanh_N = "N/A" 
                    thanh_toan_hien_hanh_N_1 = "N/A"
                except ZeroDivisionError:
                    st.warning("Không thể tính Chỉ số Thanh toán Hiện hành do giá trị Nợ ngắn hạn bằng 0.")
                    thanh_toan_hien_hanh_N = "Không tính được (Nợ NH=0)"
                    thanh_toan_hien_hanh_N_1 = "Không tính được (Nợ NH=0)"


                # --- Chức năng 5: Nhận xét AI (Giữ nguyên) ---
                st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
                
                # Chuẩn bị dữ liệu để gửi cho AI
                data_for_ai = pd.DataFrame({
                    'Chỉ tiêu': [
                        'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                        'Tăng trưởng Tài sản ngắn hạn (%)', 
                        'Thanh toán hiện hành (N-1)', 
                        'Thanh toán hiện hành (N)'
                    ],
                    'Giá trị': [
                        st.session_state["data_context"],
                        f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%" if not df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)].empty else "N/A",
                        f"{thanh_toan_hien_hanh_N_1}", 
                        f"{thanh_toan_hien_hanh_N}"
                    ]
                }).to_markdown(index=False) 

                if st.button("Yêu cầu AI Phân tích"):
                    api_key = st.secrets.get("GEMINI_API_KEY") 
                    
                    if api_key:
                        with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                            ai_result = get_ai_analysis(data_for_ai, api_key)
                            st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                            st.info(ai_result)
                    else:
                        st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

        except ValueError as ve:
            st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
        except Exception as e:
            st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

    else:
        st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
        st.session_state["data_context"] = None # Reset context nếu file bị xóa
        st.session_state["df_processed"] = None


# ------------------------------------------------------------------
# Chức năng Chatbot (Tab mới)
# ------------------------------------------------------------------
with tab_chat:
    st.header("💬 Hỏi Đáp Chuyên Sâu với Gemini AI")
    st.markdown("Bạn có thể đặt các câu hỏi về dữ liệu tài chính đã tải lên (ví dụ: *'Tài sản ngắn hạn thay đổi thế nào giữa hai năm?'*).")
    
    # Lấy API Key
    api_key = st.secrets.get("GEMINI_API_KEY")

    if st.session_state["data_context"] is None:
        st.warning("Vui lòng tải file Báo cáo Tài chính ở tab **📊 Phân Tích Báo Cáo** để bắt đầu trò chuyện.")
    elif not api_key:
        st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")
    else:
        # Hiển thị lịch sử tin nhắn
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Xử lý input từ người dùng
        if prompt := st.chat_input("Hỏi Gemini về báo cáo tài chính..."):
            
            # 1. Thêm tin nhắn người dùng vào lịch sử
            st.session_state["messages"].append({"role": "user", "content": prompt})
            # Hiển thị tin nhắn người dùng
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. Gọi API để lấy phản hồi
            with st.chat_message("assistant"):
                with st.spinner("Gemini đang phân tích và trả lời..."):
                    
                    # Gọi hàm chat mới với dữ liệu và lịch sử
                    full_response = chat_with_gemini(
                        user_query=prompt, 
                        data_context=st.session_state["data_context"], 
                        api_key=api_key,
                        history=st.session_state["messages"][:-1] # Lịch sử cũ (loại bỏ prompt hiện tại)
                    )
                
                # Hiển thị phản hồi
                st.markdown(full_response)
            
            # 3. Thêm phản hồi AI vào lịch sử
            st.session_state["messages"].append({"role": "assistant", "content": full_response})
