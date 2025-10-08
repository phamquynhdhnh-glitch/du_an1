import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# --- Khởi tạo State cho Chatbot và Dữ liệu ---
if "df_processed" not in st.session_state:
    st.session_state["df_processed"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []

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
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Nàu sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini cho Phân tích Báo cáo (Chức năng 5) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        system_instruction = "Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính được cung cấp, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành."
        
        prompt = f"""
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- HÀM MỚI: Xử lý Chat với Gemini ---
def get_chat_response(prompt, processed_df_markdown, api_key):
    """Xử lý câu hỏi chat với Gemini, sử dụng dữ liệu tài chính làm ngữ cảnh."""
    try:
        client = genai.Client(api_key=api_key)
        
        # Tạo ngữ cảnh (System Instruction)
        system_instruction = f"""
        Bạn là trợ lý phân tích tài chính thông minh (Financial Analyst Assistant). 
        Tất cả các câu trả lời của bạn phải dựa trên Bảng Báo cáo Tài chính đã được xử lý sau đây.
        Hãy trả lời các câu hỏi của người dùng một cách ngắn gọn, chính xác, và chỉ dựa trên dữ liệu bạn có. 
        Nếu người dùng hỏi câu hỏi không liên quan đến dữ liệu tài chính, hãy lịch sự từ chối.
        
        Dữ liệu tài chính được phân tích:
        {processed_df_markdown}
        """
        
        # Gửi toàn bộ lịch sử chat và prompt hiện tại để giữ ngữ cảnh cuộc trò chuyện
        contents = []
        for message in st.session_state["messages"]:
            contents.append(
                {"role": "user" if message["role"] == "user" else "model", "parts": [{"text": message["content"]}]}
            )
        
        # Thêm prompt hiện tại của người dùng
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

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
        
        # LƯU DỮ LIỆU ĐÃ XỬ LÝ VÀO SESSION STATE
        st.session_state['df_processed'] = df_processed

        # TẠO CÁC TAB
        tab_analysis, tab_chat = st.tabs(["Phân Tích Báo Cáo", "Hỏi Đáp Chuyên Sâu với AI"])

        with tab_analysis:
            if df_processed is not None:
                
                # --- Chức năng 2 & 3: Hiển thị Kết quả ---
                st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
                st.dataframe(df_processed.style.format({
                    'Năm trước': '{:,.0f}',
                    'Năm sau': '{:,.0f}',
                    'Tốc độ tăng trưởng (%)': '{:.2f}%',
                    'Tỷ trọng Năm trước (%)': '{:.2f}%',
                    'Tỷ trọng Năm sau (%)': '{:.2f}%'
                }), use_container_width=True)
                
                # --- Chức năng 4: Tính Chỉ số Tài chính ---
                st.subheader("4. Các Chỉ số Tài chính Cơ bản")
                
                thanh_toan_hien_hanh_N = "N/A"
                thanh_toan_hien_hanh_N_1 = "N/A"
                
                try:
                    # Lấy Tài sản ngắn hạn
                    tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                    tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                    # Lấy Nợ ngắn hạn
                    no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                    no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                    # Tính toán
                    thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                    thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                            value=f"{thanh_toan_hien_hanh_N_1:.2f} lần"
                        )
                    with col2:
                        st.metric(
                            label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                            value=f"{thanh_toan_hien_hanh_N:.2f} lần",
                            delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                        )
                        
                except IndexError:
                    st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                except ZeroDivisionError:
                    st.warning("Lỗi chia cho 0 khi tính chỉ số thanh toán hiện hành (Nợ ngắn hạn bằng 0).")
                    
                # --- Chức năng 5: Nhận xét AI ---
                st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
                
                # Chuẩn bị dữ liệu để gửi cho AI
                data_for_ai = pd.DataFrame({
                    'Chỉ tiêu': [
                        'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                        'Thanh toán hiện hành (N-1)', 
                        'Thanh toán hiện hành (N)'
                    ],
                    'Giá trị': [
                        df_processed.to_markdown(index=False),
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

        # =========================================================================
        # KHUNG CHAT MỚI
        # =========================================================================
        with tab_chat:
            st.subheader("Trò chuyện cùng Trợ lý Phân tích (Sử dụng dữ liệu bạn đã tải)")
            
            api_key_chat = st.secrets.get("GEMINI_API_KEY") 
            if not api_key_chat:
                st.error("Không thể sử dụng tính năng Chat: Vui lòng cấu hình Khóa 'GEMINI_API_KEY'.")
            else:
                # 1. Hiển thị lịch sử chat
                for message in st.session_state["messages"]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # 2. Xử lý input mới
                if prompt := st.chat_input("Hỏi về Tốc độ tăng trưởng, tỷ trọng, hoặc bất kỳ chỉ tiêu nào..."):
                    
                    # Thêm prompt của người dùng vào lịch sử
                    st.session_state["messages"].append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Chuẩn bị dữ liệu để đưa vào ngữ cảnh chat
                    df_markdown_context = st.session_state['df_processed'].to_markdown(index=False)

                    # Gọi API và hiển thị phản hồi
                    with st.chat_message("assistant"):
                        with st.spinner("Đang tìm kiếm câu trả lời từ dữ liệu..."):
                            response = get_chat_response(prompt, df_markdown_context, api_key_chat)
                            st.markdown(response)
                            
                    # Thêm phản hồi của AI vào lịch sử
                    st.session_state["messages"].append({"role": "assistant", "content": response})

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file và cột.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích và sử dụng khung chat.")
