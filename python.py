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

 # --- KHỞI TẠO STATE CHO CHAT VÀ PHÂN TÍCH ---
 # Sử dụng st.session_state để lưu trữ lịch sử chat và đối tượng chat session
 if "chat_history" not in st.session_state:
     st.session_state["chat_history"] = []
 # Đối tượng genai.Chat session để duy trì ngữ cảnh
 if "chat_session" not in st.session_state:
     st.session_state["chat_session"] = None
 # Lưu trữ context dữ liệu tài chính (dạng Markdown) cho AI
 if "analysis_data_context" not in st.session_state:
     st.session_state["analysis_data_context"] = None


 # --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) --- 
 @st.cache_data 
 def process_financial_data(df): 
     """Thực hiện các phép tính Tăng trưởng và Tỷ trọng.""" 
      
     # Đảm bảo các giá trị là số để tính toán 
     numeric_cols = ['Năm trước', 'Năm sau'] 
     for col in numeric_cols: 
         df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) 
      
     # 1. Tính Tốc độ Tăng trưởng 
     # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0 
     df['Tốc độ tăng trưởng (%)'] = ( 
         (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9) 
     ) * 100 

     # 2. Tính Tỷ trọng theo Tổng Tài sản 
     # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN" 
     tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)] 
      
     if tong_tai_san_row.empty: 
         raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.") 

     tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0] 
     tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0] 

     # ******************************* PHẦN SỬA LỖI BẮT ĐẦU *******************************     # Lỗi xảy ra khi dùng .replace() trên giá trị đơn lẻ (numpy.int64). 
     # Sử dụng điều kiện ternary để xử lý giá trị 0 thủ công cho mẫu số. 
      
     divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9 
     divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9 

     # Tính tỷ trọng với mẫu số đã được xử lý 
     df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100 
     df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100 
     # ******************************* PHẦN SỬA LỖI KẾT THÚC *******************************      
     return df 

 # --- Hàm gọi API Gemini cho Phân Tích Ban Đầu (Chức năng 5) --- 
 def get_ai_analysis(data_for_ai, api_key): 
     """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét.""" 
     try: 
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

 # --- Hàm xử lý Chat (Chức năng 6) ---
 def handle_chat_input(user_input, api_key):
     """
     Gửi tin nhắn người dùng đến Gemini Chat Session và cập nhật lịch sử.
     Khởi tạo session nếu chưa có, sử dụng dữ liệu tài chính làm context.
     """
     if not api_key:
         st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY'.")
         return

     try:
         client = genai.Client(api_key=api_key)
         model_name = 'gemini-2.5-flash'
         
         # 1. Khởi tạo Chat Session nếu chưa có
         if st.session_state["chat_session"] is None:
             # Dùng context dữ liệu tài chính làm System Instruction 
             system_instruction = f"""
             Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Bạn đang giúp người dùng đặt câu hỏi chuyên sâu về báo cáo tài chính đã được tải lên và phân tích.
             Tất cả câu trả lời của bạn phải dựa trên dữ liệu tài chính sau:
             --- Dữ liệu Báo cáo Tài chính Đã Xử Lý ---
             {st.session_state.analysis_data_context}
             ---
             Hãy trả lời ngắn gọn, chuyên nghiệp và chỉ sử dụng tiếng Việt. Nếu câu hỏi không liên quan đến dữ liệu tài chính, hãy lịch sự từ chối và yêu cầu hỏi các chỉ tiêu liên quan.
             """
             
             st.session_state["chat_session"] = client.chats.create(
                 model=model_name,
                 system_instruction=system_instruction
             )
         
         # 2. Gửi tin nhắn và cập nhật lịch sử
         # Thêm tin nhắn người dùng vào lịch sử
         st.session_state.chat_history.append({"role": "user", "text": user_input})
         
         # Gửi tin nhắn đến API
         response = st.session_state.chat_session.send_message(user_input)
         
         # Thêm phản hồi của AI vào lịch sử
         st.session_state.chat_history.append({"role": "assistant", "text": response.text})

     except APIError as e:
         st.error(f"Lỗi Chat Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}")
     except Exception as e:
         st.error(f"Đã xảy ra lỗi không xác định trong quá trình chat: {e}")


 # --- Chức năng 1: Tải File --- 
 uploaded_file = st.file_uploader( 
     "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)", 
     type=['xlsx', 'xls'] 
 ) 

 if uploaded_file is not None: 
     try: 
         # Reset chat session khi file mới được tải lên
         st.session_state["chat_history"] = []
         st.session_state["chat_session"] = None
         st.session_state["analysis_data_context"] = None
         
         df_raw = pd.read_excel(uploaded_file) 
          
         # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng 
         df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau'] 
          
         # Xử lý dữ liệu 
         df_processed = process_financial_data(df_raw.copy()) 

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
              
             # Khởi tạo giá trị mặc định cho chỉ số
             thanh_toan_hien_hanh_N = "N/A"
             thanh_toan_hien_hanh_N_1 = "N/A"
             tsnh_tang_truong = "N/A"
              
             try: 
                 # Lấy Tài sản ngắn hạn 
                 tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0] 
                 tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0] 
                 tsnh_tang_truong = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]

                 # Lấy Nợ ngắn hạn
                 no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]   
                 no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0] 

                 # Tính toán và xử lý lỗi chia cho 0
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
                     delta_value = thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1 if isinstance(thanh_toan_hien_hanh_N, float) and isinstance(thanh_toan_hien_hanh_N_1, float) else None
                     st.metric( 
                         label="Chỉ số Thanh toán Hiện hành (Năm sau)", 
                         value=f"{thanh_toan_hien_hanh_N:.2f} lần" if isinstance(thanh_toan_hien_hanh_N, float) else thanh_toan_hien_hanh_N, 
                         delta=f"{delta_value:.2f}" if delta_value is not None else None
                     ) 
                      
             except IndexError: 
                  st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.") 
             except ZeroDivisionError:
                 st.warning("Lỗi: Không thể tính toán Chỉ số Thanh toán Hiện hành do Nợ Ngắn Hạn bằng 0.")

             # Chuẩn bị dữ liệu để gửi cho AI (Dữ liệu context cho cả Phân tích ban đầu và Chat)
             data_for_ai = pd.DataFrame({ 
                 'Chỉ tiêu': [ 
                     'Toàn bộ Bảng phân tích (dữ liệu thô)',  
                     'Tăng trưởng Tài sản ngắn hạn (%)',  
                     'Thanh toán hiện hành (N-1)',  
                     'Thanh toán hiện hành (N)' 
                 ], 
                 'Giá trị': [ 
                     df_processed.to_markdown(index=False), 
                     f"{tsnh_tang_truong:.2f}%" if isinstance(tsnh_tang_truong, float) else tsnh_tang_truong,  
                     f"{thanh_toan_hien_hanh_N_1:.2f}" if isinstance(thanh_toan_hien_hanh_N_1, float) else thanh_toan_hien_hanh_N_1,  
                     f"{thanh_toan_hien_hanh_N:.2f}" if isinstance(thanh_toan_hien_hanh_N, float) else thanh_toan_hien_hanh_N
                 ] 
             }).to_markdown(index=False)  
             
             # Lưu context dữ liệu để Chat có thể sử dụng
             st.session_state["analysis_data_context"] = data_for_ai


             # --- Chức năng 5: Nhận xét AI --- 
             st.subheader("5. Nhận xét Tình hình Tài chính (AI)") 
              
             if st.button("Yêu cầu AI Phân tích"): 
                 api_key = st.secrets.get("GEMINI_API_KEY")  
                  
                 if api_key: 
                     with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'): 
                         ai_result = get_ai_analysis(data_for_ai, api_key) 
                         st.markdown("**Kết quả Phân tích từ Gemini AI:**") 
                         st.info(ai_result) 
                 else: 
                      st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.") 

            # --- Chức năng 6: Khung Chat AI (Hỏi Đáp Chuyên Sâu) ---
            st.markdown("---")
            st.subheader("6. Chat AI (Hỏi đáp chuyên sâu về Dữ liệu)")
            
            api_key = st.secrets.get("GEMINI_API_KEY")
            
            # Hiển thị lịch sử chat
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["text"])

            # Khung nhập liệu chat (luôn hiển thị)
            user_input = st.chat_input("Hỏi Gemini về báo cáo tài chính này...")
            
            if user_input:
                if st.session_state.analysis_data_context is None:
                    st.warning("Vui lòng đảm bảo dữ liệu đã được xử lý thành công trước khi chat.")
                else:
                    # Xử lý chat input và nhận phản hồi
                    with st.spinner('AI đang trả lời...'):
                        handle_chat_input(user_input, api_key)
                    # Sau khi xử lý, Streamlit sẽ tự động rerun và hiển thị tin nhắn mới
                    st.rerun() # Bắt buộc phải rerun để cập nhật khung chat

     except ValueError as ve: 
         st.error(f"Lỗi cấu trúc dữ liệu: {ve}") 
     except Exception as e: 
         st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.") 

 else: 
     st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
