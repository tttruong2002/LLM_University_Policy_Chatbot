import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- CẤU HÌNH CƠ BẢN ---

# Tải biến môi trường (cần file .env chứa GROQ_API_KEY)
try:
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        st.error("⚠️ Không tìm thấy GROQ_API_KEY. Vui lòng tạo file .env và thêm API key vào.")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Lỗi khi tải file .env: {e}")
    st.stop()

# Cấu hình đường dẫn và tên collection
DB_PATH = "./chroma_db"
COLLECTION_NAME = "academic_regulations"

# Kiểm tra xem thư mục ChromaDB có tồn tại không
if not os.path.exists(DB_PATH):
    st.error(f"❌ Không tìm thấy thư mục '{DB_PATH}'.")
    st.error("Vui lòng chạy file '01_Data_Ingestion.ipynb' trước để tạo database.")
    st.stop()

# --- TẢI PIPELINE RAG (SỬ DỤNG CACHE) ---

# st.cache_resource: Chỉ chạy hàm này 1 LẦN DUY NHẤT khi app khởi động
# Giúp tiết kiệm thời gian, không cần tải lại model và DB mỗi khi user hỏi
@st.cache_resource
def load_rag_pipeline():
    """
    Tải và khởi tạo toàn bộ pipeline RAG (LLM, Embedding, DB, Chain).
    """
    try:
        # 1. Khởi tạo LLM (Groq)
        llm = ChatGroq(
            model="groq/compound",
            temperature=0,
            api_key=GROQ_API_KEY
        )
        
        # 2. Khởi tạo Embedding Model (HuggingFace)
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 3. Tải Vector Store (Chroma)
        vectorstore = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME
        )
        
        # 4. Tạo Retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Lấy 5 chunks liên quan
        
        # 5. Tạo Prompt Template
        rag_template = """
        Bạn là một trợ lý AI hữu ích, chuyên trả lời các câu hỏi về quy định học thuật của trường 
        ĐH Sư Phạm Kỹ Thuật TP.HCM dựa trên các văn bản được cung cấp.
        Hãy trả lời câu hỏi của người dùng một cách ngắn gọn và chính xác, 
        chỉ dựa vào nội dung trong phần "Văn bản tham khảo" dưới đây.
        KHÔNG được bịa đặt thông tin. Nếu không tìm thấy, hãy nói "Tôi không tìm thấy thông tin này trong tài liệu."

        Văn bản tham khảo:
        {context}

        Câu hỏi:
        {question}

        Câu trả lời (chỉ dựa trên văn bản):
        """
        rag_prompt = ChatPromptTemplate.from_template(rag_template)
        
        # 6. Hàm gộp context
        def format_context(docs):
            return "\n\n---\n\n".join([d.page_content for d in docs])
        
        # 7. Tạo RAG Chain hoàn chỉnh
        # (Sử dụng logic chuẩn của LangChain, tương tự file 02)
        rag_chain = (
            {"context": retriever | format_context, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain

    except Exception as e:
        # Nếu có lỗi ở bất kỳ bước nào, hiển thị lỗi và dừng app
        st.error(f"Lỗi nghiêm trọng khi tải RAG pipeline: {e}")
        st.stop()

# --- GIAO DIỆN ỨNG DỤNG STREAMLIT ---

# Cấu hình tiêu đề trang
st.set_page_config(page_title="Chatbot Quy định HCMUTE", page_icon="🤖")

st.title("🤖 Chatbot Quy định Học vụ HCMUTE")
st.caption(f"Backend: Groq (compound) | DB: ChromaDB (16 files, 156 chunks) | Giao diện: Streamlit")

# Tải pipeline RAG
with st.spinner("⏳ Đang tải mô hình LLM và cơ sở dữ liệu vector..."):
    try:
        rag_chain = load_rag_pipeline()
        st.success("✅ Tải thành công! Chatbot đã sẵn sàng.")
    except Exception as e:
        # Lỗi này đã được xử lý bên trong hàm `load_rag_pipeline`
        pass

# Khởi tạo lịch sử chat (lưu trong session_state)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị các tin nhắn cũ
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Xử lý input mới từ người dùng
if prompt := st.chat_input("Bạn muốn hỏi gì về quy định của trường?"):
    
    # 1. Hiển thị tin nhắn của user lên giao diện
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Tạo phản hồi từ bot
    with st.chat_message("assistant"):
        # Hiệu ứng "đang gõ..."
        response_placeholder = st.empty()
        full_response = ""
        
        # Bắt đầu stream câu trả lời từ RAG chain
        try:
            for chunk in rag_chain.stream(prompt):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌") # Thêm con trỏ "gõ"
            
            # Hiển thị câu trả lời hoàn chỉnh
            response_placeholder.markdown(full_response)
        
        except Exception as e:
            st.error(f"Lỗi khi gọi RAG chain: {e}")
            full_response = "Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý."
            response_placeholder.markdown(full_response)

    # 3. Lưu tin nhắn của bot vào lịch sử
    st.session_state.messages.append({"role": "assistant", "content": full_response})