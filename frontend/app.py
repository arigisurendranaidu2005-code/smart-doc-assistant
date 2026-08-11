"""
Streamlit UI for Smart Doc Assistant.
"""
import streamlit as st
import requests
import json
import time

# --- CSS Styling ---
st.set_page_config(page_title="Smart Doc Assistant", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background-color: #1e2127;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

API_BASE_URL = "http://localhost:8000"

def get_collections():
    # Mock fallback if API is not running
    try:
        res = requests.get(f"{API_BASE_URL}/collections")
        if res.status_code == 200:
            return res.json()
    except:
        pass
    return [{"name": "default", "document_count": 0}]

def main():
    st.title("🧠 Smart Doc Assistant")
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")
        llm_provider = st.selectbox("LLM Provider", ["OpenAI", "Anthropic", "Local Llama"])
        
        st.header("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload documents", 
            accept_multiple_files=True,
            type=["pdf", "docx", "txt"]
        )
        
        target_collection = st.text_input("Collection Name", value="default")
        
        if st.button("Process Documents") and uploaded_files:
            with st.spinner("Processing files..."):
                progress_bar = st.progress(0)
                # Mock upload process
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                st.success(f"Successfully processed {len(uploaded_files)} files into '{target_collection}'!")
        
        st.header("Collection Management")
        collections = get_collections()
        for col in collections:
            st.markdown(f"**{col['name']}**: {col['document_count']} docs")
            if st.button(f"Delete {col['name']}", key=f"del_{col['name']}"):
                st.warning(f"Deleted {col['name']} (Mock)")
    
    # --- Main Area ---
    # Metrics
    col1, col2, col3 = st.columns(3)
    collections_data = get_collections()
    with col1:
        st.metric("Total Collections", len(collections_data))
    with col2:
        st.metric("Total Documents", sum(c['document_count'] for c in collections_data))
    with col3:
        st.metric("LLM Status", "Online")
        
    # Chat Interface
    st.subheader("Chat Interface")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("View Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"**{src['metadata'].get('source', 'Unknown')}** (Score: {src['score']})")
                        st.text(src['content'])

    # Chat Input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            # Mock generating response
            full_response = f"Here is what I found about: '{prompt}'."
            message_placeholder.markdown(full_response)
            
            mock_sources = [
                {"content": "Relevant snippet from page 1.", "metadata": {"source": "doc1.pdf"}, "score": 0.88},
                {"content": "Another snippet from doc.", "metadata": {"source": "notes.txt"}, "score": 0.75}
            ]
            
            with st.expander("View Sources"):
                for src in mock_sources:
                    st.markdown(f"**{src['metadata']['source']}** (Score: {src['score']})")
                    st.text(src['content'])
                    
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "sources": mock_sources
        })

if __name__ == "__main__":
    main()
