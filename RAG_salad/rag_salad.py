import streamlit as st
import boto3
import os
import requests
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.llms.bedrock import Bedrock

# Initialize AWS clients
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

# Set up knowledge base
KNOWLEDGE_DIR = "medical_kb"
os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

# Free medical knowledge sources
MEDICAL_SOURCES = {
    "CDC Guidelines": {
        "url": "https://www.cdc.gov/flu/professionals/index.htm",
        "description": "Influenza treatment guidelines"
    },
    "NIH Health Topics": {
        "url": "https://health.nih.gov/topics",
        "description": "NIH condition overviews"
    },
    "WHO Disease Factsheets": {
        "url": "https://www.who.int/news-room/fact-sheets",
        "description": "WHO global health facts"
    }
}

# --- Streamlit UI ---
st.set_page_config(page_title="Medical RAG (Bedrock)", page_icon="🏥")
st.title("RAG Salad with AWS Bedrock")
st.caption("Evidence-based answers using Claude & medical knowledge bases")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Bedrock Model",
        ["anthropic.claude-3-sonnet-20240229-v1:0", "anthropic.claude-v2"]
    )
    
    selected_sources = st.multiselect(
        "Knowledge Sources",
        options=list(MEDICAL_SOURCES.keys()),
        default=["CDC Guidelines"]
    )
    
    if st.button("Update Knowledge Base"):
        with st.spinner("Downloading latest medical content..."):
            for source in selected_sources:
                url = MEDICAL_SOURCES[source]["url"]
                response = requests.get(url)
                with open(f"{KNOWLEDGE_DIR}/{source}.html", "w") as f:
                    f.write(response.text)
            st.success(f"Updated {len(selected_sources)} sources!")

# Initialize RAG system with Bedrock
@st.cache_resource
def load_rag_system():
    # Initialize Bedrock LLM
    llm = Bedrock(
        model=selected_model,
        client=bedrock_runtime,
        temperature=0.3
    )
    
    # Initialize Bedrock embeddings
    embed_model = BedrockEmbedding(
        model_name="amazon.titan-embed-text-v1",
        client=bedrock_runtime
    )
    
    # Configure settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    
    # Load documents
    documents = SimpleDirectoryReader(KNOWLEDGE_DIR).load_data()
    
    # Create index with updated settings
    return VectorStoreIndex.from_documents(documents)

# Main interface
if st.button("Initialize RAG System"):
    with st.spinner("Building knowledge index..."):
        index = load_rag_system()
        st.session_state.query_engine = index.as_query_engine()
        st.success("Ready! Ask medical questions below.")

if 'query_engine' in st.session_state:
    question = st.text_input("Ask a medical question:", 
                           placeholder="e.g. What's the recommended flu treatment for pregnant women?")
    
    if question:
        with st.spinner("Consulting knowledge bases..."):
            response = st.session_state.query_engine.query(question)
            
            st.subheader("Evidence-Based Answer")
            st.markdown(response.response)
            
            with st.expander("🔍 See Sources Used"):
                for i, source in enumerate(response.source_nodes[:3]):  # Show top 3
                    st.caption(f"Source {i+1}:")
                    st.code(source.node.get_content()[:500] + "...", language='text')
else:
    st.warning("Please initialize the RAG system first")

# --- Info Section ---
st.sidebar.divider()
st.sidebar.markdown("""
**How This Works:**
1. Select knowledge sources
2. Click "Update Knowledge Base"
3. Initialize RAG system
4. Ask medical questions

**AWS Components Used:**
- Bedrock (Claude 3 Sonnet or Claude 2)
- Titan Embeddings
- Local vector store

**Sample Questions:**
- "CDC flu vaccine guidelines 2024"
- "NIH recommendations for diabetes management"
- "WHO malaria prevention strategies"
""")