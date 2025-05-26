Basic LLM Broth

1. Configure your environment and install packages

    python3 -m venv hlth              #create a virtual environment
    source hlth/bin/activate
    
    pip install --upgrade pip        
    pip install streamlit boto3       #install packages

2. Run the application

    streamlit run llm_broth.py
