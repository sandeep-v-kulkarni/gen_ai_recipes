import streamlit as st
import boto3
import json
from functools import lru_cache

# Initialize Bedrock clients
bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

# Cache configurations
@lru_cache(maxsize=100)
def get_medical_answer(question):
    """Standard medical Q&A with Claude"""
    prompt = f"\n\nHuman: You are a medical assistant. Rules:\n1. Provide factual information\n2. Never diagnose\n3. Include: 'Consult your doctor'\n\nQuestion: {question}\n\nAssistant:"
    
    try:
        response = bedrock.invoke_model(
            modelId="anthropic.claude-v2",
            body=json.dumps({
                "prompt": prompt,
                "max_tokens_to_sample": 400,
                "temperature": 0.3
            })
        )
        return json.loads(response['body'].read())['completion']
    except Exception as e:
        return f"Error: {str(e)}"

@lru_cache(maxsize=50)
def get_reasoned_answer(inputs, model="claude"):
    """Reason across multiple inputs using selected model
    
    This function leverages advanced LLMs (Claude 3 Sonnet or Deepseek) to perform 
    complex medical reasoning across multiple clinical inputs. Unlike simple Q&A, 
    the reasoning model identifies relationships between separate medical facts, 
    detects potential interactions or conflicts, and synthesizes an integrated 
    analysis that considers all inputs holistically. This approach mimics clinical 
    decision-making where physicians must connect disparate pieces of information 
    to form a comprehensive understanding of a patient's condition."""
    reasoning_prompt = f"""Analyze these medical inputs and provide synthesized insights:
    
    Input 1: {inputs[0]}
    Input 2: {inputs[1]}
    {f"Input 3: {inputs[2]}" if len(inputs) > 2 else ""}

    Perform step-by-step medical reasoning:
    1. Identify key facts from each input
    2. Find connections between them
    3. Highlight potential interactions/conflicts
    4. Provide integrated advice (with disclaimers)"""
    
    try:
        if model == "claude":
            # Use Claude 3 Sonnet
            response = bedrock.invoke_model(
                modelId="anthropic.claude-3-sonnet-20240229-v1:0",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [{"role": "user", "content": reasoning_prompt}],
                    "max_tokens": 1000,
                    "temperature": 0.1
                })
            )
            return json.loads(response['body'].read())['content'][0]['text']
        else:
            # Use Deepseek
            response = bedrock.invoke_model(
                modelId="us.deepseek.r1-v1:0",  # Correct Deepseek model ID
                body=json.dumps({
                    "messages": [{"role": "user", "content": reasoning_prompt}],
                    "max_tokens": 1000,
                    "temperature": 0.1
                })
            )
            return json.loads(response['body'].read())['choices'][0]['message']['content']
    except Exception as e:
        return f"Reasoning Error: {str(e)}"

# Predefined demo cases
DEMO_CASES = {
    "Statin Myopathy": {
        "inputs": [
            "I take 40mg atorvastatin daily",
            "Experiencing muscle pain in thighs",
            "My last CK enzyme was 300 U/L"
        ],
        "description": "Analyze possible statin side effects with lab results"
    },
    "Diabetic CKD": {
        "inputs": [
            "Type 2 diabetic on metformin",
            "New diagnosis of CKD stage 3",
            "Recent UTI treated with ciprofloxacin"
        ],
        "description": "Evaluate medication safety in renal impairment"
    },
    "Warfarin Interaction": {
        "inputs": [
            "Taking warfarin 5mg daily",
            "Started fish oil supplements",
            "INR last week was 2.1, today 3.8"
        ],
        "description": "Identify supplement-drug interaction"
    },
    "Shift Work Diabetes": {
        "inputs": [
            "A1c increased from 6.2% to 7.1% in 6 months",
            "Started night shift work 8 months ago",
            "Fasting glucose 150 mg/dL"
        ],
        "description": "Assess lifestyle impact on glycemic control"
    },
    "PE Emergency": {
        "inputs": [
            "History of DVT on rivaroxaban",
            "Sudden shortness of breath",
            "Left leg swelling"
        ],
        "description": "Recognize red flags for pulmonary embolism"
    }
}

# Streamlit UI
st.set_page_config(page_title="Medical Reasoning", page_icon="🧠")
st.title("Medical Reasoning Assistant")

# Tab interface
tab1, tab2 = st.tabs(["Standard Q&A", "Advanced Reasoning"])

with tab1:
    st.text_input("Ask a medical question:", key="std_question")
    if st.session_state.std_question:
        answer = get_medical_answer(st.session_state.std_question)
        st.markdown(f"**Answer:** {answer}")

with tab2:
    st.subheader("Clinical Reasoning Engine")
    
    # Model selector
    selected_model = st.radio(
        "Choose reasoning model:",
        ["Claude Sonnet", "Deepseek"],
        horizontal=True
    )
    
    # Demo case selector
    selected_case = st.selectbox(
        "Try a demo case:",
        options=list(DEMO_CASES.keys()),
        format_func=lambda x: f"{x}: {DEMO_CASES[x]['description']}",
        index=None
    )
    
    # Input fields (will auto-populate from demo cases)
    input_fields = []
    for i in range(3):
        default_val = DEMO_CASES[selected_case]["inputs"][i] if selected_case else ""
        input_fields.append(
            st.text_input(
                f"Input {i+1} {'(Optional)' if i == 2 else ''}",
                value=default_val,
                key=f"input_{i}"
            )
        )
    
    if st.button("Analyze Relationships"):
        inputs = [i for i in input_fields if i.strip()]
        
        if len(inputs) < 2:
            st.warning("Please provide at least 2 inputs")
        else:
            with st.spinner("Analyzing connections..."):
                # Display individual understandings
                st.subheader("Individual Analysis")
                
                # Display each input and its full analysis in separate sections
                for i, inp in enumerate(inputs):
                    with st.expander(f"Input {i+1}: {inp}", expanded=True):
                        answer = get_medical_answer(inp)
                        st.markdown(answer)
                
                # Show reasoned analysis
                st.subheader("Clinical Synthesis")
                model_param = "claude" if selected_model == "Claude Sonnet" else "deepseek"
                
                with st.spinner(f"Analyzing with {selected_model}..."):
                    reasoned_answer = get_reasoned_answer(tuple(inputs), model=model_param)
                
                with st.expander(f"Step-by-Step Reasoning ({selected_model})"):
                    st.markdown(reasoned_answer)
                
                # Emergency alert if detected
                if any(term in reasoned_answer.lower() for term in ["emergency", "911", "immediate"]):
                    st.error("🚨 This may be a medical emergency - seek immediate care!")

# Explanation section
with st.expander("How This Works"):
    st.markdown("""
    **The reasoning engine:**
    1. Extracts key facts from each input
    2. Identifies temporal/clinical relationships
    3. Checks for dangerous interactions
    4. Provides weighted recommendations
    
    **Demo cases illustrate:**
    - Medication side effects
    - Disease-drug interactions
    - Lifestyle impacts on chronic conditions
    - Emergency red flag recognition
    """)

# Debug info
if st.checkbox("Show technical details"):
    st.write(f"Standard cache: {get_medical_answer.cache_info()}")
    st.write(f"Reasoning cache: {get_reasoned_answer.cache_info()}")