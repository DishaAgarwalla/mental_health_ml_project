import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page config
st.set_page_config(
    page_title="Anxiety Detection AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .normal-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .anxiety-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .confidence-meter {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        border: none;
    }
    .word-cloud {
        padding: 1rem;
        background-color: #f1f1f1;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# API endpoint
API_URL = "http://127.0.0.1:8000"

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_prediction(text):
    """Get prediction from API with better error handling"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            # Try to get error details
            try:
                error_detail = response.json()
                st.error(f"API Error: {error_detail}")
            except:
                st.error(f"API Error: Status {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure FastAPI is running.")
        return None
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def get_history():
    """Get prediction history"""
    try:
        response = requests.get(f"{API_URL}/history?limit=50", timeout=3)
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except:
        return []

def get_stats():
    """Get statistics"""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=3)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/anxiety.png", width=80)
    st.title("🧠 Anxiety Detection")
    st.markdown("---")
    
    # API Status
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")
        st.info("Run: `uvicorn src.api:app --reload`")
    
    st.markdown("---")
    st.subheader("About")
    st.info(
        "This AI model analyzes text to detect potential "
        "anxiety indicators. It's trained on a dataset of "
        "anxiety-related and normal statements."
    )
    
    st.markdown("---")
    st.subheader("How It Works")
    st.write(
        """
        1. Enter a statement
        2. AI analyzes the text
        3. Get prediction with confidence score
        4. Results are logged for analytics
        """
    )

# Main content
st.markdown("<h1 class='main-header'>🧠 Anxiety Detection System</h1>", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 Analyze Text", "📊 Analytics", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text input
        text_input = st.text_area(
            "Enter a statement to analyze:",
            height=150,
            placeholder="e.g., I've been feeling really anxious lately and can't sleep...",
            help="Type or paste a statement to analyze for anxiety indicators"
        )
        
        # Analyze button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            analyze_button = st.button("🔍 Analyze", use_container_width=True)
        
        # Example statements
        with st.expander("📋 Try these examples"):
            examples = [
                "I feel great today! Everything is going well.",
                "I'm so anxious about my presentation tomorrow.",
                "Just finished my project, feeling accomplished!",
                "My heart is racing and I can't stop worrying.",
                "Looking forward to the weekend with friends!",
                "I'm restless and can't sleep at night."
            ]
            for ex in examples:
                if st.button(f"📝 {ex[:30]}...", key=ex):
                    text_input = ex
                    st.rerun()
    
    with col2:
        st.markdown("### 📊 Model Info")
        if api_healthy:
            try:
                info = requests.get(f"{API_URL}/model-info", timeout=2).json()
                st.write(f"**Model:** {info.get('model_type', 'Unknown')}")
                st.write(f"**Vectorizer:** {info.get('vectorizer', 'Unknown')}")
                st.write(f"**Features:** {info.get('total_features', 0):,}")
            except:
                st.write("Model info unavailable")
        
        st.markdown("### 📈 Quick Stats")
        stats = get_stats()
        if stats:
            st.metric("Total Predictions", stats.get('total_predictions', 0))
            anxiety_count = stats.get('by_class', {}).get('anxiety', 0)
            st.metric("Anxiety Detected", anxiety_count)
        else:
            st.metric("Total Predictions", "N/A")
    
    # Prediction results
    if analyze_button and text_input:
        with st.spinner("Analyzing..."):
            result = get_prediction(text_input)
            
        if result:
            # Display result
            st.markdown("---")
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if result['prediction'] == 1:
                    st.markdown(
                        "<div class='prediction-box anxiety-box'>"
                        "<h2>⚠️ ANXIETY DETECTED</h2>"
                        f"<p>{result['message']}</p>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        "<div class='prediction-box normal-box'>"
                        "<h2>✅ NORMAL STATEMENT</h2>"
                        f"<p>{result['message']}</p>"
                        "</div>",
                        unsafe_allow_html=True
                    )
            
            with col_res2:
                st.markdown("<div class='confidence-meter'>", unsafe_allow_html=True)
                st.markdown("### Confidence Score")
                
                # Create gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['confidence'] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Confidence %"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#4CAF50" if result['prediction'] == 0 else "#f44336"},
                        'steps': [
                            {'range': [0, 50], 'color': "#d4edda"},
                            {'range': [50, 75], 'color': "#fff3cd"},
                            {'range': [75, 100], 'color': "#f8d7da"}
                        ]
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Show analyzed text
            st.markdown("### 📝 Analyzed Text")
            st.info(text_input)
            
            st.success("✅ Analysis complete!")
        else:
            st.error("Failed to get prediction. Please check if API is running.")

with tab2:
    st.markdown("## 📊 Analytics Dashboard")
    
    # Get history
    history = get_history()
    stats = get_stats()
    
    if history:
        # Convert to DataFrame
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Total Analyses", len(df))
        with col_m2:
            anxiety_count = len(df[df['prediction'] == 'anxiety'])
            st.metric("Anxiety Detected", anxiety_count)
        with col_m3:
            normal_count = len(df[df['prediction'] == 'normal'])
            st.metric("Normal Statements", normal_count)
        with col_m4:
            anxiety_pct = (anxiety_count/len(df)*100) if len(df) > 0 else 0
            st.metric("Anxiety Rate", f"{anxiety_pct:.1f}%")
        
        # Charts
        col_ch1, col_ch2 = st.columns(2)
        
        with col_ch1:
            # Pie chart
            fig_pie = px.pie(
                df, 
                names='prediction',
                title='Prediction Distribution',
                color='prediction',
                color_discrete_map={'normal': '#4CAF50', 'anxiety': '#f44336'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_ch2:
            # Timeline
            daily_counts = df.groupby('date').size().reset_index(name='count')
            fig_line = px.line(
                daily_counts, 
                x='date', 
                y='count',
                title='Analyses Over Time',
                markers=True
            )
            st.plotly_chart(fig_line, use_container_width=True)
        
        # Recent predictions table
        st.markdown("### 📋 Recent Analyses")
        display_df = df[['timestamp', 'text', 'prediction', 'confidence']].head(10)
        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.2%}")
        display_df['text'] = display_df['text'].str[:50] + '...'
        display_df.columns = ['Timestamp', 'Text', 'Prediction', 'Confidence']
        st.dataframe(display_df, use_container_width=True)
        
    else:
        st.info("No prediction history yet. Try analyzing some text!")

with tab3:
    st.markdown("## ℹ️ About This System")
    
    st.markdown("""
    ### Purpose
    This AI-powered system analyzes text to detect potential anxiety indicators. 
    It's designed to help identify language patterns associated with anxiety.
    
    ### Dataset Information
    The model is trained on a dataset containing:
    - **Anxiety statements**: ~6,600 samples of text expressing anxiety, worry, restlessness
    - **Normal statements**: ~900 samples of regular conversations about daily life
    
    ### How It Works
    1. **Text Input**: You provide a statement
    2. **Preprocessing**: Text is cleaned and normalized
    3. **Feature Extraction**: TF-IDF vectorization converts text to numerical features
    4. **Classification**: Logistic Regression model predicts if text indicates anxiety
    5. **Confidence Score**: Model provides confidence level for the prediction
    
    ### Model Performance
    - **Algorithm**: Logistic Regression with TF-IDF features
    - **Features**: N-grams (1-2 words) with up to 10,000 features
    - **Class balancing**: Weighted to handle imbalanced classes
    
    ### ⚠️ Important Disclaimer
    **This is NOT a medical diagnosis tool.** The predictions should be used for:
    - Educational purposes
    - Research and development
    - Early screening and awareness
    
    Always consult with qualified mental health professionals for proper diagnosis and treatment.
    
    ### Crisis Resources
    If you're experiencing mental health issues, please reach out:
    - **National Suicide Prevention Lifeline**: 988
    - **Crisis Text Line**: Text HOME to 741741
    - **SAMHSA Helpline**: 1-800-662-4357
    """)
    
    st.markdown("---")
    st.markdown("**Version:** 1.0.0 | **Last Updated:** 2024")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "⚠️ This tool is for educational purposes only. Not a substitute for professional medical advice."
    "</div>",
    unsafe_allow_html=True
)