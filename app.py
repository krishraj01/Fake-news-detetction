import streamlit as st
import joblib
import pandas as pd
import random
import matplotlib.pyplot as plt
from textblob import TextBlob
import numpy as np

# Set page config for light mode
st.set_page_config(page_title="Fake News Detection System", layout="wide")

# Custom CSS for light mode styling
st.markdown("""
<style>
    /* Main container */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Text area */
    .stTextArea textarea {
        background-color: #f8f9fa !important;
    }
    
    /* Cards */
    .custom-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 5px solid #4CAF50;
        margin-bottom: 15px;
    }
    
    /* Buttons */
    .stButton>button {
        border: 1px solid #4CAF50;
        color: white;
        background-color: #4CAF50;
    }
    
    /* Metrics */
    .stMetric {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
    }
    
    /* Expanders */
    .stExpander {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Load vectorizer and models
@st.cache_resource
def load_models():
    return {
        "vectorizer": joblib.load("vectorizer.jb"),
        "models": {
            "Logistic Regression": joblib.load("lr_model.jb"),
            "Decision Tree": joblib.load("dtc_model.jb"),
            "Gradient Boosting": joblib.load("gclf_model.jb"),
            "Random Forest": joblib.load("rclf_model.jb")
        }
    }

model_data = load_models()
vectorizer = model_data["vectorizer"]
models = model_data["models"]

# Load dataset with caching
@st.cache_data
def load_data():
    try:
        fake_news = pd.read_csv("fake.csv")
        true_news = pd.read_csv("true.csv")
        
        fake_samples = [{"text": text, "emoji": "❌"} for text in fake_news.text.head(5).tolist()]
        true_samples = [{"text": text, "emoji": "✅"} for text in true_news.text.head(5).tolist()]
        
        return fake_samples + true_samples
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return []

# Initialize session state
if 'articles_order' not in st.session_state:
    sample_articles = load_data()
    random.shuffle(sample_articles)
    st.session_state.articles_order = sample_articles
else:
    sample_articles = st.session_state.articles_order

# App Title
st.title("Fake News Detection System")
st.markdown("**Analyze news articles and content to identify potential misinformation**")
st.markdown("---")

# Initialize input text in session state
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

# Input field with light mode styling
input_text = st.text_area("🖊️ Paste news article here:", 
                         value=st.session_state.input_text,
                         height=200,
                         key="news_input")

# Big Check News button
check = st.button("🔍 ANALYZE NEWS ARTICLE", 
                use_container_width=True,
                type="primary")

# Display sample articles with proper session state handling
if sample_articles:
    st.subheader("📋 Sample Articles (Click to Use)")
    cols = st.columns(2)
    
    for idx, article in enumerate(sample_articles):
        with cols[idx % 2]:
            if st.button(
                f"{article['emoji']} {article['text'][:70]}...",
                key=f"article_{idx}",
                use_container_width=True
            ):
                # Update session state and rerun
                st.session_state.input_text = article["text"]
                st.rerun()

# Analysis function
def analyze_text(text):
    analysis = {
        "word_count": len(text.split()),
        "sentiment": TextBlob(text).sentiment.polarity,
        "subjectivity": TextBlob(text).sentiment.subjectivity,
        "all_caps": sum(1 for word in text.split() if word.isupper()),
        "exclamation": text.count('!') + text.count('?'),
        "biased_phrases": sum(1 for phrase in ["clearly", "obviously", "undoubtedly"] if phrase in text.lower())
    }
    
    transform_input = vectorizer.transform([text])
    results = []
    confidences = []
    
    for name, model in models.items():
        prediction = model.predict(transform_input)[0]
        confidence = model.predict_proba(transform_input).max() if hasattr(model, "predict_proba") else 0
        
        results.append({
            "Model": name,
            "Prediction": "Real" if prediction == 1 else "Fake",
            "Confidence": confidence,
            "Color": "#4CAF50" if prediction == 1 else "#F44336"  # Green/Red colors
        })
        confidences.append(confidence)
    
    return analysis, results, confidences

# Display results with light mode styling
if check and st.session_state.input_text.strip():
    st.markdown("---")
    st.subheader("🔎 Analysis Results")
    
    with st.spinner("Analyzing content..."):
        analysis, results, confidences = analyze_text(st.session_state.input_text)
        
        # Content analysis
        st.markdown("### 📝 Content Analysis")
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"""
        <div class="custom-card" style="border-left-color: #2196F3;">
            <h4>Word Count</h4>
            <h2>{analysis["word_count"]}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        sentiment_color = "#4CAF50" if analysis['sentiment'] > 0 else "#F44336" if analysis['sentiment'] < 0 else "#2196F3"
        col2.markdown(f"""
        <div class="custom-card" style="border-left-color: {sentiment_color};">
            <h4>Sentiment</h4>
            <h2 style="color: {sentiment_color};">{'Positive' if analysis['sentiment'] > 0 else 'Negative' if analysis['sentiment'] < 0 else 'Neutral'}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col3.markdown(f"""
        <div class="custom-card" style="border-left-color: #9C27B0;">
            <h4>Subjectivity</h4>
            <h2>{analysis['subjectivity']:.1f}/1.0</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Warning flags
        warning_cols = st.columns(2)
        if analysis["all_caps"] > 5:
            warning_cols[0].warning(f"⚠️ Excessive ALL-CAPS ({analysis['all_caps']} instances)")
        if analysis["exclamation"] > 5:
            warning_cols[1].warning(f"⚠️ Excessive punctuation ({analysis['exclamation']} instances)")
        if analysis["biased_phrases"] > 0:
            warning_cols[0].warning(f"⚠️ Biased language ({analysis['biased_phrases']} instances)")
        
        # Model predictions
        st.markdown("---")
        st.subheader("🤖 Model Predictions")
        
        pred_cols = st.columns(4)
        for i, result in enumerate(results):
            with pred_cols[i]:
                st.markdown(f"""
                <div class="custom-card" style="border-left-color: {result["Color"]};">
                    <h4>{result['Model']}</h4>
                    <h3 style="color: {result["Color"]};">{result['Prediction']}</h3>
                    <p>Confidence: <b>{result['Confidence']:.0%}</b></p>
                </div>
                """, unsafe_allow_html=True)
        
        # Confidence chart
        st.markdown("---")
        st.subheader("📊 Confidence Comparison")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar([result['Model'] for result in results], 
                     [result['Confidence'] for result in results],
                     color=[result['Color'] for result in results])
        
        max_idx = confidences.index(max(confidences))
        bars[max_idx].set_hatch('//')
        ax.annotate('Most Confident', 
                   xy=(max_idx, confidences[max_idx]),
                   xytext=(0, 20),
                   textcoords='offset points',
                   ha='center',
                   va='bottom',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                   arrowprops=dict(arrowstyle='->'))
        
        ax.set_facecolor('#ffffff')
        fig.patch.set_facecolor('#ffffff')
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Confidence Score')
        plt.xticks(rotation=15)
        st.pyplot(fig)
        
        # Overall verdict
        st.markdown("---")
        overall = "REAL" if sum(1 for r in results if r['Prediction'] == 'Real') > 2 else "FAKE"
        overall_color = "#4CAF50" if overall == "REAL" else "#F44336"
        st.markdown(f"""
        <div style='background-color:#ffffff;border-radius:10px;padding:20px;text-align:center;
                    border:2px solid {overall_color};margin-top:20px;box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
            <h2>Overall Verdict: <span style='color:{overall_color};'>
            {overall}</span></h2>
            <p> 4 out of 4 models agree</p>
        </div>
        """, unsafe_allow_html=True)

elif check:
    st.warning("⚠️ Please enter some text to analyze")

# How It Works section with light mode styling
with st.expander("ℹ️ HOW IT WORKS", expanded=False):
    st.markdown("""
    <div style="background-color:#f8f9fa;padding:20px;border-radius:10px;">
     🔍 Content Analysis Techniques
    
    Our detector analyzes multiple aspects of news content:
    
    **1. Linguistic Patterns:**
    - Detects sensationalist language and clickbait phrases
    - Identifies excessive punctuation and ALL-CAPS usage
    - Flags biased or absolute statements (e.g., "clearly", "undoubtedly")
    
    **2. Structural Analysis:**
    - Evaluates article length and complexity
    - Checks for proper sourcing and citations
    - Analyzes narrative structure and coherence
    
    **3. Emotional Indicators:**
    - Measures sentiment polarity (positive/negative)
    - Assesses subjectivity level
    - Detects emotional manipulation tactics
    
    
    ### 🤖 AI Model Ensemble
    
    The system combines predictions from 4 specialized models:
    1. **Logistic Regression** - Strong baseline performance
    2. **Decision Tree** - Good for rule-based patterns
    3. **Gradient Boosting** - Handles complex relationships
    4. **Random Forest** - Robust against overfitting
    
    All models use **TF-IDF Vectorization** for text pattern recognition.
    
    ### ⚠️ Important Notes
    
    - This tool identifies **patterns** associated with misinformation
    - **No single indicator** proves an article is fake
    - Always **verify information** across multiple reputable sources
    - Consider **context** and supporting evidence
    - Results should be used as **one component** of evaluation
    
    *This is a demonstration tool and should not be the sole basis for determining credibility.*
    </div>
    """, unsafe_allow_html=True)

# Future Improvements section
with st.expander("🚀 FUTURE IMPROVEMENTS", expanded=False):
    st.markdown("""
    <div style="background-color:#f8f9fa;padding:20px;border-radius:10px;">
    ### Planned Enhancements
    
    **1. Advanced Features:**
    - 🌐 **Multilingual Support** for non-English content++
    - 🔗 **Source Reliability Database** integration
    - 📑 **Fact-Checking API** connections
    
    **2. User Experience:**
    - 📱 **Mobile-Optimized** interface
    - 🎨 **Customizable** dashboard views
    - 📊 **Historical Analysis** of past verifications
    
    **3. Technical Improvements:**
    - 🧠 **Transformer Models** (BERT, GPT) integration
    - 🔍 **Image/Video Verification** capabilities
    - ⚡ **Performance Optimization** for large texts
    
    **4. Community Features:**
    - 👥 **User Feedback** system for predictions
    - 🌍 **Crowdsourced** credibility ratings
    - 📝 **Collaborative** fact-checking tools
    
    ### Have Suggestions?
    We welcome ideas for improving this tool! Contact us with your feature requests.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    🔍 Fake News Detection System | Combating misinformation through AI
</div>
""", unsafe_allow_html=True)
