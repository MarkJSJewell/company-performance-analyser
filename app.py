import streamlit as st
import pdfplumber
import json
import re
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Financial Analyzer (Gemini)", layout="wide")

class EarningsAnalyzer:
    def __init__(self, api_key):
        # We use Gemini 1.5 Flash. If this fails, try "gemini-pro"
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=api_key,
            temperature=0
        )

    def extract_text(self, uploaded_file):
        """Reads the ENTIRE PDF."""
        text = ""
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    extract = page.extract_text()
                    if extract: text += extract
            return text
        except Exception as e:
            st.error(f"❌ Error reading PDF: {e}")
            return None

    def clean_json(self, raw_output):
        # Extract JSON from markdown blocks if present
        text = raw_output.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(text)
        except:
            return None

    def analyze_full_report(self, text, q_name):
        """
        Gemini has a 1M token window, so we can send the WHOLE PDF 
        and ask for everything in one shot.
        """
        prompt = f"""
        You are a financial analyst. Extract data from this {q_name} report.
        
        CRITICAL RULES:
        1. Ignore "Year Ended" columns. ONLY use "Three Months Ended" (Quarterly).
        2. Return ONLY a valid JSON object. No intro text.
        
        REQUIRED JSON STRUCTURE:
        {{
            "quarterly_revenue_bn": 0.0,
            "eps": 0.0,
            "net_interest_income_millions": 0,
            "dividend_per_share": 0.0,
            "assets_under_supervision_bn": 0.0,
            "average_daily_var_total": 0
        }}
        
        REPORT TEXT:
        {text}
        """
        return self.llm.invoke([HumanMessage(content=prompt)]).content

    def generate_summary(self, data):
        context = json.dumps(data, indent=2)
        prompt = f"""
        Write a professional executive summary for these quarterly results.
        Focus on the Revenue Trend and Asset Growth.
        
        Data: {context}
        """
        return self.llm.invoke([HumanMessage(content=prompt)]).content

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    google_api_key = st.text_input("Google API Key", type="password")
    st.caption("Get a free key at aistudio.google.com")
    uploaded_files = st.file_uploader("Upload Reports (Q1-Q4)", type="pdf", accept_multiple_files=True)

# --- 3. MAIN LOGIC ---
st.title("🚀 Financial Analyzer (Gemini Edition)")

if st.button("Run Analysis", type="primary"):
    if not google_api_key or not uploaded_files:
        st.error("Please provide a Google API Key and Upload Files.")
    else:
        analyzer = EarningsAnalyzer(google_api_key)
        results = {}
        
        progress_bar = st.progress(0, text="Starting analysis...")
        
        for i, file in enumerate(uploaded_files):
            q_name = file.name.replace(".pdf", "")
            progress_bar.progress((i / len(uploaded_files)), text=f"Analyzing {q_name}...")
            
            text = analyzer.extract_text(file)
            
            if text:
                # One-Shot Extraction
                try:
                    raw_json = analyzer.analyze_full_report(text, q_name)
                    data = analyzer.clean_json(raw_json)
                    
                    if data:
                        results[q_name] = data
                    else:
                        st.error(f"Failed to parse data for {q_name}")
                except Exception as e:
                    st.error(f"API Error: {e}")
        
        progress_bar.progress(1.0, text="Done!")

        # --- 4. DASHBOARD ---
        if results:
            quarters = sorted(results.keys())
            
            # Metrics
            st.subheader("Key Metrics")
            cols = st.columns(len(quarters))
            for idx, q in enumerate(quarters):
                data = results[q]
                cols[idx].metric(label=q, value=f"${data.get('quarterly_revenue_bn', 0)}B", delta=f"EPS: ${data.get('eps', 0)}")
            
            # Charts
            st.divider()
            c1, c2 = st.columns(2)
            
            # Revenue Chart
            revs = [results[q].get('quarterly_revenue_bn', 0) for q in quarters]
            fig, ax = plt.subplots()
            ax.bar(quarters, revs, color='#4285F4') # Google Blue
            ax.set_title("Quarterly Revenue ($bn)")
            c1.pyplot(fig)
            
            # AUS Chart
            aus = [results[q].get('assets_under_supervision_bn', 0) for q in quarters]
            fig2, ax2 = plt.subplots()
            ax2.bar(quarters, aus, color='#34A853') # Google Green
            ax2.set_title("Assets Under Supervision ($bn)")
            c2.pyplot(fig2)
            
            # Summary
            st.divider()
            st.subheader("📝 Executive Summary")
            with st.spinner("Generating insights..."):
                summary = analyzer.generate_summary(results)
                st.markdown(summary)
