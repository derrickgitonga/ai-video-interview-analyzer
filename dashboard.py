import streamlit as st
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from inference import InterviewAnalyzer
import tempfile
import os
from datetime import datetime
from utils.visualization import create_radar_chart, create_score_comparison_chart, create_metric_bars, create_sentiment_gauge

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .recommendation-box {
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class InterviewDashboard:
    def __init__(self):
        self.analyzer = InterviewAnalyzer()
        st.sidebar.title("AI Video Interview Analyzer")
        
    def run(self):
        st.title("🎯 AI Video Interview Analyzer")
        st.markdown("""
        **Transform your hiring process with AI-powered video interview analysis.**
        Get instant insights on candidate responses, facial engagement, and overall fit.
        """)
        
        # Sidebar options
        analysis_type = st.sidebar.radio(
            "Choose Analysis Type:",
            ["Single Candidate", "Batch Analysis", "Compare Candidates"]
        )
        
        if analysis_type == "Single Candidate":
            self.single_candidate_analysis()
        elif analysis_type == "Batch Analysis":
            self.batch_analysis()
        else:
            self.compare_candidates()
            
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.info(
            "💡 This AI system analyzes video interviews using NLP and computer vision "
            "to help HR managers make data-driven hiring decisions."
        )
    
    def single_candidate_analysis(self):
        st.header("📝 Single Candidate Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            candidate_name = st.text_input("Candidate Name", "John Doe")
            job_description = st.text_area(
                "Job Description (Optional)",
                "Enter the job description for keyword matching...",
                height=150
            )
            
            video_file = st.file_uploader("Upload Interview Video", type=['mp4', 'mov', 'avi'])
            transcript = st.text_area("Interview Transcript", "Paste the candidate's responses...", height=200)
        
        with col2:
            if st.button("🚀 Analyze Candidate", use_container_width=True):
                if video_file and transcript:
                    with st.spinner("Analyzing candidate..."):
                        # Save uploaded video to temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                            tmp_file.write(video_file.getvalue())
                            video_path = tmp_file.name
                        
                        try:
                            result = self.analyzer.analyze_candidate(
                                video_path, transcript, job_description, candidate_name
                            )
                            self.display_results(result)
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")
                        finally:
                            os.unlink(video_path)
                else:
                    st.warning("Please upload both video and transcript.")
    
    def display_results(self, result):
        st.success("✅ Analysis Complete!")
        
        # Overall score and recommendation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Score", f"{result['overall_score']:.2%}")
        
        with col2:
            sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            sentiment = sentiment_map.get(result['text_analysis']['sentiment'], "Unknown")
            st.metric("Sentiment", sentiment)
        
        with col3:
            rec_color = {
                "Strongly Recommend": "green",
                "Recommend": "blue", 
                "Consider": "orange",
                "Not Recommended": "red"
            }[result['recommendation']]
            
            st.markdown(
                f"<div class='recommendation-box' style='border-color: {rec_color}; color: {rec_color}'>"
                f"{result['recommendation']}</div>",
                unsafe_allow_html=True
            )
        
        # Add visualizations section
        st.subheader("📊 Visual Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Radar chart
            radar_scores = [
                result['text_analysis']['clarity_score'],
                result['text_analysis']['keyword_score'], 
                result['video_analysis']['engagement_score'],
                result['video_analysis']['confidence_score'],
                result['video_analysis']['attention_score']
            ]
            radar_img = create_radar_chart(radar_scores, result['candidate_name'])
            st.image(radar_img, caption="Skills Radar Chart", use_column_width=True)
        
        with col2:
            # Sentiment gauge
            sentiment_img = create_sentiment_gauge(result['text_analysis']['sentiment'])
            st.image(sentiment_img, caption="Sentiment Analysis", use_column_width=True)
        
        # Metrics bar chart
        metrics_img = create_metric_bars(result)
        st.image(metrics_img, caption="Detailed Metrics", use_column_width=True)
        
        # Detailed metrics
        st.subheader("📋 Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Text Analysis")
            text_analysis = result['text_analysis']
            
            st.progress(text_analysis['clarity_score'])
            st.caption(f"Clarity Score: {text_analysis['clarity_score']:.2%}")
            
            st.progress(text_analysis['keyword_score'])
            st.caption(f"Keyword Match: {text_analysis['keyword_score']:.2%}")
            
            if text_analysis['keyword_matches']:
                st.write("**Matched Keywords:**", ", ".join(text_analysis['keyword_matches']))
            
            st.write("**Clarity Metrics:**")
            st.json(text_analysis['clarity_metrics'])
        
        with col2:
            st.markdown("### 📹 Video Analysis")
            video_analysis = result['video_analysis']
            
            st.progress(video_analysis['engagement_score'])
            st.caption(f"Engagement Score: {video_analysis['engagement_score']:.2%}")
            
            st.progress(video_analysis['confidence_score'])
            st.caption(f"Confidence Score: {video_analysis['confidence_score']:.2%}")
            
            st.progress(video_analysis['attention_score'])
            st.caption(f"Attention Score: {video_analysis['attention_score']:.2%}")
            
            st.write("**Facial Analysis:**")
            st.json(video_analysis['facial_analysis'])
        
        # Download results
        result_json = json.dumps(result, indent=2)
        st.download_button(
            label="📥 Download Analysis Report",
            data=result_json,
            file_name=f"interview_analysis_{result['candidate_name']}_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    def batch_analysis(self):
        st.header("📊 Batch Candidate Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload CSV with candidate data (columns: name, video_path, transcript)",
            type=['csv']
        )
        
        job_description = st.text_area(
            "Job Description for Batch Analysis",
            "Enter the job description for keyword matching...",
            height=150
        )
        
        if uploaded_file and st.button("📈 Analyze Batch", use_container_width=True):
            df = pd.read_csv(uploaded_file)
            candidates = df.to_dict('records')
            
            with st.spinner(f"Analyzing {len(candidates)} candidates..."):
                results = self.analyzer.analyze_multiple_candidates(candidates, job_description)
                self.display_batch_results(results)
    
    def display_batch_results(self, results):
        st.success(f"✅ Batch analysis complete! Analyzed {len(results)} candidates.")
        
        # Create results dataframe
        df_data = []
        for result in results:
            df_data.append({
                'Rank': result['rank'],
                'Candidate': result['candidate_name'],
                'Overall Score': result['overall_score'],
                'Clarity Score': result['text_analysis']['clarity_score'],
                'Keyword Score': result['text_analysis']['keyword_score'],
                'Engagement Score': result['video_analysis']['engagement_score'],
                'Confidence Score': result['video_analysis']['confidence_score'],
                'Recommendation': result['recommendation']
            })
        
        df = pd.DataFrame(df_data)
        
        # Display results table
        st.dataframe(df.style.highlight_max(subset=['Overall Score'], color='lightgreen'))
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Score Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df['Overall Score'], bins=10, ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.subheader("🎯 Score Comparison")
            comparison_img = create_score_comparison_chart(results)
            st.image(comparison_img, caption="Candidate Comparison", use_column_width=True)
        
        # Download batch results
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Batch Results (CSV)",
            data=csv,
            file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    def compare_candidates(self):
        st.header("🔍 Compare Multiple Candidates")
        st.info("Upload multiple candidate analyses to compare their scores.")
        
        # Implementation for comparison view
        uploaded_files = st.file_uploader(
            "Upload analysis JSON files",
            type=['json'],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("📊 Compare Candidates", use_container_width=True):
            candidates = []
            for uploaded_file in uploaded_files:
                try:
                    result = json.load(uploaded_file)
                    candidates.append(result)
                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {e}")
            
            if candidates:
                self.display_batch_results(candidates)

def main():
    dashboard = InterviewDashboard()
    dashboard.run()

if __name__ == '__main__':
    main()