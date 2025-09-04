import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Circle
import io
import base64
from PIL import Image

def create_radar_chart(scores, candidate_name=""):
    """Create a radar chart for candidate analysis"""
    categories = ['Clarity', 'Keyword Match', 'Engagement', 'Confidence', 'Attention']
    num_vars = len(categories)
    
    # Compute angle for each category
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    scores += scores[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Plot data
    ax.plot(angles, scores, linewidth=2, linestyle='solid', label='Scores')
    ax.fill(angles, scores, alpha=0.25)
    
    # Add labels
    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True)
    
    # Add title
    plt.title(f'Candidate Analysis - {candidate_name}\nOverall Score: {np.mean(scores[:-1]):.2%}', 
              size=14, fontweight='bold')
    
    # Convert to base64 for Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    buf.close()
    plt.close()
    
    return f"data:image/png;base64,{img_str}"

def create_score_comparison_chart(candidates):
    """Create bar chart comparing multiple candidates"""
    names = [c['candidate_name'] for c in candidates]
    scores = [c['overall_score'] for c in candidates]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, scores, color=plt.cm.viridis(scores))
    
    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.2%}', ha='left', va='center', fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_xlabel('Overall Score')
    ax.set_title('Candidate Ranking Comparison', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    buf.close()
    plt.close()
    
    return f"data:image/png;base64,{img_str}"

def create_metric_bars(analysis_result):
    """Create horizontal bar chart for individual metrics"""
    metrics = {
        'Clarity': analysis_result['text_analysis']['clarity_score'],
        'Keyword Match': analysis_result['text_analysis']['keyword_score'],
        'Engagement': analysis_result['video_analysis']['engagement_score'],
        'Confidence': analysis_result['video_analysis']['confidence_score'],
        'Attention': analysis_result['video_analysis']['attention_score']
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.plasma(list(metrics.values()))
    bars = ax.barh(list(metrics.keys()), list(metrics.values()), color=colors)
    
    # Add value labels
    for bar, value in zip(bars, metrics.values()):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{value:.2%}', ha='left', va='center', fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_xlabel('Score')
    ax.set_title('Detailed Metrics Analysis', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    buf.close()
    plt.close()
    
    return f"data:image/png;base64,{img_str}"

def create_sentiment_gauge(sentiment_score):
    """Create a sentiment gauge chart"""
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Create gauge
    angles = np.linspace(0, 2 * np.pi, 100)
    ax.plot(angles, np.ones(100), color='lightgray', linewidth=2)
    
    # Fill based on sentiment
    if sentiment_score == 2:  # Positive
        color = 'green'
        angle = np.pi * 0.25
    elif sentiment_score == 0:  # Negative
        color = 'red' 
        angle = np.pi * 1.75
    else:  # Neutral
        color = 'orange'
        angle = np.pi
    
    ax.fill_between(angles, 0, 1, where=(angles <= angle), color=color, alpha=0.6)
    
    # Add labels
    ax.set_yticklabels([])
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    ax.set_xticklabels(['Positive', '', 'Negative', ''])
    ax.set_title('Sentiment Analysis', fontweight='bold')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    buf.close()
    plt.close()
    
    return f"data:image/png;base64,{img_str}"