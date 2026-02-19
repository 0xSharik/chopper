"""
Additional visual components for demo polish.
Risk gauges, traffic lights, hero metrics.
"""
import streamlit as st
import plotly.graph_objects as go

def create_risk_gauge(risk_score, title="Development Risk"):
    """
    Create circular gauge chart for risk visualization.
    
    Args:
        risk_score: Risk score (0-100)
        title: Gauge title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue", 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': 'lightgreen'},
                {'range': [40, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def render_hero_metrics(druglikeness_score, risk_score):
    """
    Render large hero metrics at top of page.
    
    Args:
        druglikeness_score: Drug-likeness score (0-100)
        risk_score: Overall risk score (0-100)
    """
    col1, col2 = st.columns(2)
    
    with col1:
        # Drug-likeness
        dl_color = "#28a745" if druglikeness_score > 70 else "#ffc107" if druglikeness_score > 40 else "#dc3545"
        dl_icon = "✅" if druglikeness_score > 70 else "⚠️" if druglikeness_score > 40 else "❌"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {dl_color}22 0%, {dl_color}44 100%); 
                    padding: 2rem; border-radius: 1rem; border-left: 5px solid {dl_color};">
            <div style="font-size: 0.9rem; color: #666; font-weight: 600;">DRUG-LIKENESS SCORE</div>
            <div style="font-size: 3rem; font-weight: bold; color: {dl_color}; margin: 0.5rem 0;">
                {druglikeness_score:.1f} {dl_icon}
            </div>
            <div style="font-size: 0.8rem; color: #888;">out of 100</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Risk score (inverted - lower is better)
        risk_color = "#28a745" if risk_score < 40 else "#ffc107" if risk_score < 70 else "#dc3545"
        risk_icon = "✅" if risk_score < 40 else "⚠️" if risk_score < 70 else "❌"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {risk_color}22 0%, {risk_color}44 100%); 
                    padding: 2rem; border-radius: 1rem; border-left: 5px solid {risk_color};">
            <div style="font-size: 0.9rem; color: #666; font-weight: 600;">DEVELOPMENT RISK</div>
            <div style="font-size: 3rem; font-weight: bold; color: {risk_color}; margin: 0.5rem 0;">
                {risk_score:.1f} {risk_icon}
            </div>
            <div style="font-size: 0.8rem; color: #888;">lower is better</div>
        </div>
        """, unsafe_allow_html=True)

def render_traffic_light_card(property_name, value, optimal_range, unit=""):
    """
    Render color-coded property card.
    
    Args:
        property_name: Name of property
        value: Current value
        optimal_range: Tuple of (min, max) for optimal
        unit: Unit string (optional)
    """
    min_opt, max_opt = optimal_range
    
    # Determine color and status
    if min_opt <= value <= max_opt:
        color = "#d4edda"
        border_color = "#28a745"
        icon = "✅"
        status = "Optimal"
    elif (min_opt * 0.7 <= value < min_opt) or (max_opt < value <= max_opt * 1.3):
        color = "#fff3cd"
        border_color = "#ffc107"
        icon = "⚠️"
        status = "Borderline"
    else:
        color = "#f8d7da"
        border_color = "#dc3545"
        icon = "❌"
        status = "Risk"
    
    st.markdown(f"""
    <div style="background-color: {color}; 
                padding: 1rem; 
                border-radius: 0.5rem; 
                border-left: 4px solid {border_color};
                margin: 0.5rem 0;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <div style="font-weight: 600; color: #333;">{icon} {property_name}</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: {border_color};">
                    {value:.2f}{unit}
                </div>
            </div>
            <div style="text-align: right; color: #666;">
                <div style="font-size: 0.8rem;">{status}</div>
                <div style="font-size: 0.7rem;">Target: {min_opt}-{max_opt}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_executive_summary_box(summary_text):
    """
    Render executive summary in prominent box.
    
    Args:
        summary_text: Summary text to display
    """
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; 
                border-radius: 1rem; 
                color: white;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                margin: 1rem 0;">
        <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem;">
            📋 Executive Summary
        </div>
        <div style="font-size: 1rem; line-height: 1.6; opacity: 0.95;">
            {summary_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_ranking_badge(rank):
    """
    Render ranking badge with medal emoji.
    
    Args:
        rank: Ranking position (1, 2, 3, etc.)
        
    Returns:
        HTML string for badge
    """
    badges = {
        1: ("🥇", "Best Candidate", "#FFD700", "#FFF8DC"),
        2: ("🥈", "Second Place", "#C0C0C0", "#F5F5F5"),
        3: ("🥉", "Third Place", "#CD7F32", "#FFF8DC")
    }
    
    if rank in badges:
        emoji, text, border_color, bg_color = badges[rank]
        return f"""
        <div style="background-color: {bg_color}; 
                    border: 3px solid {border_color}; 
                    padding: 0.5rem 1rem; 
                    border-radius: 0.5rem; 
                    display: inline-block;
                    font-weight: bold;
                    margin: 0.5rem 0;">
            {emoji} {text}
        </div>
        """
    else:
        return f"""
        <div style="background-color: #f8f9fa; 
                    border: 2px solid #dee2e6; 
                    padding: 0.5rem 1rem; 
                    border-radius: 0.5rem; 
                    display: inline-block;
                    margin: 0.5rem 0;">
            #{rank}
        </div>
        """
