# app/app.py
"""
Professional Retail Forecasting Platform for Corporación Favorita
==========================================

Features:
- Centralized data loading from external mounted disk
- Dynamic filters with caching
- Professional UI with responsive design
- Model artifact loading systemcd ..

- Session management across pages
"""

import streamlit as st  # type: ignore
from bootstrap import *  # noqa: F403

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from app.pages import Forecast  # noqa: E402


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Retail Forecasting Platform",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """
        ## Retail Forecasting Platform
        Advanced time series forecasting for Corporación Favorita retail demand in Guayas region.

        **Features:**
        - Multi-model forecasting (ARIMA, SARIMA, Prophet, ETS, ML models, LSTM)
        - Model artifact loading (no training in UI)
        - Professional dashboard with interactive visualizations
        - External data integration from mounted disk
        
        **Version:** 1.0.0
        **Last Updated:** January 2026
        """
    }
)


# ============================================================================
# STYLING
# ============================================================================
def load_custom_css():
    """Load custom CSS with fallback."""
    try:
        css_path = Path(__file__).parent / "assets" / "css" / "style.css"
        if css_path.exists():
            with open(css_path) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        else:
            apply_fallback_css()
    except Exception:
        apply_fallback_css()


def apply_fallback_css():
    """Apply fallback CSS styling."""
    st.markdown("""
    <style>
    /* Main layout */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Navigation cards */
    .nav-card {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .nav-card:hover {
        background-color: #e9ecef;
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .nav-card h3 {
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #e8f5e9;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #FF9800;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: white !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        border-radius: 0.5rem;
    }
    
    /* Tables */
    .dataframe {
        border-radius: 0.5rem;
        overflow: hidden;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-weight: 700;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #6c757d;
        font-size: 0.9rem;
        border-top: 1px solid #dee2e6;
        margin-top: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================
def initialize_session_state():
    """Initialize session state variables."""
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    if 'last_forecast_date' not in st.session_state:
        st.session_state.last_forecast_date = None
    
    if 'forecast_results' not in st.session_state:
        st.session_state.forecast_results = None
    
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None


# ============================================================================
# NAVIGATION
# ============================================================================
def navigate_to(page: str):
    """Navigate to a specific page."""
    st.session_state.page = page
    st.rerun()


def render_navigation():
    """Render navigation sidebar."""
    st.sidebar.title("🧭 Navigation")
    
    # Home button
    if st.sidebar.button("🏠 Home", use_container_width=True, 
                         type="primary" if st.session_state.page == 'home' else "secondary"):
        navigate_to('home')
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Features")
    
    # Forecasting button
    if st.sidebar.button("🔮 Forecasting", use_container_width=True,
                         type="primary" if st.session_state.page == 'forecast' else "secondary"):
        navigate_to('forecast')
    
    # Model comparison button (disabled for now)
    st.sidebar.button("📈 Model Comparison", use_container_width=True, 
                      disabled=True, help="Coming soon!")
    
    # Data exploration button (disabled for now)
    st.sidebar.button("🔍 Data Exploration", use_container_width=True,
                      disabled=True, help="Coming soon!")
    
    st.sidebar.markdown("---")
    
    # System info
    st.sidebar.subheader("ℹ️ System Info")
    
    if st.session_state.last_forecast_date:
        st.sidebar.info(f"Last forecast: {st.session_state.last_forecast_date}")
    
    if st.session_state.selected_model:
        st.sidebar.success(f"Active model: {st.session_state.selected_model}")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption(f"""
    **Version:** 1.0.0  
    **Updated:** {datetime.now().strftime('%Y-%m-%d')}
    """)


# ============================================================================
# HOME PAGE
# ============================================================================
def render_home_page():
    """Render the home page."""
    st.title("📈 Retail Demand Forecasting Dashboard")
    
    st.markdown("""
    <div class="success-box">
    <h3>Welcome to the Professional Retail Forecasting Platform</h3>
    <p>Advanced time series forecasting for <strong>Corporación Favorita</strong> retail demand.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="nav-card">
        <h3>🔮 Forecasting</h3>
        <p>Generate forecasts using trained models:</p>
        <ul>
        <li>ARIMA & SARIMA</li>
        <li>Prophet & ETS</li>
        <li>ML Models (RF, XGBoost, SVR)</li>
        <li>Deep Learning (LSTM)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Go to Forecasting →", key="nav_forecast", use_container_width=True):
            navigate_to('forecast')
    
    with col2:
        st.markdown("""
        <div class="nav-card">
        <h3>📈 Model Comparison</h3>
        <p>Compare model performance:</p>
        <ul>
        <li>Error metrics (MAE, RMSE, MAPE)</li>
        <li>Visual comparisons</li>
        <li>Statistical tests</li>
        <li>Best model selection</li>
        </ul>
        <p><em>Coming soon!</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="nav-card">
        <h3>🔍 Data Exploration</h3>
        <p>Explore historical data:</p>
        <ul>
        <li>Time series visualization</li>
        <li>Seasonality analysis</li>
        <li>Trend detection</li>
        <li>Anomaly detection</li>
        </ul>
        <p><em>Coming soon!</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Platform features
    st.markdown("---")
    st.subheader("🚀 Platform Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>📊 Data Management</h4>
        <ul>
        <li>Centralized data loading from external disk</li>
        <li>Dynamic filtering with caching</li>
        <li>Real-time data validation</li>
        <li>Efficient data preprocessing</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>🎯 Model Management</h4>
        <ul>
        <li>Pre-trained model artifacts</li>
        <li>No training required in UI</li>
        <li>Quick model loading</li>
        <li>Version control support</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick stats
    st.markdown("---")
    st.subheader("📈 Quick Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h2>12+</h2>
        <p>Model Types</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h2>4</h2>
        <p>Forecast Frequencies</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h2>6+</h2>
        <p>Visualization Types</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
        <h2>Real-time</h2>
        <p>Forecast Generation</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Getting started
    st.markdown("---")
    st.subheader("🎓 Getting Started")
    
    st.markdown("""
    <div class="warning-box">
    <h4>Quick Start Guide</h4>
    <ol>
    <li><strong>Navigate to Forecasting:</strong> Click the "Forecasting" button in the sidebar</li>
    <li><strong>Select a Model:</strong> Choose from ARIMA, Prophet, ML models, or LSTM</li>
    <li><strong>Configure Parameters:</strong> Set forecast horizon, frequency, and other options</li>
    <li><strong>Generate Forecast:</strong> Click "Run Forecast" to generate predictions</li>
    <li><strong>Visualize Results:</strong> Explore interactive charts and download data</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
    <p><strong>Retail Demand Forecasting Platform</strong> | Corporación Favorita</p>
    <p>Advanced Time Series Analysis & Forecasting | Version 1.0.0</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()
    
    # Load custom CSS
    load_custom_css()
    
    # Render navigation
    render_navigation()
    
    # Route to appropriate page
    if st.session_state.page == 'home':
        render_home_page()
    
    elif st.session_state.page == 'forecast':
        # Initialize and run forecast app
        forecast_app = Forecast()
        forecast_app.run()
        
        # Update session state if forecast was run
        if forecast_app.meta:
            st.session_state.selected_model = forecast_app.meta['label']
            st.session_state.last_forecast_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    else:
        st.error(f"Unknown page: {st.session_state.page}")
        st.info("Redirecting to home...")
        navigate_to('home')


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()