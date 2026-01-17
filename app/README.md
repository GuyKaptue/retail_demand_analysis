# 🔮 Retail Demand Forecasting Platform

**Enterprise-Grade Time Series Forecasting Application for Corporación Favorita**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?logo=open-source-initiative&logoColor=white)](LICENSE)


---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Application Screenshots](#application-screenshots)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start Guide](#quick-start-guide)
- [Model Registry](#model-registry)
- [User Guide](#user-guide)
- [Technical Architecture](#technical-architecture)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🎯 Overview

The **Retail Demand Forecasting Platform** is a production-ready, enterprise-grade web application built with Streamlit that leverages **pre-trained machine learning and statistical models** to generate accurate sales forecasts for retail operations. This platform eliminates the need for model training in the UI, providing instant access to validated forecasting models with comprehensive KPI metrics and business intelligence dashboards.

### 🎪 Live Demo

![Home Page](../reports/docs/screenshot/Screenshot%202026-01-16%20at%2018.59.55.png)
*Professional dashboard with modern UI design and intuitive navigation*

### 🌟 What Makes This Platform Unique

- ✅ **Pre-Trained Models**: No training required—all models are production-ready
- ✅ **Executive Dashboard**: Business-focused KPIs and financial impact analysis
- ✅ **Multi-Model Support**: 20+ trained models across classical, ML, and deep learning approaches
- ✅ **Advanced Analytics**: Performance tracking, model comparison, and batch forecasting
- ✅ **Enterprise Features**: Auto model selection, dynamic filtering, and comprehensive reporting

---

## ✨ Key Features

### 🎯 Core Capabilities

#### 1. **Pre-Trained Model Forecasting**
- **12+ Production Models** ready to use
- Classical: ARIMA, SARIMA, ETS, Prophet
- Machine Learning: Random Forest, XGBoost, SVR
- Deep Learning: LSTM
- No training overhead—instant forecasts

#### 2. **Executive Dashboard**
![Executive Dashboard](../reports/docs/screenshot/Screenshot%202026-01-16%20at%2021.21.58.png)
*Comprehensive KPI metrics with financial impact analysis*

**Business Intelligence Features:**
- 📈 Performance KPIs (Total Forecast, Growth Trends, Volatility)
- 💰 Financial Impact Analysis (Revenue, Margins, ROI, Inventory Costs)
- 🎯 Risk Analysis (Value at Risk, Uncertainty Metrics, Risk Scores)
- 📋 Executive Summary (Key Insights, Recommendations, Action Items)

#### 3. **Advanced Model Analytics**
![Auto Model Selection](../reports/docs/screenshot/Screenshot%202026-01-16%20at%2022.26.20.png)
*Intelligent model recommendation based on business context*

- 🤖 **Auto-Best Model Selection**: AI-powered model recommendations
- 📊 **Performance Tracking**: Historical model performance monitoring
- 🔄 **Model Comparison**: Side-by-side performance analysis
- 📈 **Trend Analysis**: Performance metrics over time

#### 4. **Dynamic Visualizations**
![Forecast Visualizations](../reports/docs/screenshot/Screenshot%202026-01-16%20at%2022.23.41.png)
*Interactive plots with confidence intervals and trend analysis*

**Visualization Suite:**
- Time Series Plot with historical context
- Distribution Analysis
- Uncertainty Bands (95% Confidence Intervals)
- Component Decomposition (Trend, Seasonal, Residual)
- Comparison Plots
- Dynamic Filtering Views

#### 5. **Batch Forecasting**
![Batch Forecasting](../reports/docs/screenshot/Screenshot%202026-01-16%20at%2022.35.45.png)
*Process multiple items/stores simultaneously*

- Parallel execution for faster processing
- Progress tracking with real-time updates
- Comprehensive batch reports
- Historical batch management

### 🔧 Advanced Features

#### Dynamic Dashboard Filters
![Dynamic Filters](../reports/docs/screenshot/Screenshot%202026-01-16%20at%2022.22.16.png)
*Customize analysis with real-time filtering*

- **Time Focus**: All Periods, First Week, First Month, Peak Periods
- **Threshold Highlighting**: Identify values above custom thresholds
- **Analysis Depth**: Basic to Comprehensive views
- **Trend Options**: Moving averages, trend lines, seasonal decomposition
- **Comparison Modes**: Historical average, previous periods, scenarios
- **Visualization Styles**: Professional, Minimal, Colorful, Dark Mode

#### Performance Analytics
![Performance Tracking](screenshots/performance_analytics.png)
*Track model performance over time*

- Model reliability scores
- Success rate tracking
- Error metric trends (MAE, RMSE, MAPE)
- Best/worst performing models
- Automated report generation

---

## 📸 Application Screenshots

### Home Page
![Home Page](../reports/docs/screenshot/Screenshot%202026-01-16%20at%2018.59.55.png)
*Welcome dashboard with feature overview and quick navigation*

### Single Forecast Interface
![Forecast Interface](../reports/docs/screenshot/Screenshot%202026-01-16%20at%2021.21.58.png)
*Intuitive forecast configuration with sidebar controls*

### Forecast Settings & Results
![Forecast Settings](../reports/docs/screenshots/Screenshot_2026-01-16_at_21.14.54.png)
*Detailed forecast configuration with model selection and parameters*

### Executive Dashboard
![Executive Dashboard](../reports/docs/screenshots/Screenshot_2026-01-16_at_21.20.44.png)
*Comprehensive KPI dashboard with performance metrics and growth analysis*

### Dynamic Visualization Controls
![Visualization Options](../reports/docs/screenshots/Screenshot_2026-01-16_at_21.21.24.png)
*Advanced visualization controls with dynamic filtering and comparison tools*

### Interactive Forecast Results
![Interactive Results](../reports/docs/screenshots/Screenshot_2026-01-16_at_21.21.58.png)
*Interactive forecast visualization with historical comparison and distribution analysis*

### Forecast Completion & Data Table
![Forecast Complete](../reports/docs/screenshots/Screenshot_2026-01-16_at_21.22.20.png)
*Forecast completion summary with data table export options*

### Feature Statistics
![Feature Statistics](../reports/docs/screenshots/feature_statistics.png)
*Detailed breakdown of engineered features*


### Performance Analytics
![Performance Analytics](../reports/docs/screenshot/Screenshot%202026-01-16%20at%2021.20.44.png)
*Historical tracking and trend analysis*



---

## 📁 Project Structure

```
app/
├── app.py                              # Main application entry point
├── bootstrap.py                        # Project initialization & path setup
├── README.md                           # This comprehensive guide
│
├── assets/                             # Static resources
│   └── css/
│       └── style.css                   # Custom styling & themes
│
├── components/                         # Core business logic modules
│   ├── __init__.py                     # Component exports
│   ├── batch_forecast.py               # Batch forecasting engine
│   ├── executive_dashboard.py          # KPI & BI dashboard
│   ├── feature_forecast_builder.py     # Feature engineering pipeline
│   ├── model_selector.py               # Auto model selection AI
│   └── performance_tracker.py          # Model performance monitoring
│
├── pages/                              # Application pages
│   ├── __init__.py                     # Page exports
│   └── forecast.py                     # Main forecasting interface
│
├── ui/                                 # User interface components
│   ├── __init__.py                     # UI component exports
│   ├── forecast_engine.py              # Unified prediction engine
│   └── forecast_ui.py                  # Reusable UI components
│
└── utils/                              # Utility functions
    ├── __init__.py                     # Utility exports
    ├── helpers.py                      # Model registry & utilities
    └── visualizer.py                   # Plotting & visualization tools
```

### 🏗️ Module Responsibilities

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `app.py` | Application orchestration | Navigation, routing, session management |
| `pages/forecast.py` | Forecasting interface | `Forecast` - Main app controller |
| `components/batch_forecast.py` | Batch processing | `BatchForecaster` - Multi-item forecasting |
| `components/executive_dashboard.py` | Business intelligence | `ExecutiveDashboard` - KPI generation |
| `components/feature_forecast_builder.py` | Feature engineering | `FeatureForecastBuilder` - Feature creation |
| `components/model_selector.py` | Model AI | `AutoBestModelSelector` - Model recommendation |
| `components/performance_tracker.py` | Analytics | `ModelPerformanceTracker` - Performance logging |
| `ui/forecast_engine.py` | Predictions | `ForecastEngine` - Unified prediction API |
| `ui/forecast_ui.py` | UI components | `ForecastUI` - Reusable UI elements |
| `utils/helpers.py` | Utilities | `MODEL_REGISTRY`, model loading |
| `utils/visualizer.py` | Visualizations | `ForecastVisualizer` - Plotting tools |

---

## 🚀 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **Virtual Environment**: Recommended for dependency isolation
- **Memory**: 4GB RAM minimum (8GB recommended for LSTM models)
- **Storage**: 2GB free space for models and data

### Step-by-Step Installation

```bash
# 1. Navigate to project root
cd retail_demand_analysis

# 2. Activate virtual environment
source retail_env/bin/activate  # macOS/Linux
# OR
retail_env\Scripts\activate     # Windows

# 3. Verify dependencies are installed
pip install -r requirements.txt

# 4. Test installation
python -c "import streamlit; import pandas; import plotly; print('✅ All dependencies installed')"
```

### Core Dependencies

```txt
# UI Framework
streamlit>=1.28.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Visualization
plotly>=5.17.0
matplotlib>=3.7.0

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0
statsmodels>=0.14.0
prophet>=1.1.5

# Deep Learning (Optional - for LSTM)
tensorflow>=2.13.0

# Utilities
python-dateutil>=2.8.2
```

### Verify Installation

```bash
# Launch application
streamlit run app/app.py

# Expected output:
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501
# Network URL: http://192.168.x.x:8501
```

---

## 🎬 Quick Start Guide

### 1️⃣ Launch Application

```bash
# From project root
streamlit run app/app.py

# OR navigate to app directory first
cd app
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`.

### 2️⃣ Navigate to Forecasting

**Option A**: Use sidebar navigation
- Click **"🔮 Forecasting"** button in the sidebar

**Option B**: Use home page
- Click **"Go to Forecasting →"** button on the main card

### 3️⃣ Configure Your Forecast

![Configuration Panel](screenshots/configuration_panel.png)

```
📋 Configuration Checklist:
1. ✅ Select Model (e.g., "Prophet (Week 2)")
2. ✅ Choose Frequency (Daily, Weekly, Monthly, etc.)
3. ✅ Set Forecast Horizon (e.g., 30 days)
4. ✅ Select Start Date (defaults to day after historical data ends)
5. ✅ Set Promotion Status (On/Off)
6. ✅ Choose Visualizations (Timeseries, Distribution, Uncertainty, etc.)
7. ✅ Set Confidence Level (default: 95%)
```

### 4️⃣ Generate Forecast

1. Review your configuration in the sidebar
2. Click **"🚀 Run Forecast"** button
3. Wait for processing (typically 5-30 seconds)
4. View results in the main panel

### 5️⃣ Analyze Results

**Executive Dashboard** (Automatic)
- Performance KPIs
- Financial Impact Analysis
- Risk Assessment
- Executive Summary

**Visualizations**
- Interactive time series plot
- Distribution analysis
- Uncertainty bands
- Component breakdowns

**Data Export**
- Download forecast data as CSV
- Save plots as PNG images
- Export executive reports

---

## 📊 Model Registry

All models are pre-trained and production-ready. Located in `utils/helpers.py`:

### Classical Time Series Models (Week 2)

| Model Key | Type | Configuration | Best For |
|-----------|------|---------------|----------|
| `prophet_week2` | Prophet | Default | Seasonal patterns, holidays |
| `arima_514` | ARIMA | (5,1,4) | Short-term, stationary trends |
| `sarima_300_0117` | SARIMA | (3,0,0)x(0,1,1,7) | Weekly seasonality |
| `ets_week2` | ETS | Auto | Exponential smoothing needs |

### Machine Learning Models (Week 3)

| Model Key | Type | Best For |
|-----------|------|----------|
| `rf_model` | Random Forest | Complex patterns, feature interactions |
| `xgb_model` | XGBoost | Non-linear relationships, large datasets |
| `svr_model` | SVR | Small datasets, non-linear patterns |

### Deep Learning Models (Week 3)

| Model Key | Type | Input Features | Best For |
|-----------|------|----------------|----------|
| `lstm_model` | LSTM | Sequential data | Long-term dependencies |

---

## 📖 User Guide

### Home Page Features

![Home Page Overview](screenshots/home_page_overview.png)

The home page provides:

1. **Welcome Message** - Platform introduction
2. **Feature Cards** - Quick access to main features
   - 🔮 Forecasting
   - 📈 Model Comparison (Coming Soon)
   - 🔍 Data Exploration (Coming Soon)
3. **Platform Statistics**
   - 12+ Model Types
   - 4 Forecast Frequencies
   - 6+ Visualization Types
   - Real-time Forecast Generation
4. **Quick Start Guide** - Step-by-step instructions
5. **System Information** - Last forecast timestamp, active model

### Forecasting Interface

#### Sidebar Configuration Panel

![Sidebar Configuration](screenshots/sidebar_config.png)

**Model Selection**
```
Available Categories:
├── Classical Time Series
│   ├── Prophet (Week 2)
│   ├── ARIMA & SARIMA
│   └── ETS
│
├── Machine Learning
│   ├── Random Forest
│   ├── XGBoost
│   └── SVR
│
└── Deep Learning
    └── LSTM
```

**Frequency Options**

| Frequency | Min | Max | Default | Use Case |
|-----------|-----|-----|---------|----------|
| Daily (D) | 7 | 365 | 30 | Day-to-day operations |
| Weekly (W) | 4 | 52 | 12 | Weekly reports |
| Monthly (M) | 3 | 36 | 12 | Strategic planning |
| Quarterly (Q) | 2 | 16 | 8 | Board reports |
| Yearly (Y) | 1 | 10 | 3 | Long-term strategy |

**Visualization Options**

- ✅ **Timeseries**: Historical + Forecast with confidence bands
- ✅ **Distribution**: Forecast value distribution histogram
- ✅ **Uncertainty**: Confidence interval visualization
- ✅ **Components**: Trend, seasonal, residual breakdown
- ✅ **Comparison**: Multiple forecast comparison plots

#### Main Panel Features

**1. Feature Statistics** (Expandable)
![Feature Statistics](screenshots/feature_stats_detail.png)

```
Total Features: 40+
├── Lag Features: unit_sales_lag_1, lag_3, lag_7, etc.
├── Rolling Features: r3_mean, r7_mean, r30_mean, etc.
├── Seasonal Features: month_sin, month_cos, day_sin, etc.
├── Date Features: day_of_week, month, quarter, year
└── Static Features: store_avg_sales, store_item_median
```

**2. Executive Dashboard**

Four comprehensive tabs:

1. **📈 Performance KPIs**
   - Total Forecast
   - Average vs History (with growth %)
   - Peak Forecast (with date)
   - Forecast Volatility
   - Total Growth
   - Daily Trend
   - Best/Worst Day Growth

2. **💰 Financial Impact**
   - Projected Revenue
   - Gross Margin %
   - Return on Investment %
   - Daily Profit
   - Required Inventory
   - Safety Stock
   - Stockout Risk %
   - Holding Cost

3. **🎯 Risk Analysis**
   - Value at Risk (95%)
   - Average Uncertainty
   - Max Uncertainty
   - Overall Risk Level
   - Uncertainty Over Time (chart)
   - Risk by Day of Week (chart)

4. **📋 Executive Summary**
   - Key Insights (auto-generated)
   - Recommendations (contextual)
   - Action Items (prioritized)

**3. Forecast Summary Metrics**

```
┌─────────────────────┬──────────────┬──────────┬────────────┐
│ Total Forecast      │ Average      │ Peak     │ Growth     │
├─────────────────────┼──────────────┼──────────┼────────────┤
│ 45,234.00          │ 1,507.80     │ 2,145.00 │ +12.5%     │
└─────────────────────┴──────────────┴──────────┴────────────┘
```

**4. Interactive Visualizations**

Each plot includes:
- **Zoom**: Click and drag to zoom in
- **Pan**: Double-click to reset view
- **Hover**: Detailed information on mouseover
- **Legend**: Toggle series visibility
- **Export**: Download as PNG (if enabled)

**5. Data Table**

- Expandable forecast data table
- Formatted dates and rounded numbers
- One-click CSV download
- Timestamped filename

### Advanced Features

#### Auto Model Selection

![Auto Model Selection Interface](screenshots/auto_model_detail.png)

**How It Works:**
1. Select business context:
   - General forecasting
   - Short-term predictions
   - Long-term planning
   - Promotional periods
   - Inventory management
   - Financial planning

2. Set forecast horizon (days)

3. Choose optimization metric:
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)
   - MAPE (Mean Absolute Percentage Error)
   - R² (Coefficient of Determination)
   - Correlation
   - Coverage

4. Apply filters (optional):
   - Model types
   - Training weeks

5. Click **"Find Best Models"**

6. Review:
   - Recommended model with reasoning
   - Top 5 models comparison table
   - Performance visualization charts

#### Performance Analytics

![Performance Tracking](screenshots/performance_detail.png)

**Features:**
- Overall performance dashboard
- Model reliability scores
- Success rate tracking
- Total forecast runs
- Average success rate
- Detailed model analysis
- MAE trend over time
- 7-day moving average
- Performance report generation

#### Batch Forecasting

![Batch Processing](screenshots/batch_detail.png)

**Configuration:**
1. Enter item IDs (comma-separated)
2. Enter store IDs (comma-separated)
3. Select model
4. Configure forecast settings
5. Choose execution mode:
   - Parallel (faster)
   - Sequential (safer)
6. Set max workers (if parallel)

**Features:**
- Real-time progress tracking
- Success/failure statistics
- Detailed results table
- Batch report generation
- Historical batch management

---

## 🏗️ Technical Architecture

### Design Principles

1. **Separation of Concerns**
   - UI layer (Streamlit components)
   - Business logic (forecasting engine)
   - Data layer (feature building)
   - Visualization layer (plotting)

2. **Modularity**
   - Independent, testable components
   - Clear interfaces between modules
   - Minimal coupling

3. **Robustness**
   - Multiple fallback mechanisms
   - Comprehensive error handling
   - Graceful degradation

4. **Extensibility**
   - Easy to add new models
   - Simple visualization additions
   - Pluggable components

5. **Performance**
   - Lazy loading of models
   - Cached data operations
   - Efficient feature engineering

### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                           │
│  (Streamlit App - Navigation, Configuration, Display)        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 FORECAST CONTROLLER                          │
│         (pages/forecast.py - Forecast class)                 │
│  • Configuration Management                                  │
│  • Session State                                             │
│  • Workflow Orchestration                                    │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┴──────────────┬─────────────────┐
        ▼                            ▼                  ▼
┌──────────────┐        ┌────────────────────┐  ┌─────────────┐
│ DATA LOADING │        │ FEATURE BUILDING   │  │ MODEL       │
│              │───────▶│                    │─▶│ LOADING     │
│ Historical   │        │ FeatureForecast    │  │             │
│ Data (CSV)   │        │ Builder            │  │ Registry    │
└──────────────┘        └────────────────────┘  └──────┬──────┘
                                                        │
                        ┌───────────────────────────────┘
                        ▼
        ┌───────────────────────────────────┐
        │     FORECAST ENGINE               │
        │  (Unified Prediction Interface)   │
        │  • Model-specific predictors      │
        │  • Confidence intervals           │
        │  • Error handling                 │
        └───────────────┬───────────────────┘
                        │
        ┌───────────────┴──────────────┐
        ▼                              ▼
┌──────────────────┐        ┌──────────────────────┐
│ RESULTS          │        │ BUSINESS ANALYTICS   │
│ DataFrame        │───────▶│ • Executive Dashboard│
│ • point_forecast │        │ • KPI Calculation    │
│ • lower_bound    │        │ • Financial Analysis │
│ • upper_bound    │        │ • Risk Assessment    │
│ • date           │        └──────────────────────┘
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│         VISUALIZATION LAYER              │
│  • ForecastVisualizer                    │
│  • Plotly Charts                         │
│  • Interactive Plots                     │
│  • Export Functions                      │
└──────────────────────────────────────────┘
```

### Component Interaction Flow

```python
# Simplified pseudo-code showing key interactions

# 1. User Configuration
config = forecast_app.setup_sidebar()
# Returns: {model_key, freq, horizon, start_date, ...}

# 2. Load Historical Data (Cached)
historical_df = Forecast.load_historical()
# Returns: DataFrame with historical sales data

# 3. Load Pre-Trained Model
model = load_registered_model(config["model_key"])
model_type = MODEL_REGISTRY[config["model_key"]]["model_type"]
# Returns: Trained model object

# 4. Build Future Features
builder = FeatureForecastBuilder(historical_df, model_type)
future_df = builder.build_future_features(
    start_date=config["start_date"],
    horizon=config["horizon"],
    frequency=config["freq"],
    onpromotion=config["onpromotion"],
    model_type=model_type
)
# Returns: DataFrame with engineered features for future dates

# 5. Generate Predictions
engine = ForecastEngine(model, model_type, builder)
predictions = engine.predict(future_df, horizon, frequency)
# Returns: Array of forecast values

# 6. Get Confidence Intervals (if available)
confidence_intervals = engine.get_confidence_intervals(
    future_df, 
    confidence_level=config["confidence_level"]
)
# Returns: (lower_bounds, upper_bounds) or None

# 7. Create Results DataFrame
result_df = pd.DataFrame({
    "date": future_df["date"],
    "point_forecast": predictions,
    "lower_bound": lower_bounds,  # if available
    "upper_bound": upper_bounds   # if available
})

# 8. Generate Business Analytics
ExecutiveDashboard.create_kpi_dashboard(
    result_df, 
    historical_df, 
    config
)

# 9. Create Visualizations
visualizer = ForecastVisualizer(historical_df)
fig = visualizer.plot_forecast_timeseries(result_df, ...)

# 10. Display Results
ForecastUI.display_forecast_summary(result_df)
ForecastUI.create_plots(visualizer, result_df, ...)
```

### Feature Engineering Pipeline

```
Historical Data
    │
    ├─→ Basic Features
    │   ├─ onpromotion
    │   └─ onpromotion_lag_1
    │
    ├─→ Time Features (if model != arima/sarima/ets)
    │   ├─ day_of_week, month, quarter, year
    │   ├─ week_of_year, day_of_month, day_of_year
    │   └─ is_weekend, is_month_start/end, etc.
    │
    ├─→ Cyclical Features (if model != arima/sarima/ets)
    │   ├─ month_sin, month_cos
    │   ├─ day_of_week_sin, day_of_week_cos
    │   └─ year_sin, year_cos
    │
    ├─→ Lag Features (if model != arima/sarima/ets)
    │   └─ unit_sales_lag_[1,3,7,14,30,365]
    │
    ├─→ Rolling Features (if model != arima/sarima/ets)
    │   ├─ unit_sales_r[3,7,14,30,365]_mean
    │   ├─ unit_sales_r[3,7,14,30,365]_median
    │   └─ unit_sales_r7_std
    │
    ├─→ Static Features
    │   ├─ store_avg_sales
    │   └─ store_item_median
    │
    └─→ Model-Specific Adaptation
        ├─ Prophet: Rename 'date' → 'ds'
        ├─ ARIMA/SARIMA/ETS: Keep only basic features
        └─ LSTM: Add sequence features

    ↓
Future Feature Matrix
```

---

## 🔌 API Reference

### Core Classes

#### `Forecast` (Main Controller)

```python
class Forecast:
    """Main forecasting application controller."""
    
    def __init__(self):
        """Initialize forecast application with components."""
        
    @staticmethod
    @st.cache_data
    def load_historical() -> pd.DataFrame:
        """Load historical sales data from disk."""
        
    def setup_sidebar(self) -> Dict[str, Any]:
        """Setup sidebar configuration interface."""
        
    def run_forecast_pipeline(self, config: Dict[str, Any]):
        """Execute complete forecast pipeline."""
        
    def run(self):
        """Run the main application with tab navigation."""
```

#### `FeatureForecastBuilder`

```python
class FeatureForecastBuilder:
    """Build future features for forecasting models."""
    
    def __init__(
        self, 
        historical_df: pd.DataFrame,
        model_type: str = "arima"
    ):
        """Initialize with historical data and model type."""
        
    def build_future_features(
        self,
        start_date: pd.Timestamp,
        horizon: int,
        frequency: str = "D",
        onpromotion: int = 0,
        model_type: str = "ml",
        include_lags: bool = True,
        include_rolling: bool = True,
        include_seasonal: bool = True,
        include_cyclical: bool = True
    ) -> pd.DataFrame:
        """
        Build future feature matrix.
        
        Returns:
            DataFrame with engineered features for future dates
        """
        
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Get statistics about features in historical data."""
```

#### `ForecastEngine`

```python
class ForecastEngine:
    """Unified prediction engine for all model types."""
    
    def __init__(
        self, 
        model: Any,
        model_type: str,
        builder: FeatureForecastBuilder
    ):
        """Initialize engine with model and builder."""
        
    def predict(
        self,
        future_df: pd.DataFrame,
        horizon: int,
        frequency: str = "D"
    ) -> np.ndarray:
        """
        Generate predictions using appropriate method.
        
        Returns:
            Array of forecast values
        """
        
    def get_confidence_intervals(
        self,
        future_df: pd.DataFrame,
        confidence_level: float = 0.95
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate confidence intervals if supported.
        
        Returns:
            (lower_bounds, upper_bounds) or None
        """
```

#### `ExecutiveDashboard`

```python
class ExecutiveDashboard:
    """Generate business intelligence dashboards."""
    
    @staticmethod
    def create_kpi_dashboard(
        forecast_df: pd.DataFrame,
        historical_df: pd.DataFrame,
        config: Dict[str, Any]
    ):
        """Create comprehensive KPI dashboard."""
        
    @staticmethod
    def calculate_performance_metrics(
        forecast_df: pd.DataFrame,
        historical_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate performance metrics and KPIs."""
        
    @staticmethod
    def calculate_financial_impact(
        forecast_df: pd.DataFrame,
        historical_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate financial metrics and projections."""
        
    @staticmethod
    def analyze_risk(
        forecast_df: pd.DataFrame,
        confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, Any]:
        """Analyze forecast risk and uncertainty."""
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **Models Not Loading**
```bash
# Check model files exist
ls -la models/
# Expected: .pkl, .joblib files for each model

# Verify Python version compatibility
python --version
# Should be 3.8+
```

#### 2. **Memory Issues**
```bash
# Check available memory
free -h

# Solution: Reduce batch size
# In configuration: Set max_workers to 1
```

#### 3. **Slow Performance**
```bash
# Enable caching
# Already implemented - verify @st.cache_data decorators are present

# Reduce visualization complexity
# Use fewer plot types or shorter horizons
```

#### 4. **Visualization Errors**
```bash
# Clear Streamlit cache
rm -rf .streamlit/cache

# Update Plotly
pip install --upgrade plotly
```

### Error Messages & Solutions

| Error Message | Likely Cause | Solution |
|---------------|--------------|----------|
| `Model not found in registry` | Model key incorrect | Check available models in utils/helpers.py |
| `Out of memory` | Large horizon or batch | Reduce horizon, use sequential processing |
| `No historical data found` | Path incorrect | Verify data/ directory exists with CSV files |
| `Confidence intervals not available` | Model doesn't support | Use Prophet or statistical models |
| `Feature building failed` | Data format issue | Check CSV column names and data types |

---

## 🤝 Contributing

### Development Setup

```bash
# 1. Fork the repository
# 2. Clone your fork
git clone https://github.com/GuyKaptue/retail_demand_analysis.git

# 3. Create feature branch
git checkout -b feature/amazing-feature

# 4. Make changes and commit
git commit -m "Add amazing feature"

# 5. Push to branch
git push origin feature/amazing-feature

# 6. Open Pull Request
```

### Adding New Models

1. **Train Model**: Save as `.pkl` or `.joblib` in `models/`
2. **Register Model**: Add to `MODEL_REGISTRY` in `utils/helpers.py`
3. **Add Predictor**: Extend `ForecastEngine.predict()` if new model type
4. **Test**: Verify predictions and confidence intervals work

### Code Standards

- Follow PEP 8 style guide
- Add docstrings for all functions and classes
- Include type hints
- Write unit tests for new functionality
- Update documentation when adding features

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Corporación Favorita** for the retail dataset
- **Streamlit** team for the amazing framework
- **Plotly** for interactive visualizations
- **Scikit-learn**, **Statsmodels**, **Prophet**, **XGBoost** teams for ML libraries
- All contributors and users of this platform

---

## 📞 Support

For issues, questions, or feature requests:

1. **GitHub Issues**: [Create new issue]("https://github.com/GuyKaptue/retail_demand_analysis/issues")
2. **Documentation**: Check this README and code comments
3. **Email**: your-email@example.com
4. **Slack**: Join our community channel

---

**Retail Demand Forecasting Platform | Corporación Favorita**  
**Advanced Time Series Analysis & Forecasting | Version 1.0.0**  
**Updated: 2026-01-16**