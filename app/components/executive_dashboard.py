# app/components/executive_dashboard.py
"""
Executive Dashboard Module
Business-focused metrics and visualizations for executive decision-making
"""
import streamlit as st # type: ignore
from app.bootstrap import *  # ensures project root is on sys.path  # noqa: F403
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any  # noqa: F401
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # noqa: F401



class ExecutiveDashboard:
    """Executive dashboard with business-focused metrics and visualizations."""
    
    @staticmethod
    def create_kpi_dashboard(forecast_df: pd.DataFrame, 
                            historical_df: pd.DataFrame,
                            config: Dict[str, Any]) -> None:
        """
        Create comprehensive KPI dashboard.
        
        Args:
            forecast_df: Forecast results DataFrame
            historical_df: Historical data DataFrame
            config: Forecast configuration
        """
        st.subheader("📊 Executive Dashboard")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Performance KPIs", 
            "💰 Financial Impact",
            "🎯 Risk Analysis", 
            "📋 Executive Summary"
        ])
        
        with tab1:
            ExecutiveDashboard._create_performance_kpis(forecast_df, historical_df)
        
        with tab2:
            ExecutiveDashboard._create_financial_analysis(forecast_df, historical_df, config)
        
        with tab3:
            ExecutiveDashboard._create_risk_analysis(forecast_df)
        
        with tab4:
            ExecutiveDashboard._create_executive_summary(forecast_df, historical_df, config)
    
    @staticmethod
    def _create_performance_kpis(forecast_df: pd.DataFrame, historical_df: pd.DataFrame) -> None:
        """Create performance-focused KPI cards."""
        st.write("#### 🎯 Performance Metrics")
        
        # Calculate KPIs
        forecast_values = forecast_df["point_forecast"]
        
        # Row 1: Basic Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_forecast = forecast_values.sum()
            st.metric(
                "Total Forecast",
                f"{total_forecast:,.0f}",
                help="Sum of all forecasted values"
            )
        
        with col2:
            avg_forecast = forecast_values.mean()
            avg_historical = historical_df["unit_sales"].mean()
            growth_vs_history = ((avg_forecast / avg_historical) - 1) * 100 if avg_historical != 0 else 0
            st.metric(
                "Average vs History",
                f"{avg_forecast:,.1f}",
                delta=f"{growth_vs_history:+.1f}%",
                help="Average forecast vs historical average"
            )
        
        with col3:
            peak_forecast = forecast_values.max()
            peak_day = forecast_df.loc[forecast_values.idxmax(), "date"]
            st.metric(
                "Peak Forecast",
                f"{peak_forecast:,.0f}",
                help=f"Peak on {peak_day.strftime('%b %d')}"
            )
        
        with col4:
            volatility = forecast_values.std() / forecast_values.mean() * 100 if forecast_values.mean() != 0 else 0
            st.metric(
                "Forecast Volatility",
                f"{volatility:.1f}%",
                delta_color="inverse",
                help="Standard deviation as percentage of mean"
            )
        
        # Row 2: Trend and Growth Metrics
        st.write("---")
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            if len(forecast_df) >= 2:
                start_val = forecast_values.iloc[0]
                end_val = forecast_values.iloc[-1]
                total_growth = ((end_val - start_val) / start_val * 100) if start_val != 0 else 0
                st.metric(
                    "Total Growth",
                    f"{total_growth:+.1f}%",
                    help="Growth from start to end of forecast period"
                )
        
        with col6:
            # Calculate trend (simple linear regression)
            x = np.arange(len(forecast_values))
            y = forecast_values.values
            slope = np.polyfit(x, y, 1)[0]
            trend_per_day = (slope / forecast_values.mean()) * 100 if forecast_values.mean() != 0 else 0
            st.metric(
                "Daily Trend",
                f"{trend_per_day:+.2f}%",
                delta_color="normal",
                help="Average daily growth trend"
            )
        
        with col7:
            # Best/Worst day
            best_day_growth = forecast_values.pct_change().max() * 100
            st.metric(
                "Best Day Growth",
                f"{best_day_growth:+.1f}%",
                help="Maximum single-day growth"
            )
        
        with col8:
            worst_day_growth = forecast_values.pct_change().min() * 100
            st.metric(
                "Worst Day Growth",
                f"{worst_day_growth:+.1f}%",
                delta_color="inverse",
                help="Maximum single-day decline"
            )
    
    @staticmethod
    def _create_financial_analysis(forecast_df: pd.DataFrame, 
                                  historical_df: pd.DataFrame,
                                  config: Dict[str, Any]) -> None:
        """Create financial impact analysis."""
        st.write("#### 💰 Financial Impact Analysis")
        
        # Assumptions (could be made configurable)
        avg_price = 10.0  # Average price per unit
        cost_per_unit = 6.0  # Cost per unit
        holding_cost_rate = 0.15  # 15% annual holding cost
        
        forecast_values = forecast_df["point_forecast"]
        
        # Financial metrics
        total_revenue = forecast_values.sum() * avg_price
        total_cost = forecast_values.sum() * cost_per_unit
        gross_profit = total_revenue - total_cost
        
        # Inventory metrics
        avg_inventory_needed = forecast_values.mean()
        safety_stock = forecast_values.std() * 1.65  # 95% confidence
        total_inventory_value = (avg_inventory_needed + safety_stock) * cost_per_unit
        holding_cost = total_inventory_value * (holding_cost_rate / 365) * len(forecast_df)
        
        # Row 1: Revenue and Profit
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Projected Revenue",
                f"${total_revenue:,.0f}",
                help="Total revenue from forecasted sales"
            )
        
        with col2:
            gross_margin = (gross_profit / total_revenue * 100) if total_revenue != 0 else 0
            st.metric(
                "Gross Margin",
                f"{gross_margin:.1f}%",
                help="Gross profit as percentage of revenue"
            )
        
        with col3:
            roi = (gross_profit / total_cost * 100) if total_cost != 0 else 0
            st.metric(
                "Return on Investment",
                f"{roi:.1f}%",
                help="Gross profit as percentage of cost"
            )
        
        with col4:
            st.metric(
                "Daily Profit",
                f"${(gross_profit / len(forecast_df)):,.0f}",
                help="Average daily gross profit"
            )
        
        # Row 2: Inventory Metrics
        st.write("---")
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric(
                "Required Inventory",
                f"{avg_inventory_needed:,.0f}",
                help="Average inventory needed"
            )
        
        with col6:
            st.metric(
                "Safety Stock",
                f"{safety_stock:,.0f}",
                help="Recommended safety stock (95% confidence)"
            )
        
        with col7:
            stockout_risk = 100 * (1 - config.get("confidence_level", 0.95))
            st.metric(
                "Stockout Risk",
                f"{stockout_risk:.1f}%",
                delta_color="inverse",
                help="Probability of stockout"
            )
        
        with col8:
            st.metric(
                "Holding Cost",
                f"${holding_cost:,.0f}",
                help="Estimated inventory holding cost"
            )
        
        # Visualizations
        st.write("---")
        col9, col10 = st.columns(2)
        
        with col9:
            # Revenue projection chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=forecast_df["date"],
                y=forecast_values * avg_price,
                name="Daily Revenue",
                marker_color='green',
                opacity=0.7
            ))
            
            # Add cumulative revenue line
            cumulative_revenue = (forecast_values * avg_price).cumsum()
            fig.add_trace(go.Scatter(
                x=forecast_df["date"],
                y=cumulative_revenue,
                name="Cumulative Revenue",
                yaxis="y2",
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title="Revenue Projection",
                yaxis=dict(title="Daily Revenue ($)"),
                yaxis2=dict(
                    title="Cumulative Revenue ($)",
                    overlaying="y",
                    side="right"
                ),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col10:
            # Profit margin chart
            daily_revenue = forecast_values * avg_price
            daily_cost = forecast_values * cost_per_unit
            daily_profit = daily_revenue - daily_cost
            daily_margin = (daily_profit / daily_revenue * 100).fillna(0)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_df["date"],
                y=daily_margin,
                name="Daily Margin %",
                line=dict(color='purple', width=2),
                fill='tozeroy',
                fillcolor='rgba(128, 0, 128, 0.1)'
            ))
            
            fig.add_hline(
                y=daily_margin.mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Avg: {daily_margin.mean():.1f}%"
            )
            
            fig.update_layout(
                title="Daily Profit Margin",
                yaxis=dict(title="Margin (%)"),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _create_risk_analysis(forecast_df: pd.DataFrame) -> None:
        """Create risk analysis section."""
        st.write("#### 🎯 Risk Analysis")
        
        if "lower_bound" not in forecast_df.columns or "upper_bound" not in forecast_df.columns:
            st.info("Confidence intervals not available for risk analysis")
            return
        
        forecast_values = forecast_df["point_forecast"]
        lower_bounds = forecast_df["lower_bound"]
        upper_bounds = forecast_df["upper_bound"]
        
        # Risk metrics
        uncertainty_range = upper_bounds - lower_bounds
        avg_uncertainty = uncertainty_range.mean()
        max_uncertainty = uncertainty_range.max()
        min_uncertainty = uncertainty_range.min()  # noqa: F841
        
        # Value at Risk (VaR) calculation
        confidence_level = 0.95  # noqa: F841
        var_95 = forecast_values.sum() - lower_bounds.sum()
        var_percentage = (var_95 / forecast_values.sum() * 100) if forecast_values.sum() != 0 else 0
        
        # Risk scores
        volatility_score = forecast_values.std() / forecast_values.mean() if forecast_values.mean() != 0 else 0
        uncertainty_score = avg_uncertainty / forecast_values.mean() if forecast_values.mean() != 0 else 0
        risk_score = (volatility_score + uncertainty_score) / 2 * 100
        
        # Row 1: Risk Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Value at Risk (95%)",
                f"${var_95:,.0f}",
                delta=f"{var_percentage:.1f}%",
                delta_color="inverse",
                help="Potential loss at 95% confidence level"
            )
        
        with col2:
            st.metric(
                "Average Uncertainty",
                f"±{avg_uncertainty/2:,.0f}",
                help="Average half-range of confidence interval"
            )
        
        with col3:
            st.metric(
                "Max Uncertainty",
                f"±{max_uncertainty/2:,.0f}",
                delta_color="inverse",
                help="Maximum uncertainty in forecast"
            )
        
        with col4:
            risk_level = "High" if risk_score > 20 else "Medium" if risk_score > 10 else "Low"
            st.metric(
                "Overall Risk Level",
                risk_level,
                delta=f"{risk_score:.1f}",
                delta_color="inverse" if risk_score > 15 else "normal"
            )
        
        # Risk visualization
        st.write("---")
        col5, col6 = st.columns(2)
        
        with col5:
            # Uncertainty over time
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_df["date"],
                y=uncertainty_range,
                name="Uncertainty Range",
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.2)',
                line=dict(color='red', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df["date"],
                y=[avg_uncertainty] * len(forecast_df),
                name=f"Average (±{avg_uncertainty/2:.0f})",
                line=dict(color='blue', dash='dash')
            ))
            
            fig.update_layout(
                title="Forecast Uncertainty Over Time",
                yaxis_title="Uncertainty Range",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col6:
            # Risk heatmap (by day of week)
            forecast_df_copy = forecast_df.copy()
            forecast_df_copy["day_of_week"] = forecast_df_copy["date"].dt.day_name()
            forecast_df_copy["uncertainty_pct"] = (uncertainty_range / forecast_values * 100).fillna(0)
            
            weekly_risk = forecast_df_copy.groupby("day_of_week")["uncertainty_pct"].mean()
            
            # Order days
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_risk = weekly_risk.reindex(day_order)
            
            fig = go.Figure(data=go.Bar(
                x=weekly_risk.index,
                y=weekly_risk.values,
                marker_color=['red' if x > weekly_risk.mean() else 'green' for x in weekly_risk.values]
            ))
            
            fig.update_layout(
                title="Risk by Day of Week",
                yaxis_title="Uncertainty (%)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _create_executive_summary(forecast_df: pd.DataFrame, 
                                 historical_df: pd.DataFrame,
                                 config: Dict[str, Any]) -> None:
        """Create executive summary with key insights."""
        st.write("#### 📋 Executive Summary")
        
        # Calculate key metrics
        forecast_values = forecast_df["point_forecast"]
        historical_values = historical_df["unit_sales"]
        
        total_forecast = forecast_values.sum()  # noqa: F841
        avg_forecast = forecast_values.mean()
        avg_historical = historical_values.mean()
        
        growth_vs_history = ((avg_forecast / avg_historical) - 1) * 100 if avg_historical != 0 else 0
        
        if "lower_bound" in forecast_df.columns:
            risk_level = "Medium"
            if (forecast_df["upper_bound"] - forecast_df["lower_bound"]).mean() / avg_forecast > 0.3:
                risk_level = "High"
            elif (forecast_df["upper_bound"] - forecast_df["lower_bound"]).mean() / avg_forecast < 0.1:
                risk_level = "Low"
        else:
            risk_level = "Not Available"
        
        # Key insights
        insights = []
        
        if growth_vs_history > 10:
            insights.append(f"📈 Strong growth projected: {growth_vs_history:.1f}% above historical average")
        elif growth_vs_history < -5:
            insights.append(f"📉 Decline projected: {abs(growth_vs_history):.1f}% below historical average")
        else:
            insights.append(f"➡️ Stable projection: {growth_vs_history:+.1f}% vs historical average")
        
        # Seasonality insight
        if "month" in forecast_df.columns:
            best_month = forecast_df.groupby("month")["point_forecast"].mean().idxmax()
            worst_month = forecast_df.groupby("month")["point_forecast"].mean().idxmin()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            insights.append(f"📅 Peak performance in {month_names[best_month-1]}, lowest in {month_names[worst_month-1]}")
        
        # Volatility insight
        volatility = forecast_values.std() / forecast_values.mean() * 100 if forecast_values.mean() != 0 else 0
        if volatility > 30:
            insights.append(f"⚡ High volatility detected: {volatility:.1f}% - consider risk mitigation")
        elif volatility < 10:
            insights.append(f"🎯 Low volatility: {volatility:.1f}% - stable forecast")
        
        # Display insights
        st.write("**Key Insights:**")
        for insight in insights:
            st.write(f"- {insight}")
        
        # Recommendations
        st.write("---")
        st.write("**🎯 Recommendations:**")
        
        recommendations = []
        
        if risk_level == "High":
            recommendations.append("Consider implementing safety stock or buffer inventory")
            recommendations.append("Monitor actual vs forecast closely during high-risk periods")
        
        if growth_vs_history > 15:
            recommendations.append("Plan for increased inventory and staffing")
            recommendations.append("Consider promotional activities to capitalize on growth")
        
        if len(forecast_df) > 30:
            recommendations.append("Break forecast into phases for better operational planning")
        
        for rec in recommendations:
            st.write(f"• {rec}")
        
        # Action items
        st.write("---")
        st.write("**📋 Immediate Action Items:**")
        
        action_items = [
            "Review and validate forecast assumptions",
            "Align inventory planning with forecast peaks",
            "Schedule regular forecast reviews (weekly)",
            "Document key assumptions and risks"
        ]
        
        for i, item in enumerate(action_items, 1):
            st.write(f"{i}. {item}")
    
    @staticmethod
    def create_forecast_scenarios(forecast_df: pd.DataFrame, 
                                 scenarios: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Create multiple forecast scenarios for what-if analysis.
        
        Args:
            forecast_df: Base forecast DataFrame
            scenarios: List of scenario names
            
        Returns:
            Dictionary of scenario DataFrames
        """
        if scenarios is None:
            scenarios = ["baseline", "optimistic", "pessimistic", "promotional", "conservative"]
        
        scenarios_dict = {}
        
        for scenario in scenarios:
            scenario_df = forecast_df.copy()
            
            if scenario == "optimistic":
                scenario_df["point_forecast"] *= 1.20  # 20% increase
                if "lower_bound" in scenario_df.columns:
                    scenario_df["lower_bound"] *= 1.10
                    scenario_df["upper_bound"] *= 1.30
            
            elif scenario == "pessimistic":
                scenario_df["point_forecast"] *= 0.80  # 20% decrease
                if "lower_bound" in scenario_df.columns:
                    scenario_df["lower_bound"] *= 0.70
                    scenario_df["upper_bound"] *= 0.90
            
            elif scenario == "promotional":
                # Add promotional lift on specific days (e.g., weekends)
                weekend_mask = scenario_df["date"].dt.dayofweek.isin([5, 6])
                scenario_df.loc[weekend_mask, "point_forecast"] *= 1.35  # 35% lift on weekends
                scenario_df.loc[~weekend_mask, "point_forecast"] *= 1.15  # 15% lift on weekdays
            
            elif scenario == "conservative":
                scenario_df["point_forecast"] *= 0.90  # 10% conservative
                if "lower_bound" in scenario_df.columns:
                    # Narrow confidence intervals for conservative
                    midpoint = (scenario_df["lower_bound"] + scenario_df["upper_bound"]) / 2
                    scenario_df["lower_bound"] = midpoint - (midpoint * 0.05)
                    scenario_df["upper_bound"] = midpoint + (midpoint * 0.05)
            
            scenarios_dict[scenario] = scenario_df
        
        return scenarios_dict
    
    @staticmethod
    def visualize_scenarios(scenarios_dict: Dict[str, pd.DataFrame]) -> go.Figure:
        """Visualize multiple forecast scenarios."""
        fig = go.Figure()
        
        colors = {
            "baseline": "blue",
            "optimistic": "green",
            "pessimistic": "red",
            "promotional": "orange",
            "conservative": "purple"
        }
        
        for scenario_name, scenario_df in scenarios_dict.items():
            color = colors.get(scenario_name, "gray")
            
            fig.add_trace(go.Scatter(
                x=scenario_df["date"],
                y=scenario_df["point_forecast"],
                mode="lines",
                name=f"{scenario_name.capitalize()} Scenario",
                line=dict(color=color, width=2),
                opacity=0.8
            ))
        
        fig.update_layout(
            title="Forecast Scenarios Comparison",
            xaxis_title="Date",
            yaxis_title="Unit Sales",
            hovermode="x unified",
            height=500,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig