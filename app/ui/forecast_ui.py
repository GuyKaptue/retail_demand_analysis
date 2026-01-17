# app/ui/forecast_ui.py - Refactored version

from app.bootstrap import *  # ensures project root is on sys.path  # noqa: F403

import streamlit as st  # type: ignore
import pandas as pd
from datetime import datetime
from typing import  List, Dict, Any

from app.utils.visualizer import ForecastVisualizer

from app.components.feature_forecast_builder import FeatureForecastBuilder

# ---------------------------------------------------------
# UI Components
# ---------------------------------------------------------


class ForecastUI:
    """UI components for forecast page."""

    @staticmethod
    def display_feature_statistics(builder: FeatureForecastBuilder):
        """Display feature statistics without serialization issues."""
        try:
            stats = builder.get_feature_statistics()

            st.write("#### 📊 Feature Statistics")

            # Quick metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                total = stats.get("total_features", 0)
                if isinstance(total, list):
                    total = len(total)
                st.metric("Total Features", total)
            with col2:
                lag_count = len(stats.get("lag_features", []))
                st.metric("Lag Features", lag_count)
            with col3:
                seasonal_count = len(stats.get("seasonal_features", []))
                st.metric("Seasonal Features", seasonal_count)

            # Feature breakdown
            st.write("#### 📋 Feature Breakdown")

            markdown_table = "| Category | Count | Examples |\n|----------|-------|----------|\n"

            categories = [
                ("numeric_features", "Numeric Features"),
                ("lag_features", "Lag Features"),
                ("rolling_features", "Rolling Features"),
                ("seasonal_features", "Seasonal Features"),
                ("date_features", "Date Features"),
                ("categorical_features", "Categorical Features"),
                ("other_features", "Other Features")
            ]

            for stat_key, display_name in categories:
                features = stats.get(stat_key, [])
                if not isinstance(features, list):
                    features = []

                if features:
                    examples = ", ".join(str(f) for f in features[:3])
                    if len(features) > 3:
                        examples += f", ... (+{len(features)-3} more)"
                    markdown_table += f"| {display_name} | {len(features)} | {examples} |\n"

            st.markdown(markdown_table)

            # Detailed lists in expander
            with st.expander("🔍 View Detailed Feature Lists"):
                for stat_key, display_name in categories:
                    features = stats.get(stat_key, [])
                    if isinstance(features, list) and len(features) > 0:
                        with st.expander(f"{display_name} ({len(features)})"):
                            cols = st.columns(2)
                            half = len(features) // 2 + 1

                            with cols[0]:
                                for feature in features[:half]:
                                    st.text(feature)

                            with cols[1]:
                                for feature in features[half:]:
                                    st.text(feature)

            # Insights
            with st.expander("💡 Model Insights"):
                total_features = stats.get("total_features", 0)
                if isinstance(total_features, list):
                    total_features = len(total_features)

                lag_features = len(stats.get("lag_features", []))
                rolling_features = len(stats.get("rolling_features", []))
                seasonal_features = len(stats.get("seasonal_features", []))

                st.info(f"""
                **Feature Analysis:**

                • **Total features**: {total_features}
                • **Time-based patterns**: {lag_features} lag + {rolling_features} rolling
                • **Seasonality captured**: {seasonal_features} seasonal features

                **Model Impact:**
                - More features can capture complex patterns but may risk overfitting
                - Lag features help capture historical dependencies
                - Seasonal features capture recurring patterns
                - Rolling features capture trends and momentum
                """)

        except Exception as e:
            st.warning(f"Feature statistics display error: {str(e)}")
            st.info("Proceeding with forecast generation...")

    @staticmethod
    def display_forecast_summary(result_df: pd.DataFrame):
        """Display forecast summary metrics."""
        st.subheader("📊 Forecast Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total = result_df["point_forecast"].sum()
            st.metric("Total Forecast", f"{total:,.0f}")

        with col2:
            avg = result_df["point_forecast"].mean()
            st.metric("Average", f"{avg:,.2f}")

        with col3:
            peak = result_df["point_forecast"].max()
            st.metric("Peak", f"{peak:,.0f}")

        with col4:
            if result_df["point_forecast"].iloc[0] != 0:
                growth = ((result_df["point_forecast"].iloc[-1] /
                          result_df["point_forecast"].iloc[0]) - 1) * 100
                st.metric("Growth", f"{growth:+.1f}%")
            else:
                st.metric("Growth", "N/A")

    @staticmethod
    def create_plots(
        visualizer: ForecastVisualizer,
        result_df: pd.DataFrame,
        plot_types: List[str],
        meta: Dict[str, Any],
        show_confidence: bool,
        download_plots: bool,
        historical_df: pd.DataFrame,
        actual_df: pd.DataFrame = None,
        confidence_level: float = 0.95
    ):
        """Create and display selected plots."""
        for plot_type in plot_types:
            st.subheader(f"📈 {plot_type}")

            try:
                fig = None

                if plot_type == "Timeseries":
                    fig = visualizer.plot_forecast_timeseries(
                        forecast_df=result_df,
                        historical_df=historical_df,
                        title=f"{meta['label']} Forecast",
                        show_confidence=show_confidence,
                        confidence_level=confidence_level
                    )

                elif plot_type == "Distribution":
                    fig = visualizer.plot_forecast_distribution_plotly(
                        forecast_df=result_df,
                        title="Forecast Value Distribution"
                    )

                elif plot_type == "Uncertainty":
                    if "lower_bound" in result_df.columns and "upper_bound" in result_df.columns:
                        fig = visualizer.plot_forecast_uncertainty(
                            forecast_df=result_df,
                            title="Forecast Uncertainty",
                            confidence_level=confidence_level
                        )
                    else:
                        st.info("⚠️ Confidence intervals not available for this model")
                        continue

                elif plot_type == "Components":
                    component_cols = [col for col in result_df.columns if any(
                        kw in col.lower() for kw in ["trend", "seasonal", "cycle", "residual"]
                    )]
                    if component_cols:
                        fig = visualizer.plot_forecast_components(
                            forecast_df=result_df,
                            component_cols=component_cols,
                            title="Forecast Components"
                        )
                    else:
                        st.info("ℹ️ No component columns found in forecast data")
                        continue

                elif plot_type == "Heatmap":
                    # Ensure proper data for heatmap
                    if "date" in result_df.columns and len(result_df) > 0:
                        try:
                            fig = visualizer.plot_forecast_heatmap(
                                forecast_df=result_df,
                                title="Monthly Forecast Heatmap"
                            )
                        except Exception as e:
                            st.warning(f"Could not create heatmap: {str(e)}")
                            st.info("Heatmap requires sufficient date range data")
                            continue
                    else:
                        st.info("ℹ️ Insufficient data for heatmap visualization")
                        continue

                elif plot_type == "Waterfall":
                    if len(historical_df) > 0 and "unit_sales" in historical_df.columns:
                        try:
                            fig = visualizer.plot_forecast_waterfall(
                                forecast_df=result_df,
                                start_value=historical_df["unit_sales"].iloc[-1],
                                title="Forecast Waterfall Chart"
                            )
                        except Exception as e:
                            st.warning(f"Could not create waterfall chart: {str(e)}")
                            continue
                    else:
                        st.info("ℹ️ Historical data not available for waterfall chart")
                        continue

                elif plot_type == "Forecast vs Actual":
                    if actual_df is not None and len(actual_df) > 0:
                        try:
                            fig = visualizer.plot_forecast_vs_actual(
                                forecast_df=result_df,
                                actual_df=actual_df,
                                title="Forecast vs Actual"
                            )
                        except Exception as e:
                            st.warning(f"Could not create comparison plot: {str(e)}")
                            continue
                    else:
                        st.info("ℹ️ Actual data not available for comparison")
                        continue

                elif plot_type == "Error Analysis":
                    if actual_df is not None and len(actual_df) > 0:
                        try:
                            fig = visualizer.plot_forecast_error_analysis(
                                forecast_df=result_df,
                                actual_df=actual_df,
                                title="Error Analysis"
                            )
                        except Exception as e:
                            st.warning(f"Could not create error analysis: {str(e)}")
                            continue
                    else:
                        st.info("ℹ️ Actual data not available for error analysis")
                        continue

                # Display the plot if created successfully
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)

                    # Download option
                    if download_plots:
                        col1, col2, col3 = st.columns(3)
                        with col2:
                            if st.button(
                                f"💾 Download {plot_type} Plot",
                                key=f"download_{plot_type}"
                            ):
                                filename = (
                                    f"{meta['label'].replace(' ', '_')}_{plot_type}_"
                                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                )
                                try:
                                    visualizer.save_figure(fig, filename, format="png")
                                    st.success(f"✅ Plot saved as {filename}.png")
                                except Exception as e:
                                    st.error(f"Failed to save plot: {e}")

            except Exception as e:
                st.error(f"❌ Error creating {plot_type} plot: {str(e)}")
                with st.expander("Show error details"):
                    st.code(str(e))

    @staticmethod
    def display_data_table(
        result_df: pd.DataFrame,
        meta: Dict[str, Any],
        freq: str,
        horizon: int
    ):
        """Display forecast data table with download option."""
        with st.expander("📄 View Forecast Data Table", expanded=False):
            # Format the dataframe for better display
            display_df = result_df.copy()

            # Format date column if it exists
            if "date" in display_df.columns:
                display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime('%Y-%m-%d')

            # Round numeric columns
            numeric_cols = display_df.select_dtypes(include=['float64', 'float32']).columns
            for col in numeric_cols:
                display_df[col] = display_df[col].round(2)

            # Display the table
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )

            # Download button
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast Data (CSV)",
                data=csv,
                file_name=f"forecast_{meta['label'].replace(' ', '_')}_{freq}_{horizon}.csv",
                mime="text/csv",
                use_container_width=True
            )

    @staticmethod
    def display_error_message(error: Exception, context: str = ""):
        """Display formatted error message."""
        st.error(f"❌ Error {context}: {str(error)}")

        with st.expander("🔍 View Error Details"):
            import traceback
            st.code("".join(traceback.format_exception(type(error), error, error.__traceback__)))

        st.info("""
        **Troubleshooting Tips:**
        - Ensure the model file exists in the correct directory
        - Check that all required dependencies are installed
        - Verify that the historical data is properly formatted
        - Try selecting a different model or adjusting parameters
        """)

    @staticmethod
    def display_warning_box(message: str, title: str = "⚠️ Warning"):
        """Display formatted warning box."""
        st.warning(f"**{title}**\n\n{message}")

    @staticmethod
    def display_info_box(message: str, title: str = "ℹ️ Information"):
        """Display formatted info box."""
        st.info(f"**{title}**\n\n{message}")

    @staticmethod
    def display_success_box(message: str, title: str = "✅ Success"):
        """Display formatted success box."""
        st.success(f"**{title}**\n\n{message}")
