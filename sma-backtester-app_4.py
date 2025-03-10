import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from itertools import product
import io
import base64

# Set page configuration
st.set_page_config(page_title="SMA Strategy Backtester", layout="wide")

# App title and description
st.title("SMA Strategy Backtester")
st.markdown("""
This application allows you to backtest Simple Moving Average (SMA) crossover strategies 
on financial data. Upload your own data or use the sample EUR/USD data.
""")

# Sidebar for inputs
st.sidebar.header("Settings")

# Data upload or use sample data
data_option = st.sidebar.radio("Data Source", ["Use sample EUR/USD data", "Upload your own data"])

df = None
if data_option == "Use sample EUR/USD data":
    try:
        # In a real app, you would have this file available
        # For demonstration, we'll create synthetic data
        st.sidebar.info("Using synthetic EUR/USD data for demonstration")
        
        # Create sample data
        dates = pd.date_range(start='2015-01-01', end='2020-12-31', freq='B')
        np.random.seed(42)  # For reproducibility
        
        # Create a price series with some trend and randomness
        price = 1.1  # Starting price
        prices = [price]
        for _ in range(1, len(dates)):
            price *= (1 + np.random.normal(0.0001, 0.005))  # Small daily changes
            prices.append(price)
        
        df = pd.DataFrame({"price": prices}, index=dates)
        st.sidebar.success("Sample data loaded successfully")
    except Exception as e:
        st.sidebar.error(f"Error loading sample data: {e}")
else:
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file with price data", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
            
            # Check if 'price' column exists
            if 'price' not in df.columns:
                # If not, try to use the first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df = df.rename(columns={numeric_cols[0]: 'price'})
                    st.sidebar.info(f"Using '{numeric_cols[0]}' column as price data")
                else:
                    st.sidebar.error("No numeric columns found in the uploaded file")
                    df = None
            
            if df is not None:
                st.sidebar.success("Data loaded successfully")
        except Exception as e:
            st.sidebar.error(f"Error loading uploaded data: {e}")
            st.sidebar.info("Make sure your CSV has a 'Date' column and at least one price column")

# Strategy selection
strategy_type = st.sidebar.selectbox(
    "Strategy Type", 
    ["Data Overview", "Buy and Hold", "SMA Crossover", "Parameter Optimization"]
)

# Function to download dataframe as CSV
def get_table_download_link(df, filename="data.csv", text="Download CSV"):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to test SMA strategy
def test_sma_strategy(data, sma_s, sma_l):
    # Create a copy of the data to avoid modifying the original
    df_strat = data.copy()
    
    # Calculate returns
    df_strat["returns"] = np.log(df_strat.price.div(df_strat.price.shift(1)))
    
    # Calculate SMAs
    df_strat["SMA_S"] = df_strat.price.rolling(int(sma_s)).mean()
    df_strat["SMA_L"] = df_strat.price.rolling(int(sma_l)).mean()
    df_strat.dropna(inplace=True)
    
    # Calculate positions and strategy returns
    df_strat["position"] = np.where(df_strat["SMA_S"] > df_strat["SMA_L"], 1, -1)
    df_strat["strategy"] = df_strat.position.shift(1) * df_strat["returns"]
    df_strat.dropna(inplace=True)
    
    # Calculate cumulative returns using numpy for reliability
    returns_cumsum = df_strat["returns"].astype(float).cumsum().values
    strategy_cumsum = df_strat["strategy"].astype(float).cumsum().values
    
    # Apply exponential to cumulative sum using numpy
    df_strat["creturns"] = pd.Series(np.exp(returns_cumsum), index=df_strat.index)
    df_strat["cstrategy"] = pd.Series(np.exp(strategy_cumsum), index=df_strat.index)
    
    return df_strat

# Function to calculate performance metrics
def calculate_metrics(returns_series, strategy_series=None):
    metrics = {}
    
    # Basic performance
    metrics["Total Return"] = f"{np.exp(returns_series.sum()) - 1:.2%}"
    metrics["Annualized Return"] = f"{returns_series.mean() * 252:.2%}"
    metrics["Annualized Risk"] = f"{returns_series.std() * np.sqrt(252):.2%}"
    # Avoid division by zero for Sharpe ratio
    std = returns_series.std()
    metrics["Sharpe Ratio"] = f"{(returns_series.mean() / std if std > 0 else 0) * np.sqrt(252):.2f}"
    
    # Max Drawdown
    cum_returns = returns_series.cumsum().apply(np.exp)
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max - 1)
    metrics["Maximum Drawdown"] = f"{drawdown.min():.2%}"
    
    # Strategy specific metrics
    if strategy_series is not None:
        metrics["Strategy Return"] = f"{np.exp(strategy_series.sum()) - 1:.2%}"
        metrics["Strategy Annualized Return"] = f"{strategy_series.mean() * 252:.2%}"
        metrics["Strategy Annualized Risk"] = f"{strategy_series.std() * np.sqrt(252):.2%}"
        # Avoid division by zero for Strategy Sharpe ratio
        std = strategy_series.std()
        metrics["Strategy Sharpe Ratio"] = f"{(strategy_series.mean() / std if std > 0 else 0) * np.sqrt(252):.2f}"
        metrics["Outperformance"] = f"{np.exp(strategy_series.sum()) - np.exp(returns_series.sum()):.2%}"
    
    return metrics

if df is not None:
    if strategy_type == "Data Overview":
        st.header("Data Overview")
        
        # Show basic data information
        st.subheader("Dataset Information")
        st.write(f"Date range: {df.index.min()} to {df.index.max()}")
        st.write(f"Number of observations: {df.shape[0]}")
        
        # Display the head of the dataframe
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Plot price data
        st.subheader("Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df.price, mode='lines', name='Price'))
        fig.update_layout(
            height=500,
            xaxis_title='Date',
            yaxis_title='Price'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and plot returns
        returns = np.log(df.price.div(df.price.shift(1)))
        returns.dropna(inplace=True)
        
        # Using Plotly for returns histogram instead of matplotlib
        st.subheader("Returns Distribution")
        fig = px.histogram(returns, nbins=100)
        fig.update_layout(
            height=500,
            title="Returns Distribution",
            xaxis_title="Log Returns",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("Returns Statistics")
        st.dataframe(returns.describe())
        
    elif strategy_type == "Buy and Hold":
        st.header("Buy and Hold Strategy")
        
        try:
            # Calculate returns
            df_bh = df.copy()
            df_bh["returns"] = np.log(df_bh.price.div(df_bh.price.shift(1)))
            df_bh.dropna(inplace=True)
            
            # Calculate cumulative returns
            df_bh["creturns"] = df_bh["returns"].astype(float).cumsum().apply(np.exp)
            
            # Calculate drawdown using numpy directly
            cummax_array = np.maximum.accumulate(df_bh["creturns"].values)
            drawdown_array = (df_bh["creturns"].values - cummax_array) / np.where(cummax_array > 0, cummax_array, 1)
            df_bh["cummax"] = pd.Series(cummax_array, index=df_bh.index)
            df_bh["drawdown"] = pd.Series(drawdown_array, index=df_bh.index)
            
            # Performance metrics
            metrics = calculate_metrics(df_bh.returns)
            
            # Display metrics
            st.subheader("Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", metrics["Total Return"])
            col2.metric("Annualized Return", metrics["Annualized Return"])
            col3.metric("Annualized Risk", metrics["Annualized Risk"])
            col4.metric("Sharpe Ratio", metrics["Sharpe Ratio"])
            
            st.metric("Maximum Drawdown", metrics["Maximum Drawdown"])
            
            # Plot cumulative returns
            st.subheader("Cumulative Returns")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_bh.index, y=df_bh.creturns, mode='lines', name='Buy and Hold'))
            fig.update_layout(
                height=500,
                xaxis_title='Date',
                yaxis_title='Cumulative Returns'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot drawdown
            st.subheader("Drawdown")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_bh.index, 
                y=df_bh.drawdown, 
                mode='lines', 
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red')
            ))
            fig.update_layout(
                height=400,
                xaxis_title='Date',
                yaxis_title='Drawdown'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.subheader("Results Table")
            st.dataframe(df_bh[["returns", "creturns", "drawdown"]].tail(10))
            st.markdown(get_table_download_link(df_bh, filename="buy_and_hold_results.csv"), unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"An error occurred in the Buy and Hold strategy: {e}")
            st.info("Try using a different dataset or check if the data format is correct.")
        
    elif strategy_type == "SMA Crossover":
        st.header("SMA Crossover Strategy")
        
        # SMA parameters with input validation
        col1, col2 = st.columns(2)
        with col1:
            sma_s = st.slider("Short SMA Period", min_value=5, max_value=100, value=50, step=1)
        with col2:
            sma_l = st.slider("Long SMA Period", min_value=50, max_value=300, value=200, step=1)
            
        if sma_s >= sma_l:
            st.error("Short SMA period must be less than Long SMA period")
        else:
            try:
                # Run strategy
                df_sma = test_sma_strategy(df, sma_s, sma_l)
                
                # Calculate metrics
                metrics = calculate_metrics(df_sma.returns, df_sma.strategy)
                
                # Display metrics
                st.subheader("Performance Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Buy and Hold Return", metrics["Total Return"])
                col2.metric("Strategy Return", metrics["Strategy Return"])
                col3.metric("Outperformance", metrics["Outperformance"])
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Strategy Ann. Return", metrics["Strategy Annualized Return"])
                col2.metric("Strategy Ann. Risk", metrics["Strategy Annualized Risk"])
                col3.metric("Strategy Sharpe Ratio", metrics["Strategy Sharpe Ratio"])
                col4.metric("Maximum Drawdown", metrics["Maximum Drawdown"])
                
                # Plot SMAs and price
                st.subheader("Price and Moving Averages")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_sma.index, y=df_sma.price, mode='lines', name='Price'))
                fig.add_trace(go.Scatter(x=df_sma.index, y=df_sma.SMA_S, mode='lines', name=f'SMA {sma_s}'))
                fig.add_trace(go.Scatter(x=df_sma.index, y=df_sma.SMA_L, mode='lines', name=f'SMA {sma_l}'))
                fig.update_layout(
                    height=500,
                    xaxis_title='Date',
                    yaxis_title='Price'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Plot positions
                st.subheader("Trading Positions")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_sma.index, 
                    y=df_sma.position, 
                    mode='lines', 
                    name='Position',
                    line=dict(color='green', width=2)
                ))
                fig.update_layout(
                    height=300,
                    xaxis_title='Date',
                    yaxis_title='Position (1=Long, -1=Short)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Plot returns comparison
                st.subheader("Strategy vs Buy and Hold")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_sma.index, y=df_sma.creturns, mode='lines', name='Buy and Hold'))
                fig.add_trace(go.Scatter(x=df_sma.index, y=df_sma.cstrategy, mode='lines', name='SMA Strategy'))
                fig.update_layout(
                    height=500,
                    xaxis_title='Date',
                    yaxis_title='Cumulative Returns'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Results table
                st.subheader("Results Table")
                st.dataframe(df_sma[["returns", "strategy", "creturns", "cstrategy"]].tail(10))
                st.markdown(get_table_download_link(df_sma, filename="sma_strategy_results.csv"), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred in the SMA Crossover strategy: {e}")
                st.info("Try using different SMA periods or check if the data format is correct.")
            
    elif strategy_type == "Parameter Optimization":
        st.header("SMA Parameter Optimization")
        
        # Parameter ranges
        st.subheader("Set Parameter Ranges")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Short SMA Settings**")
            sma_s_min = st.number_input("Min Short SMA", min_value=5, max_value=50, value=10, step=1)
            sma_s_max = st.number_input("Max Short SMA", min_value=10, max_value=100, value=50, step=1)
            sma_s_step = st.number_input("Short SMA Step", min_value=1, max_value=10, value=5, step=1)
        
        with col2:
            st.markdown("**Long SMA Settings**")
            sma_l_min = st.number_input("Min Long SMA", min_value=50, max_value=150, value=100, step=5)
            sma_l_max = st.number_input("Max Long SMA", min_value=100, max_value=300, value=200, step=5)
            sma_l_step = st.number_input("Long SMA Step", min_value=5, max_value=20, value=10, step=5)
            
        if sma_s_max >= sma_l_min:
            st.error("Max Short SMA must be less than Min Long SMA")
        else:
            SMA_S_range = range(sma_s_min, sma_s_max + 1, sma_s_step)
            SMA_L_range = range(sma_l_min, sma_l_max + 1, sma_l_step)
            
            total_combinations = len(list(SMA_S_range)) * len(list(SMA_L_range))
            
            st.write(f"Total combinations to test: {total_combinations}")
            
            if st.button("Run Optimization"):
                if total_combinations > 100:
                    st.warning(f"Testing {total_combinations} combinations may take a while. Consider reducing the parameter ranges.")
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Generate combinations
                combinations = list(product(SMA_S_range, SMA_L_range))
                
                # Initialize results list
                results = []
                
                # Run tests for each combination
                for i, (s_sma, l_sma) in enumerate(combinations):
                    # Update progress
                    progress = (i + 1) / total_combinations
                    progress_bar.progress(progress)
                    status_text.text(f"Testing combination {i+1}/{total_combinations}: SMA_S={s_sma}, SMA_L={l_sma}")
                    
                    try:
                        # Test strategy
                        df_test = test_sma_strategy(df, s_sma, l_sma)
                        performance = np.exp(df_test["strategy"].sum())
                        
                        # Calculate metrics with safety checks
                        ann_return = df_test["strategy"].mean() * 252
                        ann_risk = df_test["strategy"].std() * np.sqrt(252)
                        sharpe = (ann_return / ann_risk) if ann_risk > 0 else 0
                        
                        # Store results
                        results.append({
                            "SMA_S": s_sma,
                            "SMA_L": l_sma,
                            "performance": performance,
                            "annualized_return": ann_return,
                            "annualized_risk": ann_risk,
                            "sharpe": sharpe
                        })
                    except Exception as e:
                        st.error(f"Error with SMA_S={s_sma}, SMA_L={l_sma}: {e}")
                        # Add a placeholder result to keep the loop going
                        results.append({
                            "SMA_S": s_sma,
                            "SMA_L": l_sma,
                            "performance": 1.0,  # Neutral performance (no gain/loss)
                            "annualized_return": 0,
                            "annualized_risk": 0,
                            "sharpe": 0
                        })
                
                # Create results DataFrame
                results_df = pd.DataFrame(results)
                
                # Display top performers
                st.subheader("Top 10 Parameter Combinations")
                top_results = results_df.nlargest(10, "performance").reset_index(drop=True)
                top_results["performance"] = top_results["performance"] - 1  # Convert to percentage
                st.dataframe(top_results.style.format({
                    "performance": "{:.2%}",
                    "annualized_return": "{:.2%}",
                    "annualized_risk": "{:.2%}",
                    "sharpe": "{:.2f}"
                }))
                
                # Allow download of results
                st.markdown(get_table_download_link(results_df, "optimization_results.csv"), unsafe_allow_html=True)
                
                # Create heatmap for visualization using Plotly instead of matplotlib
                st.subheader("Performance Heatmap")
                
                pivot_data = results_df.pivot_table(
                    index="SMA_S", 
                    columns="SMA_L", 
                    values="performance",
                    aggfunc='first'
                )
                
                # Convert pivot table to format suitable for Plotly heatmap
                # We need to handle the case where indices are not continuous
                fig = go.Figure(data=go.Heatmap(
                    z=pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    colorscale='Viridis',
                    colorbar=dict(title='Performance')
                ))
                
                fig.update_layout(
                    title="SMA Parameter Optimization Heatmap",
                    xaxis_title="Long SMA (SMA_L)",
                    yaxis_title="Short SMA (SMA_S)",
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Option to use best parameters
                best_params = results_df.loc[results_df["performance"].idxmax()]
                
                st.subheader("Best Parameters")
                col1, col2, col3 = st.columns(3)
                col1.metric("Short SMA", int(best_params["SMA_S"]))
                col2.metric("Long SMA", int(best_params["SMA_L"]))
                col3.metric("Performance", f"{best_params['performance'] - 1:.2%}")
                
                st.info(f"💡 Tip: You can use these optimal parameters ({int(best_params['SMA_S'])}, {int(best_params['SMA_L'])}) in the SMA Crossover strategy section.")
                
else:
    st.info("Please upload data or use sample data to proceed")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit for SMA strategy backtesting")
