import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from itertools import product
import base64
import io
from iterative_base import IterativeBase
from iterative_backtest import IterativeBacktest

# Set page configuration
st.set_page_config(page_title="Advanced Trading Strategy Backtester", layout="wide")

# App title and description
st.title("Advanced Trading Strategy Backtester")
st.markdown("""
This application allows you to backtest multiple trading strategies:
- Simple Moving Average (SMA) crossover strategy
- Contrarian strategy
- Bollinger Bands strategy

Upload your own data or use the sample data.
""")

# Function to download dataframe as CSV
def get_table_download_link(df, filename="data.csv", text="Download CSV"):
    """
    Creates a download link for a DataFrame as CSV.
    
    Parameters
    ----------
    df: pandas.DataFrame
        DataFrame to be downloaded
    filename: str
        Name of the file to be downloaded
    text: str
        Text to display for the download link
        
    Returns
    -------
    str
        HTML link for downloading the DataFrame
    """
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to prepare data for strategy testing
def prepare_price_data(data, pair):
    """
    Prepares data for backtesting by extracting the price and spread for a specified pair.
    
    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame containing price data for multiple pairs
    pair: str
        Currency pair to extract
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with price, spread and returns for the specified pair
    """
    # Create a copy with the selected pair, renamed to 'price'
    df_pair = pd.DataFrame(data[pair].copy()).rename(columns={pair: 'price'})
    
    # Add spread if available
    spread_col = f"{pair}_spread"
    if spread_col in data.columns:
        df_pair["spread"] = data[spread_col]
    else:
        df_pair["spread"] = 0.0001  # Default 1 pip spread
    
    # Calculate returns
    df_pair["returns"] = np.log(df_pair.price.div(df_pair.price.shift(1)))
    
    return df_pair

# Function to calculate performance metrics
def calculate_metrics(returns_series, strategy_series=None):
    """
    Calculates performance metrics for a returns series and optionally a strategy series.
    
    Parameters
    ----------
    returns_series: pandas.Series
        Series of returns for the buy and hold strategy
    strategy_series: pandas.Series, optional
        Series of returns for the trading strategy
        
    Returns
    -------
    dict
        Dictionary of performance metrics
    """
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

# Function to get numerical value from metric string for cross-pair analysis
def get_metric_value(metric_str):
    """
    Extracts the numerical value from a metric string.
    
    Parameters
    ----------
    metric_str: str
        Metric string potentially containing a percentage
        
    Returns
    -------
    float
        Numerical value of the metric
    """
    try:
        # Remove % sign and convert to float
        return float(metric_str.strip('%')) / 100
    except:
        # If it's not a percentage, just convert to float
        return float(metric_str)

# Sidebar for inputs
st.sidebar.header("Settings")

# Data upload or use sample data
data_option = st.sidebar.radio("Data Source", ["Use sample data", "Upload your own data"])

df = None
available_pairs = []

if data_option == "Use sample data":
    try:
        # Create sample multi-currency data
        dates = pd.date_range(start='2015-01-01', end='2020-12-31', freq='B')
        np.random.seed(42)  # For reproducibility
        
        sample_data = {'time': dates}
        
        # Simulate multiple currency pairs
        pairs = ['EURUSD', 'GBPUSD', 'EURAUD']
        
        for pair in pairs:
            # Create a price series with some trend and randomness
            start_price = 1.1 if 'EUR' in pair else (1.3 if 'GBP' in pair else 1.5)
            price = start_price
            prices = [price]
            
            # Add a small spread for transaction costs
            spread = 0.0001  # 1 pip spread
            spreads = [spread]
            
            for _ in range(1, len(dates)):
                price *= (1 + np.random.normal(0.0001, 0.005))  # Small daily changes
                prices.append(price)
                spreads.append(spread)
                
            sample_data[pair] = prices
            sample_data[f"{pair}_spread"] = spreads
        
        df = pd.DataFrame(sample_data)
        df.set_index('time', inplace=True)
        
        available_pairs = [col for col in df.columns if not col.endswith('_spread')]
        st.sidebar.success(f"Sample data loaded successfully with {len(available_pairs)} currency pairs")
    except Exception as e:
        st.sidebar.error(f"Error loading sample data: {e}")
else:
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file with currency price data", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=["time"], index_col="time")
            
            # Get available currency pairs (all columns except those ending with _spread)
            available_pairs = [col for col in df.columns if not col.endswith('_spread')]
            
            if len(available_pairs) == 0:
                st.sidebar.error("No currency pairs found in the uploaded file")
                df = None
            else:
                st.sidebar.success(f"Data loaded successfully with {len(available_pairs)} currency pairs")
        except Exception as e:
            st.sidebar.error(f"Error loading uploaded data: {e}")
            st.sidebar.info("Make sure your CSV has a 'time' column and at least one currency pair column")

# Currency pair selection (only if data is loaded)
selected_pair = None
if df is not None and len(available_pairs) > 0:
    selected_pair = st.sidebar.selectbox("Select Currency Pair", available_pairs, index=0)
    
    # Strategy selection
    strategy_type = st.sidebar.selectbox(
        "Strategy Type", 
        ["Data Overview", "Buy and Hold", "SMA Crossover", "Contrarian Strategy", "Bollinger Bands Strategy",
         "SMA Parameter Optimization", "Contrarian Parameter Optimization", "Bollinger Parameter Optimization",
         "Cross-Pair Analysis"]
    )
    
    # Initial capital
    initial_capital = st.sidebar.number_input("Initial Capital", min_value=1000.0, max_value=1000000.0, value=10000.0, step=1000.0)
    
    # Transaction costs settings
    use_spread = st.sidebar.checkbox("Use Spread for Transaction Costs", value=True)
    
    # For Cross-Pair Analysis, allow selection of multiple pairs for comparison
    if strategy_type == "Cross-Pair Analysis":
        comparison_pairs = st.sidebar.multiselect(
            "Select Pairs to Compare",
            available_pairs,
            default=[available_pairs[0]] if available_pairs else []
        )
        if not comparison_pairs:
            st.sidebar.warning("Please select at least one currency pair to compare.")
    else:
        comparison_pairs = [selected_pair]

# Main content area - only if data is loaded
if df is not None and selected_pair is not None:
    # Display selected currency pair (except for Cross-Pair Analysis)
    if strategy_type != "Cross-Pair Analysis":
        st.header(f"Analysis for {selected_pair}")
    
    if strategy_type == "Data Overview":
        st.header("Data Overview")
        
        # Show basic data information
        st.subheader("Dataset Information")
        st.write(f"Date range: {df.index.min()} to {df.index.max()}")
        st.write(f"Number of observations: {df.shape[0]}")
        st.write(f"Available currency pairs: {', '.join(available_pairs)}")
        
        # Display the head of the dataframe for the selected pair
        st.subheader("Data Preview")
        st.dataframe(df[[selected_pair]].head())
        
        # Plot price data for the selected pair
        st.subheader("Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df[selected_pair], mode='lines', name=f'{selected_pair}'))
        fig.update_layout(
            height=500,
            xaxis_title='Date',
            yaxis_title='Price'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and plot returns
        df_pair = prepare_price_data(df, selected_pair)
        returns = df_pair["returns"].dropna()
        
        # Using Plotly for returns histogram
        st.subheader("Returns Distribution")
        fig = px.histogram(returns, nbins=100)
        fig.update_layout(
            height=500,
            title=f"{selected_pair} Returns Distribution",
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
            # Prepare data for the selected pair
            df_pair = prepare_price_data(df, selected_pair)
            
            # Create backtest instance
            backtest = IterativeBacktest(
                symbol=selected_pair,
                start=str(df.index.min().date()),
                end=str(df.index.max().date()),
                amount=initial_capital,
                use_spread=use_spread
            )
            
            # Set the prepared data
            backtest.set_data(df_pair)
            
            # Calculate buy and hold performance (just buy at start, sell at end)
            backtest.position = 0
            backtest.trades = 0
            backtest.current_balance = initial_capital
            backtest.results = []
            
            # Buy at the beginning
            backtest.go_long(0, amount="all")
            backtest.position = 1
            
            # Close position at the end
            final_bar = len(backtest.data) - 1
            performance = backtest.close_pos(final_bar)
            
            # Calculate metrics
            metrics = calculate_metrics(df_pair.returns.dropna())
            
            # Display metrics
            st.subheader("Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", metrics["Total Return"])
            col2.metric("Annualized Return", metrics["Annualized Return"])
            col3.metric("Annualized Risk", metrics["Annualized Risk"])
            col4.metric("Sharpe Ratio", metrics["Sharpe Ratio"])
            
            st.metric("Maximum Drawdown", metrics["Maximum Drawdown"])
            
            # Display final performance from backtest
            st.subheader("Buy and Hold Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Initial Balance", f"${initial_capital:,.2f}")
            col2.metric("Final Balance", f"${performance['final_balance']:,.2f}")
            col3.metric("Performance", f"{performance['performance_pct']:.2f}%")
            
            # Plot cumulative returns
            df_pair["creturns"] = df_pair["returns"].cumsum().apply(np.exp)
            
            # Calculate drawdown
            drawdown = (df_pair["creturns"] / df_pair["creturns"].cummax() - 1)
            df_pair["drawdown"] = drawdown
            
            st.subheader("Cumulative Returns")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_pair.index, y=df_pair.creturns, mode='lines', name=f'{selected_pair} Buy and Hold'))
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
                x=df_pair.index, 
                y=df_pair.drawdown, 
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
            
            # Trade summary
            if backtest.results:
                st.subheader("Trade Summary")
                trades_df = pd.DataFrame(backtest.results)
                st.dataframe(trades_df)
                st.markdown(get_table_download_link(trades_df, filename=f"{selected_pair}_buy_and_hold_trades.csv", text="Download Trades Data"), unsafe_allow_html=True)
            
            # Results table
            st.subheader("Results Table")
            st.dataframe(df_pair[["returns", "creturns", "drawdown"]].tail(10))
            st.markdown(get_table_download_link(df_pair, filename=f"{selected_pair}_buy_and_hold_results.csv"), unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"An error occurred in the Buy and Hold strategy: {e}")
            st.info("Try using a different currency pair or check if the data format is correct.")

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
                # Prepare data for the selected pair
                df_pair = prepare_price_data(df, selected_pair)
                
                # Create backtest instance
                backtest = IterativeBacktest(
                    symbol=selected_pair,
                    start=str(df.index.min().date()),
                    end=str(df.index.max().date()),
                    amount=initial_capital,
                    use_spread=use_spread
                )
                
                # Set the prepared data
                backtest.set_data(df_pair)
                
                # Run SMA strategy
                performance = backtest.test_sma_strategy(sma_s, sma_l)
                
                # Calculate metrics
                metrics = calculate_metrics(backtest.data.returns, backtest.data.strategy)
                
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
                
                # Display final performance from backtest
                st.subheader("Backtest Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("Initial Balance", f"${initial_capital:,.2f}")
                col2.metric("Final Balance", f"${performance['final_balance']:,.2f}")
                col3.metric("Total Trades", f"{performance['trades']}")
                
                # Plot SMAs and price
                st.subheader("Price and Moving Averages")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.price, mode='lines', name=f'{selected_pair}'))
                fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.SMA_S, mode='lines', name=f'SMA {sma_s}'))
                fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.SMA_L, mode='lines', name=f'SMA {sma_l}'))
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
                    x=backtest.data.index, 
                    y=backtest.data.position, 
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
                fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.creturns, mode='lines', name='Buy and Hold'))
                fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.cstrategy, mode='lines', name='SMA Strategy'))
                fig.update_layout(
                    height=500,
                    xaxis_title='Date',
                    yaxis_title='Cumulative Returns'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Trade summary
                if performance['trades_df'] is not None:
                    st.subheader("Trade Summary")
                    trades_df = performance['trades_df']
                    st.dataframe(trades_df)
                    st.markdown(get_table_download_link(trades_df, filename=f"{selected_pair}_sma_trades.csv", text="Download Trades Data"), unsafe_allow_html=True)
                
                # Results table
                st.subheader("Results Table")
                st.dataframe(backtest.data[["returns", "strategy", "creturns", "cstrategy"]].tail(10))
                st.markdown(get_table_download_link(backtest.data, filename=f"{selected_pair}_sma_strategy_results.csv"), unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"An error occurred in the SMA Crossover strategy: {e}")
                st.info("Try using different SMA periods or check if the data format is correct.")
    
    elif strategy_type == "Contrarian Strategy":
        st.header("Contrarian Strategy")
        
        # Contrarian parameters
        window = st.slider("Look-back Window", min_value=1, max_value=100, value=10, step=1)
        
        try:
            # Prepare data for the selected pair
            df_pair = prepare_price_data(df, selected_pair)
            
            # Create backtest instance
            backtest = IterativeBacktest(
                symbol=selected_pair,
                start=str(df.index.min().date()),
                end=str(df.index.max().date()),
                amount=initial_capital,
                use_spread=use_spread
            )
            
            # Set the prepared data
            backtest.set_data(df_pair)
            
            # Run Contrarian strategy
            performance = backtest.test_con_strategy(window)
            
            # Calculate metrics
            metrics = calculate_metrics(backtest.data.returns, backtest.data.strategy)
            
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
            
            # Display final performance from backtest
            st.subheader("Backtest Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Initial Balance", f"${initial_capital:,.2f}")
            col2.metric("Final Balance", f"${performance['final_balance']:,.2f}")
            col3.metric("Total Trades", f"{performance['trades']}")
            
            # Plot positions
            st.subheader("Trading Positions")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=backtest.data.index, 
                y=backtest.data.position, 
                mode='lines', 
                name='Position',
                line=dict(color='purple', width=2)
            ))
            fig.update_layout(
                height=300,
                xaxis_title='Date',
                yaxis_title='Position (1=Long, -1=Short)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot rolling returns (to show what the strategy is reacting to)
            st.subheader("Rolling Returns (Signal)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=backtest.data.index, 
                y=backtest.data["rolling_returns"], 
                mode='lines', 
                name=f'Rolling {window}-Period Returns',
                line=dict(color='orange', width=2)
            ))
            fig.update_layout(
                height=300,
                xaxis_title='Date',
                yaxis_title='Rolling Returns'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot returns comparison
            st.subheader("Strategy vs Buy and Hold")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.creturns, mode='lines', name='Buy and Hold'))
            fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.cstrategy, mode='lines', name='Contrarian Strategy'))
            fig.update_layout(
                height=500,
                xaxis_title='Date',
                yaxis_title='Cumulative Returns'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Trade summary
            if performance['trades_df'] is not None:
                st.subheader("Trade Summary")
                trades_df = performance['trades_df']
                st.dataframe(trades_df)
                st.markdown(get_table_download_link(trades_df, filename=f"{selected_pair}_contrarian_trades.csv", text="Download Trades Data"), unsafe_allow_html=True)
            
            # Results table
            st.subheader("Results Table")
            st.dataframe(backtest.data[["returns", "strategy", "creturns", "cstrategy"]].tail(10))
            st.markdown(get_table_download_link(backtest.data, filename=f"{selected_pair}_contrarian_strategy_results.csv"), unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"An error occurred in the Contrarian strategy: {e}")
            st.info("Try using a different window parameter or check if the data format is correct.")
    
    elif strategy_type == "Bollinger Bands Strategy":
        st.header("Bollinger Bands Strategy")
        
        # Bollinger parameters
        col1, col2 = st.columns(2)
        with col1:
            sma = st.slider("SMA Period", min_value=5, max_value=100, value=20, step=1)
        with col2:
            dev = st.slider("Standard Deviation Factor", min_value=1.0, max_value=3.0, value=2.0, step=0.1)
            
        try:
            # Prepare data for the selected pair
            df_pair = prepare_price_data(df, selected_pair)
            
            # Create backtest instance
            backtest = IterativeBacktest(
                symbol=selected_pair,
                start=str(df.index.min().date()),
                end=str(df.index.max().date()),
                amount=initial_capital,
                use_spread=use_spread
            )
            
            # Set the prepared data
            backtest.set_data(df_pair)
            
            # Run Bollinger Bands strategy
            performance = backtest.test_boll_strategy(sma, dev)
            
            # Calculate metrics
            metrics = calculate_metrics(backtest.data.returns, backtest.data.strategy)
            
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
            
            # Display final performance from backtest
            st.subheader("Backtest Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Initial Balance", f"${initial_capital:,.2f}")
            col2.metric("Final Balance", f"${performance['final_balance']:,.2f}")
            col3.metric("Total Trades", f"{performance['trades']}")
            
            # Plot Bollinger Bands and price
            st.subheader("Price and Bollinger Bands")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.price, mode='lines', name=f'{selected_pair}'))
            fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.SMA, mode='lines', name=f'SMA {sma}'))
            fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.Upper, mode='lines', name='Upper Band'))
            fig.add_trace(go.Scatter(x=backtest.data.index, y=backtest.data.Lower, mode='lines', name='Lower Band'))
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
                x=backtest.data.index, 
                y=backtest.data.position, 
                mode='lines', 
                name='Position',
                line=dict(color='blue', width=2)
            ))
            fig.