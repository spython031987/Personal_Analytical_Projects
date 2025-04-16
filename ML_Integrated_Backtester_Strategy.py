#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Backtester Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from itertools import product
import io
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Multi-Currency Trading Strategy Backtester", layout="wide")

# App title and description
st.title("Multi-Currency Trading Strategy Backtester")
st.markdown("""
This application allows you to backtest multiple trading strategies across different currency pairs:
- Simple Moving Average (SMA) crossover strategy
- Contrarian strategy
- Machine Learning (ML) strategy

Upload your own data or use the sample data.
""")

# Function to download dataframe as CSV
def get_table_download_link(df, filename="data.csv", text="Download CSV"):
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to prepare data for strategy testing
def prepare_price_data(data, pair):
    # Create a copy with only the selected pair, renamed to 'price'
    df_pair = pd.DataFrame(data[pair].copy()).rename(columns={pair: 'price'})
    return df_pair

# Function to test SMA strategy
def test_sma_strategy(data, pair, sma_s, sma_l):
    # Prepare data for the specified pair
    df_pair = prepare_price_data(data, pair)
    
    # Calculate returns
    df_pair["returns"] = np.log(df_pair.price.div(df_pair.price.shift(1)))
    
    # Calculate SMAs
    df_pair["SMA_S"] = df_pair.price.rolling(int(sma_s)).mean()
    df_pair["SMA_L"] = df_pair.price.rolling(int(sma_l)).mean()
    df_pair.dropna(inplace=True)
    
    # Calculate positions and strategy returns
    df_pair["position"] = np.where(df_pair["SMA_S"] > df_pair["SMA_L"], 1, -1)
    df_pair["strategy"] = df_pair.position.shift(1) * df_pair["returns"]
    df_pair.dropna(inplace=True)
    
    # Calculate cumulative returns using numpy for reliability
    returns_cumsum = df_pair["returns"].astype(float).cumsum().values
    strategy_cumsum = df_pair["strategy"].astype(float).cumsum().values
    
    # Apply exponential to cumulative sum using numpy
    df_pair["creturns"] = pd.Series(np.exp(returns_cumsum), index=df_pair.index)
    df_pair["cstrategy"] = pd.Series(np.exp(strategy_cumsum), index=df_pair.index)
    
    return df_pair

# Function to test Contrarian strategy
def test_contrarian_strategy(data, pair, window, tc):
    # Prepare data for the specified pair
    df_pair = prepare_price_data(data, pair)
    
    # Calculate returns
    df_pair["returns"] = np.log(df_pair.price.div(df_pair.price.shift(1)))
    df_pair.dropna(inplace=True)
    
    # Calculate positions - take position opposite to the recent price movement
    df_pair["position"] = -np.sign(df_pair["returns"].rolling(window).mean())
    df_pair["strategy"] = df_pair.position.shift(1) * df_pair["returns"]
    df_pair.dropna(inplace=True)
    
    # Calculate trading costs
    df_pair["trades"] = df_pair.position.diff().fillna(0).abs()
    df_pair["strategy"] = df_pair["strategy"] - df_pair["trades"] * tc
    
    # Calculate cumulative returns using numpy for reliability
    returns_cumsum = df_pair["returns"].astype(float).cumsum().values
    strategy_cumsum = df_pair["strategy"].astype(float).cumsum().values
    
    # Apply exponential to cumulative sum using numpy
    df_pair["creturns"] = pd.Series(np.exp(returns_cumsum), index=df_pair.index)
    df_pair["cstrategy"] = pd.Series(np.exp(strategy_cumsum), index=df_pair.index)
    
    return df_pair

# Function to test ML strategy
def test_ml_strategy(data, pair, lags, train_ratio, tc):
    # Prepare data for the specified pair
    df_pair = prepare_price_data(data, pair)
    
    # Calculate returns
    df_pair["returns"] = np.log(df_pair.price.div(df_pair.price.shift(1)))
    df_pair.dropna(inplace=True)
    
    # Feature engineering - create lag features
    feature_columns = []
    for lag in range(1, lags + 1):
        col = f"lag{lag}"
        df_pair[col] = df_pair["returns"].shift(lag)
        feature_columns.append(col)
        
    df_pair.dropna(inplace=True)
    
    # Split data into training and testing sets
    split_index = int(len(df_pair) * train_ratio)
    train_data = df_pair.iloc[:split_index].copy()
    test_data = df_pair.iloc[split_index:].copy()
    
    # Standardize features
    train_mean = train_data[feature_columns].mean()
    train_std = train_data[feature_columns].std()
    
    train_data[feature_columns] = (train_data[feature_columns] - train_mean) / train_std
    test_data[feature_columns] = (test_data[feature_columns] - train_mean) / train_std
    
    # Train ML model
    model = OneVsRestClassifier(LogisticRegression(C=1e6, max_iter=100000))
    model.fit(train_data[feature_columns], np.sign(train_data["returns"]))
    
    # Make predictions on test set
    test_data["position"] = model.predict(test_data[feature_columns])
    test_data["strategy"] = test_data["position"].shift(1) * test_data["returns"]
    test_data.dropna(inplace=True)
    
    # Calculate trading costs
    test_data["trades"] = test_data["position"].diff().fillna(0).abs()
    test_data["strategy"] = test_data["strategy"] - test_data["trades"] * tc
    
    # Calculate cumulative returns
    returns_cumsum = test_data["returns"].astype(float).cumsum().values
    strategy_cumsum = test_data["strategy"].astype(float).cumsum().values
    
    test_data["creturns"] = pd.Series(np.exp(returns_cumsum), index=test_data.index)
    test_data["cstrategy"] = pd.Series(np.exp(strategy_cumsum), index=test_data.index)
    
    # Calculate confusion matrix for model evaluation
    y_true = np.sign(test_data["returns"]).astype(int)
    y_pred = test_data["position"].astype(int)
    
    # Filter out zeros (flat market signals) for analysis
    non_zero_indices = (y_true != 0)
    y_true_filtered = y_true[non_zero_indices]
    y_pred_filtered = y_pred[non_zero_indices]
    
    # Calculate confusion matrix only if we have non-zero values
    if len(y_true_filtered) > 0:
        conf_matrix = confusion_matrix(y_true_filtered, y_pred_filtered)
    else:
        conf_matrix = np.array([[0, 0], [0, 0]])
    
    # Additional model information
    feature_importance = np.abs(model.estimators_[0].coef_[0])
    feature_names = feature_columns
    
    return test_data, conf_matrix, feature_importance, feature_names, model

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

# Function to get numerical value from metric string for cross-pair analysis
def get_metric_value(metric_str):
    try:
        # Remove % sign and convert to float
        return float(metric_str.strip('%')) / 100
    except:
        # If it's not a percentage, just convert to float
        return float(metric_str)

# Function to plot confusion matrix using plotly
def plot_confusion_matrix(conf_matrix):
    # Create a plotly figure for confusion matrix
    labels = ["-1 (Short)", "1 (Long)"]
    
    fig = px.imshow(
        conf_matrix,
        x=labels,
        y=labels,
        color_continuous_scale="Viridis",
        labels=dict(x="Predicted", y="Actual", color="Count")
    )
    
    # Add text annotations
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            fig.add_annotation(
                x=j, y=i,
                text=str(conf_matrix[i, j]),
                showarrow=False,
                font=dict(color="white" if conf_matrix[i, j] > conf_matrix.max()/2 else "black")
            )
    
    fig.update_layout(
        title="Confusion Matrix",
        height=400,
        width=400
    )
    
    return fig

# Sidebar for inputs
st.sidebar.header("Settings")

# Data upload or use sample data
data_option = st.sidebar.radio("Data Source", ["Use sample data", "Upload your own data"])

df = None
available_pairs = []

if data_option == "Use sample data":
    try:
        # Create sample multi-currency data (simulating the structure of intraday_pairs.csv)
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
            for _ in range(1, len(dates)):
                price *= (1 + np.random.normal(0.0001, 0.005))  # Small daily changes
                prices.append(price)
            sample_data[pair] = prices
        
        df = pd.DataFrame(sample_data)
        df.set_index('time', inplace=True)
        
        available_pairs = [col for col in df.columns]
        st.sidebar.success(f"Sample data loaded successfully with {len(available_pairs)} currency pairs")
    except Exception as e:
        st.sidebar.error(f"Error loading sample data: {e}")
else:
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file with currency price data", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=["time"], index_col="time")
            
            # Get available currency pairs (all columns except the index)
            available_pairs = [col for col in df.columns]
            
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
        ["Data Overview", "Buy and Hold", "SMA Crossover", "Contrarian Strategy", 
         "ML Strategy", "SMA Parameter Optimization", "Contrarian Parameter Optimization",
         "ML Parameter Optimization", "Cross-Pair Analysis"]
    )
    
    # For Cross-Pair Analysis, allow selection of multiple pairs for comparison
    if strategy_type == "Cross-Pair Analysis":
        comparison_pairs = st.sidebar.multiselect(
            "Select Pairs to Compare",
            available_pairs,
            default=[available_pairs[0]] if available_pairs else []
        )
    else:
        comparison_pairs = [selected_pair]
    
    # Set transaction costs
    if "Contrarian" in strategy_type or "ML" in strategy_type:
        tc = st.sidebar.slider("Transaction Costs (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01) / 100
    else:
        tc = 0.001  # Default transaction cost

# Main content area
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
        returns = np.log(df[selected_pair].div(df[selected_pair].shift(1)))
        returns.dropna(inplace=True)
        
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
            
            # Calculate returns
            df_pair["returns"] = np.log(df_pair.price.div(df_pair.price.shift(1)))
            df_pair.dropna(inplace=True)
            
            # Calculate cumulative returns
            df_pair["creturns"] = df_pair["returns"].astype(float).cumsum().apply(np.exp)
            
            # Calculate drawdown using numpy directly
            cummax_array = np.maximum.accumulate(df_pair["creturns"].values)
            drawdown_array = (df_pair["creturns"].values - cummax_array) / np.where(cummax_array > 0, cummax_array, 1)
            df_pair["cummax"] = pd.Series(cummax_array, index=df_pair.index)
            df_pair["drawdown"] = pd.Series(drawdown_array, index=df_pair.index)
            
            # Performance metrics
            metrics = calculate_metrics(df_pair.returns)
            
            # Display metrics
            st.subheader("Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Strategy Ann. Return", metrics["Strategy Annualized Return"])
            col2.metric("Strategy Ann. Risk", metrics["Strategy Annualized Risk"])
            col3.metric("Strategy Sharpe Ratio", metrics["Strategy Sharpe Ratio"])
            col4.metric("Maximum Drawdown", metrics["Maximum Drawdown"])
            
            # Plot positions
            st.subheader("Trading Positions")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_ml.index, 
                y=df_ml.position, 
                mode='lines', 
                name='Position',
                line=dict(color='blue', width=2)
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
            fig.add_trace(go.Scatter(x=df_ml.index, y=df_ml.creturns, mode='lines', name='Buy and Hold'))
            fig.add_trace(go.Scatter(x=df_ml.index, y=df_ml.cstrategy, mode='lines', name='ML Strategy'))
            fig.update_layout(
                height=500,
                xaxis_title='Date',
                yaxis_title='Cumulative Returns'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Trading activity
            st.subheader("Trading Activity")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_ml.index, 
                y=df_ml.trades.cumsum(), 
                mode='lines', 
                name='Cumulative Trades',
                line=dict(color='darkblue', width=2)
            ))
            fig.update_layout(
                height=300,
                xaxis_title='Date',
                yaxis_title='Total Number of Trades'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Model Evaluation
            st.subheader("Model Evaluation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Plot confusion matrix
                st.markdown("**Confusion Matrix**")
                st.markdown("Evaluates prediction accuracy by comparing predicted vs actual market directions")
                conf_fig = plot_confusion_matrix(conf_matrix)
                st.plotly_chart(conf_fig)
                
            with col2:
                # Plot feature importance
                st.markdown("**Feature Importance**")
                st.markdown("Shows which lags have the most influence on predictions")
                
                fig = px.bar(
                    x=feature_names,
                    y=feature_importance,
                    title="Feature Importance",
                    labels={"x": "Feature", "y": "Importance"}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig)
            
            # Model prediction analysis
            st.subheader("Model Prediction Analysis")
            
            # Calculate prediction accuracy metrics
            prediction_correct = (np.sign(df_ml["returns"]) == df_ml["position"].shift(1)).astype(int)
            accuracy = prediction_correct.mean()
            
            # Profitable trades analysis
            profitable_trades = (df_ml["strategy"] > 0).astype(int)
            profitable_pct = profitable_trades.mean()
            
            col1, col2 = st.columns(2)
            col1.metric("Direction Prediction Accuracy", f"{accuracy:.2%}")
            col2.metric("Profitable Trades", f"{profitable_pct:.2%}")
            
            # Plot prediction accuracy over time
            st.markdown("**Prediction Accuracy Over Time**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_ml.index,
                y=prediction_correct.rolling(50).mean(),
                mode='lines',
                name='Rolling 50-period Accuracy',
                line=dict(color='green', width=2)
            ))
            fig.update_layout(
                height=300,
                xaxis_title='Date',
                yaxis_title='Accuracy (50-period rolling)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.subheader("Results Table")
            st.dataframe(df_ml[["returns", "position", "strategy", "trades", "creturns", "cstrategy"]].tail(10))
            st.markdown(get_table_download_link(df_ml, filename=f"{selected_pair}_ml_strategy_results.csv"), unsafe_allow_html=True)("Total Return", metrics["Total Return"])
            col2.metric("Annualized Return", metrics["Annualized Return"])
            col3.metric("Annualized Risk", metrics["Annualized Risk"])
            col4.metric("Sharpe Ratio", metrics["Sharpe Ratio"])
            
            st.metric("Maximum Drawdown", metrics["Maximum Drawdown"])
            
            # Plot cumulative returns
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
                # Run strategy
                df_sma = test_sma_strategy(df, selected_pair, sma_s, sma_l)
                
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
                fig.add_trace(go.Scatter(x=df_sma.index, y=df_sma.price, mode='lines', name=f'{selected_pair}'))
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
                st.markdown(get_table_download_link(df_sma, filename=f"{selected_pair}_sma_strategy_results.csv"), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred in the SMA Crossover strategy: {e}")
                st.info("Try using different SMA periods or check if the data format is correct.")

    elif strategy_type == "Contrarian Strategy":
        st.header("Contrarian Strategy")
        
        # Contrarian parameters
        window = st.slider("Look-back Window", min_value=1, max_value=100, value=10, step=1)
        
        try:
            # Run strategy
            df_con = test_contrarian_strategy(df, selected_pair, window, tc)
            
            # Calculate metrics
            metrics = calculate_metrics(df_con.returns, df_con.strategy)
            
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
            
            # Plot positions
            st.subheader("Trading Positions")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_con.index, 
                y=df_con.position, 
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
                x=df_con.index, 
                y=df_con["returns"].rolling(window).mean(), 
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
            fig.add_trace(go.Scatter(x=df_con.index, y=df_con.creturns, mode='lines', name='Buy and Hold'))
            fig.add_trace(go.Scatter(x=df_con.index, y=df_con.cstrategy, mode='lines', name='Contrarian Strategy'))
            fig.update_layout(
                height=500,
                xaxis_title='Date',
                yaxis_title='Cumulative Returns'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Trading activity
            st.subheader("Trading Activity")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_con.index, 
                y=df_con.trades.cumsum(), 
                mode='lines', 
                name='Cumulative Trades',
                line=dict(color='darkorange', width=2)
            ))
            fig.update_layout(
                height=300,
                xaxis_title='Date',
                yaxis_title='Total Number of Trades'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.subheader("Results Table")
            st.dataframe(df_con[["returns", "strategy", "trades", "creturns", "cstrategy"]].tail(10))
            st.markdown(get_table_download_link(df_con, filename=f"{selected_pair}_contrarian_strategy_results.csv"), unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred in the Contrarian strategy: {e}")
            st.info("Try using a different window parameter or check if the data format is correct.")

    # ML Strategy Section
    elif strategy_type == "ML Strategy":
        st.header("Machine Learning Strategy")
        
        # ML parameters
        col1, col2 = st.columns(2)
        with col1:
            lags = st.slider("Number of Lags (Features)", min_value=1, max_value=20, value=5, step=1)
            train_ratio = st.slider("Training Set Ratio", min_value=0.1, max_value=0.9, value=0.7, step=0.05)
        
        try:
            # Run strategy
            df_ml, conf_matrix, feature_importance, feature_names, ml_model = test_ml_strategy(df, selected_pair, lags, train_ratio, tc)
            
            # Calculate metrics
            metrics = calculate_metrics(df_ml.returns, df_ml.strategy)
            
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
            
            # Plot positions
            st.subheader("Trading Positions")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_ml.index, 
                y=df_ml.position, 
                mode='lines', 
                name='Position',
                line=dict(color='blue', width=2)
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
            fig.add_trace(go.Scatter(x=df_ml.index, y=df_ml.creturns, mode='lines', name='Buy and Hold'))
            fig.add_trace(go.Scatter(x=df_ml.index, y=df_ml.cstrategy, mode='lines', name='ML Strategy'))
            fig.update_layout(
                height=500,
                xaxis_title='Date',
                yaxis_title='Cumulative Returns'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Trading activity
            st.subheader("Trading Activity")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_ml.index, 
                y=df_ml.trades.cumsum(), 
                mode='lines', 
                name='Cumulative Trades',
                line=dict(color='darkblue', width=2)
            ))
            fig.update_layout(
                height=300,
                xaxis_title='Date',
                yaxis_title='Total Number of Trades'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Model Evaluation
            st.subheader("Model Evaluation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Plot confusion matrix
                st.markdown("**Confusion Matrix**")
                st.markdown("Evaluates prediction accuracy by comparing predicted vs actual market directions")
                conf_fig = plot_confusion_matrix(conf_matrix)
                st.plotly_chart(conf_fig)
                
            with col2:
                # Plot feature importance
                st.markdown("**Feature Importance**")
                st.markdown("Shows which lags have the most influence on predictions")
                
                fig = px.bar(
                    x=feature_names,
                    y=feature_importance,
                    title="Feature Importance",
                    labels={"x": "Feature", "y": "Importance"}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig)
            
            # Model prediction analysis
            st.subheader("Model Prediction Analysis")
            
            # Calculate prediction accuracy metrics
            prediction_correct = (np.sign(df_ml["returns"]) == df_ml["position"].shift(1)).astype(int)
            accuracy = prediction_correct.mean()
            
            # Profitable trades analysis
            profitable_trades = (df_ml["strategy"] > 0).astype(int)
            profitable_pct = profitable_trades.mean()
            
            col1, col2 = st.columns(2)
            col1.metric("Direction Prediction Accuracy", f"{accuracy:.2%}")
            col2.metric("Profitable Trades", f"{profitable_pct:.2%}")
            
            # Plot prediction accuracy over time
            st.markdown("**Prediction Accuracy Over Time**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_ml.index,
                y=prediction_correct.rolling(50).mean(),
                mode='lines',
                name='Rolling 50-period Accuracy',
                line=dict(color='green', width=2)
            ))
            fig.update_layout(
                height=300,
                xaxis_title='Date',
                yaxis_title='Accuracy (50-period rolling)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.subheader("Results Table")
            st.dataframe(df_ml[["returns", "position", "strategy", "trades", "creturns", "cstrategy"]].tail(10))
            st.markdown(get_table_download_link(df_ml, filename=f"{selected_pair}_ml_strategy_results.csv"), unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred in the ML strategy: {e}")
            st.info("Try using different lag periods or check if the data format is correct.")
