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
         "ML Parameter Optimization", "Cross-Pair Analysis"]  # Added ML Strategy option
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
            col1.metric("Total Return", metrics["Total Return"])
            col2.metric("Annualized Return", metrics["Annualized Return"])
            col3.metric("Annualized Risk", metrics["Annualized Risk"])
            col4.metric("Sharpe Ratio", metrics["Sharpe Ratio"])
            
            st.metric("Maximum Drawdown", metrics["Maximum Drawdown"])
            
            # Plot cumulative returns
            st.subheader("Cumulative Returns")
            fig = go.Figure(data=go.Heatmap(
                z=pivot_data.values,
                x=[f"{tc*100:.2f}%" for tc in pivot_data.columns],
                y=pivot_data.index,
                colorscale='Viridis',
                colorbar=dict(title='Performance')
            ))
            
            fig.update_layout(
                title=f"ML Parameter Optimization Heatmap for {selected_pair}",
                xaxis_title="Transaction Costs (%)",
                yaxis_title="Number of Lags",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display accuracy heatmap
            st.subheader("Prediction Accuracy Heatmap")
            
            pivot_acc = results_df.pivot_table(
                index="Lags", 
                columns="TC", 
                values="accuracy",
                aggfunc='first'
            )
            
            fig_acc = go.Figure(data=go.Heatmap(
                z=pivot_acc.values,
                x=[f"{tc*100:.2f}%" for tc in pivot_acc.columns],
                y=pivot_acc.index,
                colorscale='RdYlGn',
                colorbar=dict(title='Accuracy')
            ))
            
            fig_acc.update_layout(
                title="Prediction Accuracy by Parameter Combination",
                xaxis_title="Transaction Costs (%)",
                yaxis_title="Number of Lags",
                height=500
            )
            
            st.plotly_chart(fig_acc, use_container_width=True)
            
            # Option to use best parameters
            best_params = results_df.loc[results_df["performance"].idxmax()]
            
            st.subheader("Best Parameters")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Lags", int(best_params["Lags"]))
            col2.metric("Transaction Costs", f"{best_params['TC']*100:.2f}%")
            col3.metric("Performance", f"{best_params['performance'] - 1:.2%}")
            col4.metric("Accuracy", f"{best_params['accuracy']*100:.2f}%")
            
            st.info(f"💡 Tip: You can use these optimal parameters (Lags: {int(best_params['Lags'])}, TC: {best_params['TC']*100:.2f}%) in the ML Strategy section.")
            
    elif strategy_type == "Cross-Pair Analysis":
        st.header("Cross-Pair Analysis")
        
        if len(comparison_pairs) < 1:
            st.warning("Please select at least one currency pair to analyze.")
        else:
            # Select analysis type
            analysis_type = st.radio(
                "Analysis Type",
                ["Price Comparison", "Returns Comparison", "Strategy Performance"]
            )
            
            if analysis_type == "Price Comparison":
                st.subheader("Price Comparison")
                
                # Normalize prices for comparison
                fig = go.Figure()
                
                for pair in comparison_pairs:
                    # Normalize to starting at 100
                    normalized_price = df[pair] / df[pair].iloc[0] * 100
                    fig.add_trace(go.Scatter(
                        x=df.index, 
                        y=normalized_price, 
                        mode='lines', 
                        name=f'{pair}'
                    ))
                
                fig.update_layout(
                    height=600,
                    title="Normalized Price Comparison (Base=100)",
                    xaxis_title='Date',
                    yaxis_title='Normalized Price'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation matrix
                if len(comparison_pairs) > 1:
                    st.subheader("Price Correlation Matrix")
                    corr_matrix = df[comparison_pairs].corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1
                    )
                    fig.update_layout(
                        height=500,
                        title="Price Correlation Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Returns Comparison":
                st.subheader("Returns Comparison")
                
                # Calculate returns for each pair
                returns_df = pd.DataFrame(index=df.index)
                for pair in comparison_pairs:
                    returns_df[pair] = np.log(df[pair].div(df[pair].shift(1)))
                
                returns_df.dropna(inplace=True)
                
                # Plot returns comparison
                fig = go.Figure()
                for pair in comparison_pairs:
                    fig.add_trace(go.Scatter(
                        x=returns_df.index, 
                        y=returns_df[pair].cumsum().apply(np.exp), 
                        mode='lines', 
                        name=f'{pair}'
                    ))
                
                fig.update_layout(
                    height=600,
                    title="Cumulative Returns Comparison",
                    xaxis_title='Date',
                    yaxis_title='Cumulative Returns'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Volatility comparison
                if len(comparison_pairs) > 0:
                    st.subheader("Volatility Comparison")
                    # Calculate rolling standard deviation (annualized)
                    window = st.slider("Rolling Window (days)", min_value=5, max_value=252, value=30, step=5)
                    
                    vol_df = pd.DataFrame(index=returns_df.index)
                    for pair in comparison_pairs:
                        vol_df[pair] = returns_df[pair].rolling(window).std() * np.sqrt(252)
                    
                    # Plot volatility
                    fig = go.Figure()
                    for pair in comparison_pairs:
                        fig.add_trace(go.Scatter(
                            x=vol_df.index, 
                            y=vol_df[pair], 
                            mode='lines', 
                            name=f'{pair}'
                        ))
                    
                    fig.update_layout(
                        height=500,
                        title=f"Rolling {window}-Day Annualized Volatility",
                        xaxis_title='Date',
                        yaxis_title='Volatility'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Returns correlation matrix
                    st.subheader("Returns Correlation Matrix")
                    returns_corr = returns_df.corr()
                    
                    fig = px.imshow(
                        returns_corr,
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1
                    )
                    fig.update_layout(
                        height=500,
                        title="Returns Correlation Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistical summary
                    st.subheader("Returns Statistics")
                    
                    # Create a summary table
                    summary_data = []
                    for pair in comparison_pairs:
                        pair_returns = returns_df[pair]
                        
                        ann_return = pair_returns.mean() * 252
                        ann_vol = pair_returns.std() * np.sqrt(252)
                        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
                        
                        summary_data.append({
                            "Pair": pair,
                            "Mean Daily Return": pair_returns.mean(),
                            "Annualized Return": ann_return,
                            "Annualized Volatility": ann_vol,
                            "Sharpe Ratio": sharpe,
                            "Min Return": pair_returns.min(),
                            "Max Return": pair_returns.max()
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df.style.format({
                        "Mean Daily Return": "{:.6f}",
                        "Annualized Return": "{:.2%}",
                        "Annualized Volatility": "{:.2%}",
                        "Sharpe Ratio": "{:.2f}",
                        "Min Return": "{:.2%}",
                        "Max Return": "{:.2%}"
                    }))
            
            elif analysis_type == "Strategy Performance":
                st.subheader("Strategy Performance Comparison")
                
                # Strategy selection for comparison
                strategy_selection = st.radio(
                    "Strategy Type",
                    ["SMA Crossover", "Contrarian Strategy", "ML Strategy"]  # Added ML Strategy option
                )
                
                if strategy_selection == "SMA Crossover":
                    # SMA parameters
                    col1, col2 = st.columns(2)
                    with col1:
                        sma_s = st.slider("Short SMA Period", min_value=5, max_value=100, value=50, step=1)
                    with col2:
                        sma_l = st.slider("Long SMA Period", min_value=50, max_value=300, value=200, step=1)
                    
                    if sma_s >= sma_l:
                        st.error("Short SMA period must be less than Long SMA period")
                    else:
                        try:
                            # Run strategy for each pair
                            results_data = []
                            
                            # Plot for cumulative strategy returns
                            fig = go.Figure()
                            
                            for pair in comparison_pairs:
                                # Test SMA strategy
                                df_sma = test_sma_strategy(df, pair, sma_s, sma_l)
                                
                                # Add to plot
                                fig.add_trace(go.Scatter(
                                    x=df_sma.index, 
                                    y=df_sma.cstrategy, 
                                    mode='lines', 
                                    name=f'{pair} Strategy'
                                ))
                                
                                # Calculate metrics
                                metrics = calculate_metrics(df_sma.returns, df_sma.strategy)
                                
                                # Store results
                                results_data.append({
                                    "Pair": pair,
                                    "Buy & Hold Return": metrics["Total Return"],
                                    "Strategy Return": metrics["Strategy Return"],
                                    "Outperformance": metrics["Outperformance"],
                                    "Strategy Ann. Return": metrics["Strategy Annualized Return"],
                                    "Strategy Ann. Risk": metrics["Strategy Annualized Risk"],
                                    "Sharpe Ratio": metrics["Strategy Sharpe Ratio"],
                                    "Maximum Drawdown": metrics["Maximum Drawdown"]
                                })
                            
                            # Display the cumulative returns plot
                            fig.update_layout(
                                height=600,
                                title=f"SMA({sma_s}, {sma_l}) Strategy Performance by Currency Pair",
                                xaxis_title="Date",
                                yaxis_title="Cumulative Returns"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Create results DataFrame and sort by performance
                            results_df = pd.DataFrame(results_data)
                            
                            # Convert percentage strings to floats for sorting
                            # Create a new DataFrame for ranking with numerical values
                            ranking_df = pd.DataFrame(index=results_df.index)
                            for pair in results_df["Pair"]:
                                pair_row = results_df[results_df["Pair"] == pair].iloc[0]
                                ranking_df.loc[pair_row.name, "Pair"] = pair
                                ranking_df.loc[pair_row.name, "Strategy Return"] = get_metric_value(pair_row["Strategy Return"])
                                ranking_df.loc[pair_row.name, "Sharpe Ratio"] = float(pair_row["Sharpe Ratio"])
                            
                            # Sort by strategy return
                            ranking_df = ranking_df.sort_values("Strategy Return", ascending=False)
                            
                            # Reorder the original DataFrame based on ranking
                            sorted_results = pd.DataFrame()
                            for idx in ranking_df.index:
                                sorted_results = pd.concat([sorted_results, results_df.iloc[[idx]]])
                            
                            # Display the metrics table
                            st.subheader("Performance Metrics by Currency Pair")
                            st.dataframe(sorted_results)
                            
                            # Bar chart comparing key metrics
                            st.subheader("Key Metrics Comparison")
                            
                            # Bar chart for Strategy Returns
                            fig = px.bar(
                                ranking_df,
                                x="Pair",
                                y="Strategy Return",
                                title="Strategy Returns by Currency Pair",
                                color="Strategy Return",
                                color_continuous_scale="RdYlGn"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Bar chart for Sharpe Ratios
                            fig = px.bar(
                                ranking_df,
                                x="Pair",
                                y="Sharpe Ratio",
                                title="Sharpe Ratios by Currency Pair",
                                color="Sharpe Ratio",
                                color_continuous_scale="RdYlGn"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"An error occurred in the SMA strategy comparison: {e}")
                
                elif strategy_selection == "Contrarian Strategy":
                    # Contrarian parameters
                    window = st.slider("Look-back Window", min_value=1, max_value=100, value=10, step=1)
                    tc = st.slider("Transaction Costs (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01) / 100
                    
                    try:
                        # Run strategy for each pair
                        results_data = []
                        
                        # Plot for cumulative strategy returns
                        fig = go.Figure()
                        
                        for pair in comparison_pairs:
                            # Test contrarian strategy
                            df_con = test_contrarian_strategy(df, pair, window, tc)
                            
                            # Add to plot
                            fig.add_trace(go.Scatter(
                                x=df_con.index, 
                                y=df_con.cstrategy, 
                                mode='lines', 
                                name=f'{pair} Strategy'
                            ))
                            
                            # Calculate metrics
                            metrics = calculate_metrics(df_con.returns, df_con.strategy)
                            
                            # Calculate total trades
                            total_trades = df_con["trades"].sum()
                            
                            # Store results
                            results_data.append({
                                "Pair": pair,
                                "Buy & Hold Return": metrics["Total Return"],
                                "Strategy Return": metrics["Strategy Return"],
                                "Outperformance": metrics["Outperformance"],
                                "Strategy Ann. Return": metrics["Strategy Annualized Return"],
                                "Strategy Ann. Risk": metrics["Strategy Annualized Risk"],
                                "Sharpe Ratio": metrics["Strategy Sharpe Ratio"],
                                "Maximum Drawdown": metrics["Maximum Drawdown"],
                                "Total Trades": int(total_trades)
                            })
                        
                        # Display the cumulative returns plot
                        fig.update_layout(
                            height=600,
                            title=f"Contrarian Strategy (Window={window}, TC={tc*100:.2f}%) Performance by Currency Pair",
                            xaxis_title="Date",
                            yaxis_title="Cumulative Returns"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create results DataFrame and sort by performance
                        results_df = pd.DataFrame(results_data)
                        
                        # Convert percentage strings to floats for sorting
                        # Create a new DataFrame for ranking with numerical values
                        ranking_df = pd.DataFrame(index=results_df.index)
                        for pair in results_df["Pair"]:
                            pair_row = results_df[results_df["Pair"] == pair].iloc[0]
                            ranking_df.loc[pair_row.name, "Pair"] = pair
                            ranking_df.loc[pair_row.name, "Strategy Return"] = get_metric_value(pair_row["Strategy Return"])
                            ranking_df.loc[pair_row.name, "Sharpe Ratio"] = float(pair_row["Sharpe Ratio"])
                            ranking_df.loc[pair_row.name, "Total Trades"] = pair_row["Total Trades"]
                        
                        # Sort by strategy return
                        ranking_df = ranking_df.sort_values("Strategy Return", ascending=False)
                        
                        # Reorder the original DataFrame based on ranking
                        sorted_results = pd.DataFrame()
                        for idx in ranking_df.index:
                            sorted_results = pd.concat([sorted_results, results_df.iloc[[idx]]])
                        
                        # Display the metrics table
                        st.subheader("Performance Metrics by Currency Pair")
                        st.dataframe(sorted_results)
                        
                        # Bar charts for key metrics
                        st.subheader("Key Metrics Comparison")
                        
                        # Bar chart for Strategy Returns
                        fig = px.bar(
                            ranking_df,
                            x="Pair",
                            y="Strategy Return",
                            title="Strategy Returns by Currency Pair",
                            color="Strategy Return",
                            color_continuous_scale="RdYlGn"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Bar chart for Total Trades
                        fig = px.bar(
                            ranking_df,
                            x="Pair",
                            y="Total Trades",
                            title="Trading Activity by Currency Pair",
                            color="Total Trades"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"An error occurred in the Contrarian strategy comparison: {e}")
                        
                # New ML strategy cross-pair analysis
                elif strategy_selection == "ML Strategy":
                    # ML parameters
                    col1, col2 = st.columns(2)
                    with col1:
                        lags = st.slider("Number of Lags (Features)", min_value=1, max_value=20, value=5, step=1)
                        train_ratio = st.slider("Training Set Ratio", min_value=0.3, max_value=0.9, value=0.7, step=0.05)
                    with col2:
                        tc = st.slider("Transaction Costs (%)", min_value=0.0, max_value=1.0, value=0.1, step=0.01) / 100
                    
                    try:
                        # Run strategy for each pair
                        results_data = []
                        
                        # Plot for cumulative strategy returns
                        fig = go.Figure()
                        
                        # Container for accuracy data
                        accuracy_data = {}
                        
                        for pair in comparison_pairs:
                            # Test ML strategy
                            df_ml, conf_matrix, feature_importance, feature_names, _ = test_ml_strategy(df, pair, lags, train_ratio, tc)
                            
                            # Add to plot
                            fig.add_trace(go.Scatter(
                                x=df_ml.index, 
                                y=df_ml.cstrategy, 
                                mode='lines', 
                                name=f'{pair} Strategy'
                            ))
                            
                            # Calculate metrics
                            metrics = calculate_metrics(df_ml.returns, df_ml.strategy)
                            
                            # Calculate trading statistics
                            total_trades = df_ml["trades"].sum()
                            accuracy = (np.sign(df_ml["returns"]) == df_ml["position"].shift(1)).astype(int).mean()
                            
                            # Store results
                            results_data.append({
                                "Pair": pair,
                                "Buy & Hold Return": metrics["Total Return"],
                                "Strategy Return": metrics["Strategy Return"],
                                "Outperformance": metrics["Outperformance"],
                                "Strategy Ann. Return": metrics["Strategy Annualized Return"],
                                "Strategy Ann. Risk": metrics["Strategy Annualized Risk"],
                                "Sharpe Ratio": metrics["Strategy Sharpe Ratio"],
                                "Maximum Drawdown": metrics["Maximum Drawdown"],
                                "Total Trades": int(total_trades),
                                "Prediction Accuracy": f"{accuracy:.2%}"
                            })
                            
                            # Store accuracy data for separate visualization
                            accuracy_data[pair] = accuracy
                        
                        # Display the cumulative returns plot
                        fig.update_layout(
                            height=600,
                            title=f"ML Strategy (Lags={lags}, TC={tc*100:.2f}%) Performance by Currency Pair",
                            xaxis_title="Date",
                            yaxis_title="Cumulative Returns"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create results DataFrame and sort by performance
                        results_df = pd.DataFrame(results_data)
                        
                        # Convert percentage strings to floats for sorting
                        ranking_df = pd.DataFrame(index=results_df.index)
                        for pair in results_df["Pair"]:
                            pair_row = results_df[results_df["Pair"] == pair].iloc[0]
                            ranking_df.loc[pair_row.name, "Pair"] = pair
                            ranking_df.loc[pair_row.name, "Strategy Return"] = get_metric_value(pair_row["Strategy Return"])
                            ranking_df.loc[pair_row.name, "Sharpe Ratio"] = float(pair_row["Sharpe Ratio"])
                            ranking_df.loc[pair_row.name, "Prediction Accuracy"] = get_metric_value(pair_row["Prediction Accuracy"])
                        
                        # Sort by strategy return
                        ranking_df = ranking_df.sort_values("Strategy Return", ascending=False)
                        
                        # Reorder the original DataFrame based on ranking
                        sorted_results = pd.DataFrame()
                        for idx in ranking_df.index:
                            sorted_results = pd.concat([sorted_results, results_df.iloc[[idx]]])
                        
                        # Display the metrics table
                        st.subheader("Performance Metrics by Currency Pair")
                        st.dataframe(sorted_results)
                        
                        # Bar charts for key metrics
                        st.subheader("Key Metrics Comparison")
                        
                        # Bar chart for Strategy Returns
                        fig = px.bar(
                            ranking_df,
                            x="Pair",
                            y="Strategy Return",
                            title="Strategy Returns by Currency Pair",
                            color="Strategy Return",
                            color_continuous_scale="RdYlGn"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Bar chart for Prediction Accuracy
                        fig = px.bar(
                            ranking_df,
                            x="Pair",
                            y="Prediction Accuracy",
                            title="Prediction Accuracy by Currency Pair",
                            color="Prediction Accuracy",
                            color_continuous_scale="RdYlGn"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Correlation between accuracy and returns
                        st.subheader("Accuracy vs Returns Analysis")
                        
                        correlation = np.corrcoef(
                            ranking_df["Prediction Accuracy"], 
                            ranking_df["Strategy Return"]
                        )[0, 1]
                        
                        st.write(f"Correlation between prediction accuracy and strategy returns: **{correlation:.4f}**")
                        
                        # Scatter plot
                        fig = px.scatter(
                            ranking_df,
                            x="Prediction Accuracy",
                            y="Strategy Return",
                            text="Pair",
                            title="Prediction Accuracy vs Strategy Returns",
                            color="Sharpe Ratio",
                            color_continuous_scale="RdYlGn"
                        )
                        fig.update_traces(textposition='top center')
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"An error occurred in the ML strategy comparison: {e}")
else:
    st.warning("Please upload data or use the sample data to proceed with the analysis.")Figure()
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

    # ML Strategy Section - New Addition
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
            st.info("Try using different parameters or check if the data format is correct.")
            
    elif strategy_type == "SMA Parameter Optimization":
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
                        df_test = test_sma_strategy(df, selected_pair, s_sma, l_sma)
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
                st.markdown(get_table_download_link(results_df, f"{selected_pair}_sma_optimization_results.csv"), unsafe_allow_html=True)
                
                # Create heatmap for visualization using Plotly
                st.subheader("Performance Heatmap")
                
                pivot_data = results_df.pivot_table(
                    index="SMA_S", 
                    columns="SMA_L", 
                    values="performance",
                    aggfunc='first'
                )
                
                fig = go.Figure(data=go.Heatmap(
                    z=pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    colorscale='Viridis',
                    colorbar=dict(title='Performance')
                ))
                
                fig.update_layout(
                    title=f"SMA Parameter Optimization Heatmap for {selected_pair}",
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
    
    elif strategy_type == "Contrarian Parameter Optimization":
        st.header("Contrarian Parameter Optimization")
        
        # Parameter ranges
        st.subheader("Set Parameter Ranges")
        col1, col2 = st.columns(2)
        with col1:
            window_min = st.number_input("Min Window", min_value=1, max_value=50, value=1, step=1)
            window_max = st.number_input("Max Window", min_value=5, max_value=100, value=30, step=1)
            window_step = st.number_input("Window Step", min_value=1, max_value=5, value=1, step=1)
        
        with col2:
            tc_values = st.slider("Transaction Costs Range (%)", 
                                 min_value=0.0, 
                                 max_value=1.0, 
                                 value=(0.1, 0.5), 
                                 step=0.1)
            tc_min, tc_max = tc_values
            tc_step = st.number_input("TC Step (%)", min_value=0.05, max_value=0.2, value=0.1, step=0.05)
            
        # Generate parameter ranges
        window_range = range(window_min, window_max + 1, window_step)
        tc_range = np.arange(tc_min/100, tc_max/100 + tc_step/100, tc_step/100)  # Convert to decimal
        
        total_combinations = len(list(window_range)) * len(tc_range)
        
        st.write(f"Total combinations to test: {total_combinations}")
        
        if st.button("Run Optimization"):
            if total_combinations > 100:
                st.warning(f"Testing {total_combinations} combinations may take a while. Consider reducing the parameter ranges.")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Generate combinations
            combinations = list(product(window_range, tc_range))
            
            # Initialize results list
            results = []
            
            # Run tests for each combination
            for i, (window, tc_val) in enumerate(combinations):
                # Update progress
                progress = (i + 1) / total_combinations
                progress_bar.progress(progress)
                status_text.text(f"Testing combination {i+1}/{total_combinations}: Window={window}, TC={tc_val:.4f}")
                
                try:
                    # Test strategy
                    df_test = test_contrarian_strategy(df, selected_pair, window, tc_val)
                    performance = np.exp(df_test["strategy"].sum())
                    
                    # Calculate metrics with safety checks
                    ann_return = df_test["strategy"].mean() * 252
                    ann_risk = df_test["strategy"].std() * np.sqrt(252)
                    sharpe = (ann_return / ann_risk) if ann_risk > 0 else 0
                    
                    # Calculate number of trades
                    total_trades = df_test["trades"].sum()
                    
                    # Store results
                    results.append({
                        "Window": window,
                        "TC": tc_val,
                        "performance": performance,
                        "annualized_return": ann_return,
                        "annualized_risk": ann_risk,
                        "sharpe": sharpe,
                        "total_trades": total_trades
                    })
                except Exception as e:
                    st.error(f"Error with Window={window}, TC={tc_val}: {e}")
                    # Add a placeholder result to keep the loop going
                    results.append({
                        "Window": window,
                        "TC": tc_val,
                        "performance": 1.0,  # Neutral performance (no gain/loss)
                        "annualized_return": 0,
                        "annualized_risk": 0,
                        "sharpe": 0,
                        "total_trades": 0
                    })
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Display top performers
            st.subheader("Top 10 Parameter Combinations")
            top_results = results_df.nlargest(10, "performance").reset_index(drop=True)
            top_results["performance"] = top_results["performance"] - 1  # Convert to percentage
            top_results["TC"] = top_results["TC"] * 100  # Convert to percentage for display
            st.dataframe(top_results.style.format({
                "performance": "{:.2%}",
                "annualized_return": "{:.2%}",
                "annualized_risk": "{:.2%}",
                "sharpe": "{:.2f}",
                "TC": "{:.2f}%"
            }))
            
            # Allow download of results
            st.markdown(get_table_download_link(results_df, f"{selected_pair}_contrarian_optimization_results.csv"), unsafe_allow_html=True)
            
            # Create heatmap for visualization using Plotly
            st.subheader("Performance Heatmap")
            
            pivot_data = results_df.pivot_table(
                index="Window", 
                columns="TC", 
                values="performance",
                aggfunc='first'
            )
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot_data.values,
                x=[f"{tc*100:.2f}%" for tc in pivot_data.columns],
                y=pivot_data.index,
                colorscale='Viridis',
                colorbar=dict(title='Performance')
            ))
            
            fig.update_layout(
                title=f"Contrarian Parameter Optimization Heatmap for {selected_pair}",
                xaxis_title="Transaction Costs (%)",
                yaxis_title="Window",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display trades heatmap
            st.subheader("Trading Activity Heatmap")
            
            pivot_trades = results_df.pivot_table(
                index="Window", 
                columns="TC", 
                values="total_trades",
                aggfunc='first'
            )
            
            fig_trades = go.Figure(data=go.Heatmap(
                z=pivot_trades.values,
                x=[f"{tc*100:.2f}%" for tc in pivot_trades.columns],
                y=pivot_trades.index,
                colorscale='Blues',
                colorbar=dict(title='Number of Trades')
            ))
            
            fig_trades.update_layout(
                title="Trading Activity by Parameter Combination",
                xaxis_title="Transaction Costs (%)",
                yaxis_title="Window",
                height=500
            )
            
            st.plotly_chart(fig_trades, use_container_width=True)
            
            # Option to use best parameters
            best_params = results_df.loc[results_df["performance"].idxmax()]
            
            st.subheader("Best Parameters")
            col1, col2, col3 = st.columns(3)
            col1.metric("Window", int(best_params["Window"]))
            col2.metric("Transaction Costs", f"{best_params['TC']*100:.2f}%")
            col3.metric("Performance", f"{best_params['performance'] - 1:.2%}")
            
            st.info(f"💡 Tip: You can use these optimal parameters (Window: {int(best_params['Window'])}, TC: {best_params['TC']*100:.2f}%) in the Contrarian Strategy section.")
    
    elif strategy_type == "ML Parameter Optimization":
        st.header("ML Parameter Optimization")
        
        # Parameter ranges
        st.subheader("Set Parameter Ranges")
        col1, col2 = st.columns(2)
        with col1:
            lags_min = st.number_input("Min Lags", min_value=1, max_value=10, value=1, step=1)
            lags_max = st.number_input("Max Lags", min_value=2, max_value=20, value=10, step=1)
            lags_step = st.number_input("Lags Step", min_value=1, max_value=5, value=1, step=1)
        
        with col2:
            train_ratio = st.slider("Training Set Ratio", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
            tc_values = st.slider("Transaction Costs Range (%)", 
                                min_value=0.0, 
                                max_value=1.0, 
                                value=(0.1, 0.5), 
                                step=0.1)
            tc_min, tc_max = tc_values
            tc_step = st.number_input("TC Step (%)", min_value=0.05, max_value=0.2, value=0.1, step=0.05)
            
        # Generate parameter ranges
        lags_range = range(lags_min, lags_max + 1, lags_step)
        tc_range = np.arange(tc_min/100, tc_max/100 + tc_step/100, tc_step/100)  # Convert to decimal
        
        total_combinations = len(list(lags_range)) * len(tc_range)
        
        st.write(f"Total combinations to test: {total_combinations}")
        
        if st.button("Run Optimization"):
            if total_combinations > 50:
                st.warning(f"Testing {total_combinations} combinations with ML may take a while. Consider reducing the parameter ranges.")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Generate combinations
            combinations = list(product(lags_range, tc_range))
            
            # Initialize results list
            results = []
            
            # Run tests for each combination
            for i, (lags, tc_val) in enumerate(combinations):
                # Update progress
                progress = (i + 1) / total_combinations
                progress_bar.progress(progress)
                status_text.text(f"Testing combination {i+1}/{total_combinations}: Lags={lags}, TC={tc_val:.4f}")
                
                try:
                    # Test strategy
                    df_test, conf_matrix, _, _, _ = test_ml_strategy(df, selected_pair, lags, train_ratio, tc_val)
                    performance = np.exp(df_test["strategy"].sum())
                    
                    # Calculate metrics with safety checks
                    ann_return = df_test["strategy"].mean() * 252
                    ann_risk = df_test["strategy"].std() * np.sqrt(252)
                    sharpe = (ann_return / ann_risk) if ann_risk > 0 else 0
                    
                    # Calculate number of trades and prediction accuracy
                    total_trades = df_test["trades"].sum()
                    accuracy = (np.sign(df_test["returns"]) == df_test["position"].shift(1)).astype(int).mean()
                    
                    # Store results
                    results.append({
                        "Lags": lags,
                        "TC": tc_val,
                        "performance": performance,
                        "annualized_return": ann_return,
                        "annualized_risk": ann_risk,
                        "sharpe": sharpe,
                        "total_trades": total_trades,
                        "accuracy": accuracy
                    })
                except Exception as e:
                    st.error(f"Error with Lags={lags}, TC={tc_val}: {e}")
                    # Add a placeholder result to keep the loop going
                    results.append({
                        "Lags": lags,
                        "TC": tc_val,
                        "performance": 1.0,  # Neutral performance (no gain/loss)
                        "annualized_return": 0,
                        "annualized_risk": 0,
                        "sharpe": 0,
                        "total_trades": 0,
                        "accuracy": 0
                    })
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Display top performers
            st.subheader("Top 10 Parameter Combinations")
            top_results = results_df.nlargest(10, "performance").reset_index(drop=True)
            top_results["performance"] = top_results["performance"] - 1  # Convert to percentage
            top_results["TC"] = top_results["TC"] * 100  # Convert to percentage for display
            top_results["accuracy"] = top_results["accuracy"] * 100  # Convert to percentage
            st.dataframe(top_results.style.format({
                "performance": "{:.2%}",
                "annualized_return": "{:.2%}",
                "annualized_risk": "{:.2%}",
                "sharpe": "{:.2f}",
                "TC": "{:.2f}%",
                "accuracy": "{:.2f}%"
            }))
            
            # Allow download of results
            st.markdown(get_table_download_link(results_df, f"{selected_pair}_ml_optimization_results.csv"), unsafe_allow_html=True)
            
            # Create heatmap for visualization using Plotly
            st.subheader("Performance Heatmap")
            
            pivot_data = results_df.pivot_table(
                index="Lags", 
                columns="TC", 
                values="performance",
                aggfunc='first'
            )
            
            fig = go.