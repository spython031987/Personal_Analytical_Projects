import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import io
import base64
from datetime import datetime, timedelta
import plotly.graph_objects as go
import sys
import os

# Import the MLBacktester class
class MLBacktester():
    ''' Class for the vectorized backtesting of Machine Learning-based trading strategies (Classification).
    '''

    def __init__(self, symbol, start, end, tc):
        '''
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        start: str
            start date for data import
        end: str
            end date for data import
        tc: float
            proportional transaction/trading costs per trade
        '''
        self.symbol = symbol
        self.start = start
        self.end = end
        self.tc = tc
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.results = None
        self.data = None  # Will be set by set_data method
    
    def __repr__(self):
        rep = "MLBacktester(symbol = {}, start = {}, end = {}, tc = {})"
        return rep.format(self.symbol, self.start, self.end, self.tc)
    
    def set_data(self, dataframe):
        '''Sets the data from a provided DataFrame instead of loading from CSV.'''
        raw = dataframe[self.symbol].to_frame().dropna()
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol: "price"}, inplace=True)
        raw["returns"] = np.log(raw / raw.shift(1))
        self.data = raw
                             
    def split_data(self, start, end):
        ''' Splits the data into training set & test set.
        '''
        data = self.data.loc[start:end].copy()
        return data
    
    def prepare_features(self, start, end):
        ''' Prepares the feature columns for training set and test set.
        '''
        self.data_subset = self.split_data(start, end)
        self.feature_columns = []
        for lag in range(1, self.lags + 1):
            col = "lag{}".format(lag)
            self.data_subset[col] = self.data_subset["returns"].shift(lag)
            self.feature_columns.append(col)
        self.data_subset.dropna(inplace=True)

    def scale_features(self, recalc = True):
        ''' Scales/Standardizes Features
        '''
        if recalc:
            self.means = self.data_subset[self.feature_columns].mean()
            self.stand_devs = self.data_subset[self.feature_columns].std()
        
        self.data_subset[self.feature_columns] = (self.data_subset[self.feature_columns] - self.means) / self.stand_devs
        
    def fit_model(self, start, end):
        ''' Fitting the ML Model.
        '''
        self.prepare_features(start, end)
        self.scale_features(recalc = True) # calculate mean & std of train set and scale train set
        self.model.fit(self.data_subset[self.feature_columns], np.sign(self.data_subset["returns"]))
        
    def test_strategy(self, train_ratio = 0.7, lags = 5):
        ''' 
        Backtests the ML-based strategy.
        
        Parameters
        ----------
        train_ratio: float (between 0 and 1.0 excl.)
            Splitting the dataset into training set (train_ratio) and test set (1 - train_ratio).
        lags: int
            number of lags serving as model features.
        '''
        self.lags = lags
                  
        # determining datetime for start, end and split (for training an testing period)
        full_data = self.data.copy()
        split_index = int(len(full_data) * train_ratio)
        split_date = full_data.index[split_index-1]
        train_start = full_data.index[0]
        test_end = full_data.index[-1]
        
        # fit the model on the training set
        self.fit_model(train_start, split_date)
        
        # prepare the test set
        self.prepare_features(split_date, test_end)
        self.scale_features(recalc = False) # scale test set features with train set mean & std
                  
        # make predictions on the test set
        predict = self.model.predict(self.data_subset[self.feature_columns])
        self.data_subset["pred"] = predict
        
        # calculate Strategy Returns
        self.data_subset["strategy"] = self.data_subset["pred"] * self.data_subset["returns"]
        
        # determine the number of trades in each bar
        self.data_subset["trades"] = self.data_subset["pred"].diff().fillna(0).abs()
        
        # subtract transaction/trading costs from pre-cost return
        self.data_subset.strategy = self.data_subset.strategy - self.data_subset.trades * self.tc
        
        # calculate cumulative returns for strategy & buy and hold
        self.data_subset["creturns"] = self.data_subset["returns"].cumsum().apply(np.exp)
        self.data_subset["cstrategy"] = self.data_subset['strategy'].cumsum().apply(np.exp)
        self.results = self.data_subset
        
        perf = self.results["cstrategy"].iloc[-1] # absolute performance of the strategy
        outperf = perf - self.results["creturns"].iloc[-1] # out-/underperformance of strategy
        
        return round(perf, 6), round(outperf, 6)
        
    def plot_results(self):
        ''' Plots the performance of the trading strategy and compares to "buy and hold".
        '''
        if self.results is None:
            print("Run test_strategy() first.")
            return None
        else:
            title = "XGBoost Strategy: {} | TC = {}".format(self.symbol, self.tc)
            fig, ax = plt.subplots(figsize=(12, 8))
            self.results[["creturns", "cstrategy"]].plot(title=title, ax=ax)
            plt.tight_layout()
            return fig

    def get_performance_metrics(self):
        """Calculate and return additional performance metrics"""
        if self.results is None:
            return None
        
        metrics = {}
        
        # Total Return
        metrics['strategy_return'] = round((self.results["cstrategy"].iloc[-1] - 1) * 100, 2)
        metrics['buy_hold_return'] = round((self.results["creturns"].iloc[-1] - 1) * 100, 2)
        
        # Annualized Return (assuming 252 trading days per year)
        days = (self.results.index[-1] - self.results.index[0]).days
        years = days / 365
        metrics['ann_strategy_return'] = round(((metrics['strategy_return']/100 + 1) ** (1/years) - 1) * 100, 2)
        metrics['ann_buy_hold_return'] = round(((metrics['buy_hold_return']/100 + 1) ** (1/years) - 1) * 100, 2)
        
        # Maximum Drawdown
        strategy_cummax = self.results["cstrategy"].cummax()
        buy_hold_cummax = self.results["creturns"].cummax()
        strategy_drawdown = (self.results["cstrategy"] / strategy_cummax - 1) * 100
        buy_hold_drawdown = (self.results["creturns"] / buy_hold_cummax - 1) * 100
        metrics['max_strategy_drawdown'] = round(strategy_drawdown.min(), 2)
        metrics['max_buy_hold_drawdown'] = round(buy_hold_drawdown.min(), 2)
        
        # Sharpe Ratio (assuming 252 trading days per year and risk-free rate of 0)
        strategy_returns = self.results["strategy"]
        buy_hold_returns = self.results["returns"]
        strategy_sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
        buy_hold_sharpe = np.sqrt(252) * buy_hold_returns.mean() / buy_hold_returns.std()
        metrics['strategy_sharpe'] = round(strategy_sharpe, 2)
        metrics['buy_hold_sharpe'] = round(buy_hold_sharpe, 2)
        
        # Win Rate
        trades = self.results["trades"] > 0
        winning_trades = (self.results["strategy"] > 0) & trades
        metrics['win_rate'] = round(winning_trades.sum() / trades.sum() * 100, 2) if trades.sum() > 0 else 0
        
        # Total Number of Trades
        metrics['total_trades'] = int(self.results["trades"].sum())
        
        # Calculate profit factor (gross profits / gross losses)
        profits = strategy_returns[strategy_returns > 0].sum()
        losses = abs(strategy_returns[strategy_returns < 0].sum())
        metrics['profit_factor'] = round(profits / losses, 2) if losses != 0 else float('inf')
        
        # Average profit per trade
        metrics['avg_profit_per_trade'] = round(strategy_returns.sum() / metrics['total_trades'], 5) if metrics['total_trades'] > 0 else 0
        
        return metrics
        
    def plot_feature_importance(self):
        """Plot feature importance from the trained XGBoost model"""
        if hasattr(self, 'model') and hasattr(self.model, 'feature_importances_'):
            # Get feature importances
            importances = self.model.feature_importances_
            
            # Create a DataFrame for easier plotting
            importance_df = pd.DataFrame({
                'Feature': self.feature_columns,
                'Importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 6))
            importance_df.plot(kind='bar', x='Feature', y='Importance', ax=ax)
            plt.title(f'Feature Importance for {self.symbol}')
            plt.tight_layout()
            
            return fig, importance_df
        else:
            return None, None

# Streamlit app
def main():
    st.set_page_config(page_title="ML Trading Strategy Backtester", layout="wide")
    
    st.title("Machine Learning Trading Strategy Backtester")
    st.markdown("""
    This application allows you to backtest a machine learning-based trading strategy using XGBoost.
    Upload your time series data and configure the backtest parameters to evaluate the performance.
    """)
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload CSV file with time series data", type=["csv"])
    
    # Add option to generate sample data
    st.sidebar.markdown("---")
    st.sidebar.subheader("No data? Generate a sample")
    
    if st.sidebar.button("Generate Sample Data"):
        with st.spinner("Generating sample data..."):
            # Generate sample data
            start_date = datetime.now() - timedelta(days=365)
            start_date_str = start_date.strftime('%Y-%m-%d')
            
            # Generate timestamps (excluding weekends)
            timestamps = []
            current_dt = start_date
            end_dt = datetime.now()
            
            while current_dt < end_dt:
                # Skip weekends (5 = Saturday, 6 = Sunday)
                if current_dt.weekday() < 5:
                    # Generate timestamps for trading hours (9:30 AM to 4:00 PM)
                    trading_start = datetime.combine(current_dt.date(), datetime.strptime("09:30", "%H:%M").time())
                    trading_end = datetime.combine(current_dt.date(), datetime.strptime("16:00", "%H:%M").time())
                    
                    current_time = trading_start
                    while current_time <= trading_end:
                        timestamps.append(current_time)
                        current_time += timedelta(minutes=5)
                
                current_dt += timedelta(days=1)
            
            # Initialize DataFrame with timestamps
            df = pd.DataFrame(index=timestamps)
            df.index.name = 'time'
            
            # Generate price data for sample symbols
            sample_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'FB']
            volatility = 0.001
            trend = 0.0002
            
            for symbol in sample_symbols:
                # Random starting price between 50 and 500
                price = np.random.uniform(50, 500)
                
                # Random trend and volatility modifiers for each symbol
                symbol_trend = trend * np.random.uniform(0.5, 2.0)
                symbol_volatility = volatility * np.random.uniform(0.8, 1.5)
                
                prices = []
                for i in range(len(timestamps)):
                    # Add trend
                    price *= (1 + symbol_trend)
                    
                    # Add random movement (volatility)
                    price *= (1 + np.random.normal(0, symbol_volatility))
                    
                    # Add to list
                    prices.append(price)
                
                df[symbol] = prices
            
            st.sidebar.success("Sample data generated successfully!")
            
            # Use the generated data for the rest of the app
            uploaded_file = "sample_data"  # Flag to indicate we're using sample data
        
    # Data processing
    if uploaded_file is not None:
        # Load the data
        try:
            if uploaded_file == "sample_data":
                # Use the sample data generated above
                pass  # df is already created
            else:
                # Load from uploaded file
                df = pd.read_csv(uploaded_file, parse_dates=["time"], index_col="time")
            st.sidebar.success("Data loaded successfully!")
            
            # Show available symbols
            symbols = df.columns.tolist()
            
            # Parameters
            st.sidebar.header("Backtest Parameters")
            
            symbol = st.sidebar.selectbox("Select symbol", options=symbols)
            
            min_date = df.index.min().date()
            max_date = df.index.max().date()
            
            start_date = st.sidebar.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
            end_date = st.sidebar.date_input("End date", max_date, min_value=min_date, max_value=max_date)
            
            # Convert dates to string format for the MLBacktester
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            tc = st.sidebar.number_input("Transaction costs (e.g., 0.001 for 0.1%)", min_value=0.0, max_value=0.1, value=0.001, step=0.0001, format="%.4f")
            train_ratio = st.sidebar.slider("Training set ratio", min_value=0.1, max_value=0.9, value=0.7, step=0.05)
            lags = st.sidebar.slider("Number of lags (features)", min_value=1, max_value=20, value=5, step=1)
            
            # XGBoost parameters
            st.sidebar.header("XGBoost Parameters")
            with st.sidebar.expander("XGBoost Parameters", expanded=False):
                n_estimators = st.number_input("Number of estimators", min_value=10, max_value=1000, value=100, step=10)
                max_depth = st.number_input("Max depth", min_value=1, max_value=15, value=3, step=1)
                learning_rate = st.number_input("Learning rate", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
                subsample = st.number_input("Subsample", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
                colsample_bytree = st.number_input("Column sample by tree", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
            
            run_button = st.sidebar.button("Run Backtest")
            
            # Create tabs for different sections
            tab1, tab2, tab3, tab4 = st.tabs(["Performance Charts", "Performance Metrics", "Model Analysis", "Data Preview"])
            
            with tab3:
                st.subheader(f"Data Preview for {symbol}")
                st.dataframe(df[symbol].head(10))
                
                # Get data statistics
                st.subheader("Data Statistics")
                st.dataframe(df[symbol].describe())
                
                # Show time range
                st.info(f"Data ranges from {min_date} to {max_date} ({(max_date - min_date).days} days)")
            
            if run_button:
                with st.spinner("Running backtest..."):
                    try:
                        # Create an instance of MLBacktester
                        ml_backtester = MLBacktester(symbol=symbol, start=start_date_str, end=end_date_str, tc=tc)
                        
                        # Update XGBoost parameters based on user input
                        ml_backtester.model = xgb.XGBClassifier(
                            objective='binary:logistic',
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            use_label_encoder=False,
                            eval_metric='logloss'
                        )
                        
                        ml_backtester.set_data(df)
                        
                        # Run the backtest
                        perf, outperf = ml_backtester.test_strategy(train_ratio=train_ratio, lags=lags)
                        
                        # Get performance metrics
                        metrics = ml_backtester.get_performance_metrics()
                        
                        with tab1:
                            st.subheader("Backtest Results")
                            st.markdown(f"""
                            **Strategy Performance:** {perf:.6f} ({metrics['strategy_return']}%)  
                            **Out/Underperformance:** {outperf:.6f} ({metrics['strategy_return'] - metrics['buy_hold_return']}%)
                            """)
                            
                            # Plot using matplotlib
                            fig = ml_backtester.plot_results()
                            st.pyplot(fig)
                            
                            # Create interactive plot with Plotly
                            st.subheader("Interactive Performance Chart")
                            results = ml_backtester.results
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=results.index, 
                                y=results["creturns"],
                                mode='lines',
                                name='Buy and Hold'
                            ))
                            fig.add_trace(go.Scatter(
                                x=results.index, 
                                y=results["cstrategy"],
                                mode='lines',
                                name='ML Strategy'
                            ))
                            
                            fig.update_layout(
                                title=f"ML Strategy vs Buy and Hold: {symbol}",
                                xaxis_title="Date",
                                yaxis_title="Cumulative Return",
                                height=600
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add drawdown chart
                            st.subheader("Drawdown Analysis")
                            strategy_cummax = results["cstrategy"].cummax()
                            buy_hold_cummax = results["creturns"].cummax()
                            strategy_drawdown = (results["cstrategy"] / strategy_cummax - 1) * 100
                            buy_hold_drawdown = (results["creturns"] / buy_hold_cummax - 1) * 100
                            
                            fig_dd = go.Figure()
                            fig_dd.add_trace(go.Scatter(
                                x=results.index,
                                y=buy_hold_drawdown,
                                mode='lines',
                                name='Buy and Hold Drawdown',
                                line=dict(color='red')
                            ))
                            fig_dd.add_trace(go.Scatter(
                                x=results.index,
                                y=strategy_drawdown,
                                mode='lines',
                                name='Strategy Drawdown',
                                line=dict(color='blue')
                            ))
                            
                            fig_dd.update_layout(
                                title=f"Drawdown Analysis: {symbol}",
                                xaxis_title="Date",
                                yaxis_title="Drawdown (%)",
                                height=400
                            )
                            
                            st.plotly_chart(fig_dd, use_container_width=True)
                        
                        with tab2:
                            st.subheader("Performance Metrics")
                            
                            # Create two columns for metrics display
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("### Strategy Metrics")
                                st.markdown(f"""
                                - **Total Return:** {metrics['strategy_return']}%
                                - **Annualized Return:** {metrics['ann_strategy_return']}%
                                - **Maximum Drawdown:** {metrics['max_strategy_drawdown']}%
                                - **Sharpe Ratio:** {metrics['strategy_sharpe']}
                                - **Win Rate:** {metrics['win_rate']}%
                                - **Total Trades:** {metrics['total_trades']}
                                """)
                            
                            with col2:
                                st.markdown("### Buy & Hold Metrics")
                                st.markdown(f"""
                                - **Total Return:** {metrics['buy_hold_return']}%
                                - **Annualized Return:** {metrics['ann_buy_hold_return']}%
                                - **Maximum Drawdown:** {metrics['max_buy_hold_drawdown']}%
                                - **Sharpe Ratio:** {metrics['buy_hold_sharpe']}
                                """)
                            
                            # Create a comparison bar chart
                            st.subheader("Strategy vs Buy & Hold Comparison")
                            
                            comparison_data = {
                                'Metric': ['Total Return (%)', 'Annualized Return (%)', 'Max Drawdown (%)', 'Sharpe Ratio'],
                                'Strategy': [metrics['strategy_return'], metrics['ann_strategy_return'], 
                                            metrics['max_strategy_drawdown'], metrics['strategy_sharpe']],
                                'Buy & Hold': [metrics['buy_hold_return'], metrics['ann_buy_hold_return'], 
                                              metrics['max_buy_hold_drawdown'], metrics['buy_hold_sharpe']]
                            }
                            
                            fig_comp = go.Figure()
                            
                            # Add traces
                            fig_comp.add_trace(go.Bar(
                                x=comparison_data['Metric'],
                                y=comparison_data['Strategy'],
                                name='Strategy',
                                marker_color='blue'
                            ))
                            fig_comp.add_trace(go.Bar(
                                x=comparison_data['Metric'],
                                y=comparison_data['Buy & Hold'],
                                name='Buy & Hold',
                                marker_color='red'
                            ))
                            
                            # Update layout
                            fig_comp.update_layout(
                                barmode='group',
                                title='Strategy vs Buy & Hold Performance Metrics',
                                xaxis_tickangle=-45,
                                height=500
                            )
                            
                            st.plotly_chart(fig_comp, use_container_width=True)
                            
                            # Display monthly returns
                            st.subheader("Monthly Returns Analysis")
                            
                            # Calculate monthly returns
                            monthly_strategy_returns = results['strategy'].resample('M').sum() * 100
                            monthly_bh_returns = results['returns'].resample('M').sum() * 100
                            
                            # Create a DataFrame for display
                            monthly_df = pd.DataFrame({
                                'Strategy (%)': monthly_strategy_returns,
                                'Buy & Hold (%)': monthly_bh_returns
                            })
                            
                            # Display as a table
                            st.dataframe(monthly_df.style.format("{:.2f}"))
                            
                        with tab3:
                            st.subheader("XGBoost Model Analysis")
                            
                            # Plot feature importance
                            feature_imp_fig, importance_df = ml_backtester.plot_feature_importance()
                            if feature_imp_fig:
                                st.pyplot(feature_imp_fig)
                                
                                # Display feature importance as a table
                                st.subheader("Feature Importance Values")
                                st.dataframe(importance_df.style.format({"Importance": "{:.4f}"}))
                                
                                # Create plotly bar chart for feature importance
                                fig_imp = go.Figure()
                                fig_imp.add_trace(go.Bar(
                                    x=importance_df['Feature'],
                                    y=importance_df['Importance'],
                                    marker_color='teal'
                                ))
                                fig_imp.update_layout(
                                    title='Feature Importance (Interactive)',
                                    xaxis_title='Feature',
                                    yaxis_title='Importance Score',
                                    height=400
                                )
                                st.plotly_chart(fig_imp, use_container_width=True)
                                
                                # Information about model interpretation
                                st.info("""
                                **Interpreting Feature Importance:**
                                - Higher importance means the feature has more influence on the model's predictions
                                - Features typically represent lag periods of returns (e.g., lag1 = previous return)
                                - The model relies more heavily on features with higher importance scores
                                """)
                                
                                # Add prediction distribution analysis
                                st.subheader("Prediction Distribution")
                                pred_counts = ml_backtester.results['pred'].value_counts()
                                
                                # Create a pie chart for prediction distribution
                                labels = {1: 'Buy/Long', -1: 'Sell/Short', 0: 'Hold'}
                                values = pred_counts.values
                                
                                fig_pie = go.Figure(data=[go.Pie(
                                    labels=[labels.get(i, i) for i in pred_counts.index],
                                    values=values
                                )])
                                fig_pie.update_layout(title='Distribution of Trading Signals')
                                st.plotly_chart(fig_pie, use_container_width=True)
                                
                                # Display confusion matrix
                                st.subheader("Model Performance")
                                st.markdown("""
                                This section shows how well the model performed in terms of direction prediction:
                                - **Correct direction:** Times when the model correctly predicted the price movement direction
                                - **Incorrect direction:** Times when the model got the direction wrong
                                - **Total predictions:** Total number of trading signals generated
                                """)
                                
                                # Calculate direction accuracy
                                correct_direction = ((ml_backtester.results['pred'] > 0) & (ml_backtester.results['returns'] > 0) | 
                                                   (ml_backtester.results['pred'] < 0) & (ml_backtester.results['returns'] < 0)).sum()
                                incorrect_direction = ((ml_backtester.results['pred'] > 0) & (ml_backtester.results['returns'] < 0) | 
                                                     (ml_backtester.results['pred'] < 0) & (ml_backtester.results['returns'] > 0)).sum()
                                total_signals = (ml_backtester.results['pred'] != 0).sum()
                                
                                direction_accuracy = round(correct_direction / total_signals * 100, 2) if total_signals > 0 else 0
                                
                                # Display metrics
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Correct Direction", f"{correct_direction} ({direction_accuracy}%)")
                                col2.metric("Incorrect Direction", f"{incorrect_direction} ({100-direction_accuracy}%)")
                                col3.metric("Total Predictions", total_signals)
                            
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        raise e
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    else:
        # Instructions when no file is uploaded
        st.info("Please upload a CSV file with time series data. The file should have a 'time' column as the index and at least one symbol column.")
        st.markdown("""
        ### Sample Data Format:
        
        ```
        time,AAPL,MSFT,GOOGL
        2020-01-01 09:30:00,100.1,200.2,1500.3
        2020-01-01 09:35:00,100.2,200.3,1500.4
        ...
        ```
        
        The 'time' column will be used as the index and should be in a format that pandas can parse as a datetime.
        """)
        
        # About section
        st.markdown("---")
        st.subheader("About ML Trading Strategy Backtester")
        
        tabs = st.tabs(["ML Backtesting", "XGBoost Algorithm", "Risk Management"])
        
        with tabs[0]:
            st.markdown("""
            ### Machine Learning in Algorithmic Trading
            
            Machine learning approaches to algorithmic trading involve using historical price data to train models that can predict future price movements. These models learn patterns from past data and make predictions that inform trading decisions.
            
            #### Key Advantages:
            - **Pattern Recognition**: ML models can identify complex patterns in market data
            - **Adaptability**: With retraining, models can adapt to changing market conditions
            - **Objectivity**: Removes emotional bias from trading decisions
            - **Scalability**: Can analyze multiple markets and timeframes simultaneously
            
            #### Key Challenges:
            - **Overfitting**: Models may learn patterns specific to historical data that don't generalize
            - **Market Regimes**: Markets change behavior over time, requiring model updates
            - **Feature Selection**: Choosing the right inputs is critical for model performance
            - **Transaction Costs**: High-frequency strategies must account for trading costs
            """)
            
        with tabs[1]:
            st.markdown("""
            ### XGBoost Algorithm
            
            XGBoost (eXtreme Gradient Boosting) is an optimized gradient boosting framework particularly well-suited for trading applications:
            
            #### How XGBoost Works:
            1. **Ensemble Method**: Builds multiple decision trees sequentially
            2. **Gradient Boosting**: Each new tree focuses on correcting errors made by previous trees
            3. **Regularization**: Built-in regularization helps prevent overfitting
            4. **Feature Importance**: Provides insights into which features drive predictions
            
            #### Advantages for Trading:
            - **Handles Non-Linear Patterns**: Markets often exhibit complex, non-linear behavior
            - **Robust to Noise**: Performs well with noisy financial data
            - **Feature Importance**: Helps identify which factors drive market movements
            - **Efficiency**: Fast training and prediction times compared to many other algorithms
            """)
            
        with tabs[2]:
            st.markdown("""
            ### Risk Management
            
            Successful trading strategies require proper risk management:
            
            #### Key Risk Metrics:
            - **Maximum Drawdown**: Largest peak-to-trough decline in portfolio value
            - **Sharpe Ratio**: Measures risk-adjusted returns (return divided by volatility)
            - **Win Rate**: Percentage of profitable trades
            - **Profit Factor**: Gross profits divided by gross losses
            
            #### Best Practices:
            - **Position Sizing**: Limit exposure to any single trade
            - **Stop Losses**: Implement mechanisms to limit losses on individual trades
            - **Diversification**: Trade multiple uncorrelated instruments
            - **Stress Testing**: Test strategy performance in extreme market conditions
            - **Walkforward Testing**: Test on multiple out-of-sample periods
            
            Remember that past performance does not guarantee future results. Always employ proper risk management when deploying any algorithmic trading strategy.
            """)
        
        st.markdown("---")
        st.markdown("""
        ### About the Backtester
        
        This application uses a machine learning approach to backtest trading strategies:
        
        1. **Training Phase**: The model learns patterns from historical price data using lagged returns as features
        2. **Testing Phase**: The trained model makes predictions on unseen data
        3. **Performance Evaluation**: The strategy is compared to a buy-and-hold approach
        
        #### Key Parameters:
        
        - **Transaction costs**: The cost per trade (e.g., 0.001 for 0.1%)
        - **Training set ratio**: Proportion of data used for training
        - **Number of lags**: The number of previous returns used as features
        """)


if __name__ == "__main__":
    main()
