import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
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
        self.model = OneVsRestClassifier(LogisticRegression(C = 1e6, max_iter = 100000))
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
            title = "Logistic Regression: {} | TC = {}".format(self.symbol, self.tc)
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
        
        return metrics

# Streamlit app
def main():
    st.set_page_config(page_title="ML Trading Strategy Backtester", layout="wide")
    
    st.title("Machine Learning Trading Strategy Backtester")
    st.markdown("""
    This application allows you to backtest a machine learning-based trading strategy using logistic regression.
    Upload your time series data and configure the backtest parameters to evaluate the performance.
    """)
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload CSV file with time series data", type=["csv"])
    
    if uploaded_file is not None:
        # Load the data
        try:
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
            
            run_button = st.sidebar.button("Run Backtest")
            
            # Create tabs for different sections
            tab1, tab2, tab3 = st.tabs(["Performance Charts", "Performance Metrics", "Data Preview"])
            
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
