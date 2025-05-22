import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Trading Strategy Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .positive {
        color: #00cc00;
    }
    .negative {
        color: #ff0000;
    }
</style>
""", unsafe_allow_html=True)

class MLTradingStrategy:
    """
    A machine learning-based trading strategy using Random Forest
    with technical indicators and risk management.
    """
    
    def __init__(self, symbol='SPY', lookback_days=365*2, 
                 initial_capital=100000, position_size=0.95,
                 stop_loss=0.02, take_profit=0.03):
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.model = None
        self.scaler = StandardScaler()
        
    def fetch_data(self):
        """Fetch historical price data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        # Download data with auto_adjust=True to avoid the warning
        data = yf.download(self.symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)
        
        # Handle MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        self.data = data
        return len(self.data)
        
    def calculate_features(self):
        """Calculate technical indicators as features"""
        # Create a clean copy with just OHLCV data
        df = pd.DataFrame()
        df['Open'] = self.data['Open'].copy()
        df['High'] = self.data['High'].copy()
        df['Low'] = self.data['Low'].copy()
        df['Close'] = self.data['Close'].copy()
        df['Volume'] = self.data['Volume'].copy()
        
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close']/df['Close'].shift(1))
        
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            df[f'SMA_{period}_ratio'] = df['Close'] / df[f'SMA_{period}']
        
        # Exponential moving averages
        for period in [12, 26]:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_diff'] = df['MACD'] - df['MACD_signal']
        
        # RSI
        def calculate_rsi(data, period=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['RSI'] = calculate_rsi(df['Close'])
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * bb_std
        df['BB_lower'] = df['BB_middle'] - 2 * bb_std
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Volume features
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        df['volume_change'] = df['Volume'].pct_change()
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Price position features
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        
        # Trend features
        df['trend_strength'] = (df['SMA_10'] - df['SMA_50']) / df['SMA_50']
        
        # Create target variable (1 if price goes up, 0 if down)
        df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        self.processed_data = df
        
    def prepare_features(self):
        """Prepare features for machine learning"""
        # Select features
        feature_columns = [col for col in self.processed_data.columns 
                         if col not in ['Open', 'High', 'Low', 'Close', 
                                       'Adj Close', 'Volume', 'target']]
        
        X = self.processed_data[feature_columns]
        y = self.processed_data['target']
        
        return X, y
    
    def train_model(self, test_size=0.2):
        """Train the Random Forest model"""
        X, y = self.prepare_features()
        
        # Split data
        split_index = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return train_score, test_score, feature_importance
    
    def backtest(self):
        """Backtest the trading strategy"""
        X, y = self.prepare_features()
        
        # Get predictions for the entire dataset
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Create trading signals dataframe
        signals = pd.DataFrame(index=X.index)
        signals['price'] = self.processed_data.loc[X.index, 'Close']
        signals['prediction'] = predictions
        signals['probability'] = probabilities
        signals['actual_return'] = self.processed_data.loc[X.index, 'returns']
        
        # Generate trading signals based on probability threshold
        threshold = 0.55
        signals['position'] = 0
        signals.loc[signals['probability'] > threshold, 'position'] = 1
        signals.loc[signals['probability'] < (1 - threshold), 'position'] = -1
        
        # Calculate strategy returns
        signals['strategy_returns'] = signals['position'].shift(1) * signals['actual_return']
        
        # Apply transaction costs
        transaction_cost = 0.001
        signals['trades'] = signals['position'].diff().abs()
        signals['costs'] = signals['trades'] * transaction_cost
        signals['strategy_returns_net'] = signals['strategy_returns'] - signals['costs']
        
        # Calculate cumulative returns
        signals['cumulative_returns'] = (1 + signals['actual_return']).cumprod()
        signals['cumulative_strategy_returns'] = (1 + signals['strategy_returns_net']).cumprod()
        
        # Calculate performance metrics
        total_return = signals['cumulative_strategy_returns'].iloc[-1] - 1
        buy_hold_return = signals['cumulative_returns'].iloc[-1] - 1
        
        # Sharpe ratio
        sharpe_ratio = np.sqrt(252) * signals['strategy_returns_net'].mean() / signals['strategy_returns_net'].std()
        
        # Maximum drawdown
        cumulative = signals['cumulative_strategy_returns']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = signals[signals['strategy_returns_net'] > 0]['strategy_returns_net'].count()
        total_trades = signals[signals['trades'] == 1]['trades'].count()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        metrics = {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades
        }
        
        return signals, metrics
    
    def generate_current_signal(self):
        """Generate trading signal for the current market"""
        X, _ = self.prepare_features()
        latest_features = X.iloc[-1:].values
        latest_features_scaled = self.scaler.transform(latest_features)
        
        prediction = self.model.predict(latest_features_scaled)[0]
        probability = self.model.predict_proba(latest_features_scaled)[0, 1]
        
        if probability > 0.55:
            signal = "BUY"
        elif probability < 0.45:
            signal = "SELL"
        else:
            signal = "HOLD"
        
        return signal, probability

# Initialize session state
if 'strategy' not in st.session_state:
    st.session_state.strategy = None
if 'signals' not in st.session_state:
    st.session_state.signals = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

# Main app
st.title("🤖 ML Trading Strategy Dashboard")
st.markdown("### AI-Powered Trading Strategy with Technical Analysis")

# Sidebar
with st.sidebar:
    st.header("⚙️ Strategy Configuration")
    
    symbol = st.text_input("Stock Symbol", value="SPY")
    
    st.subheader("Model Parameters")
    lookback_days = st.slider("Lookback Period (days)", 180, 1080, 730)
    
    st.subheader("Risk Management")
    initial_capital = st.number_input("Initial Capital ($)", 
                                    value=100000, 
                                    min_value=1000, 
                                    step=1000)
    position_size = st.slider("Max Position Size", 0.1, 1.0, 0.95)
    stop_loss = st.slider("Stop Loss %", 0.01, 0.10, 0.02)
    take_profit = st.slider("Take Profit %", 0.01, 0.20, 0.03)
    
    if st.button("🚀 Run Strategy", type="primary"):
        with st.spinner("Training model and running backtest..."):
            # Initialize strategy
            strategy = MLTradingStrategy(
                symbol=symbol,
                lookback_days=lookback_days,
                initial_capital=initial_capital,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Fetch data
            data_points = strategy.fetch_data()
            
            # Calculate features
            strategy.calculate_features()
            
            # Train model
            train_score, test_score, feature_importance = strategy.train_model()
            
            # Run backtest
            signals, metrics = strategy.backtest()
            
            # Generate current signal
            current_signal, confidence = strategy.generate_current_signal()
            
            # Store in session state
            st.session_state.strategy = strategy
            st.session_state.signals = signals
            st.session_state.metrics = metrics
            st.session_state.train_score = train_score
            st.session_state.test_score = test_score
            st.session_state.feature_importance = feature_importance
            st.session_state.current_signal = current_signal
            st.session_state.confidence = confidence
            
            st.success("✅ Strategy executed successfully!")

# Main content
if st.session_state.strategy is not None:
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Performance", "🎯 Signals", "🧠 Model Insights", "⚡ Live Trading"])
    
    with tab1:
        # Current market status
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = st.session_state.strategy.processed_data['Close'].iloc[-1]
        signal = st.session_state.current_signal
        confidence = st.session_state.confidence
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            signal_color = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
            st.metric("Signal", f"{signal_color} {signal}")
        
        with col3:
            st.metric("Confidence", f"{confidence:.1%}")
        
        with col4:
            recommended_position = st.session_state.strategy.initial_capital * position_size * confidence
            st.metric("Position Size", f"${recommended_position:,.0f}")
        
        # Performance metrics
        st.subheader("📊 Strategy Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_return = st.session_state.metrics['total_return']
            color = "positive" if total_return > 0 else "negative"
            st.metric("Total Return", f"{total_return:.2%}", 
                     delta=f"vs B&H: {total_return - st.session_state.metrics['buy_hold_return']:.2%}")
        
        with col2:
            st.metric("Sharpe Ratio", f"{st.session_state.metrics['sharpe_ratio']:.2f}")
        
        with col3:
            st.metric("Max Drawdown", f"{st.session_state.metrics['max_drawdown']:.2%}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Win Rate", f"{st.session_state.metrics['win_rate']:.1%}")
        
        with col2:
            st.metric("Total Trades", f"{st.session_state.metrics['total_trades']:,}")
        
        with col3:
            st.metric("Model Accuracy", f"{st.session_state.test_score:.1%}")
    
    with tab2:
        st.subheader("📈 Performance Analysis")
        
        # Cumulative returns chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Cumulative Returns', 'Strategy vs Buy & Hold'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Main chart
        fig.add_trace(
            go.Scatter(x=st.session_state.signals.index, 
                      y=st.session_state.signals['cumulative_strategy_returns'],
                      name='Strategy', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=st.session_state.signals.index, 
                      y=st.session_state.signals['cumulative_returns'],
                      name='Buy & Hold', line=dict(color='gray', width=1, dash='dash')),
            row=1, col=1
        )
        
        # Relative performance
        relative_perf = st.session_state.signals['cumulative_strategy_returns'] / st.session_state.signals['cumulative_returns']
        fig.add_trace(
            go.Scatter(x=st.session_state.signals.index, 
                      y=relative_perf,
                      name='Relative Performance', line=dict(color='green', width=2)),
            row=2, col=1
        )
        
        fig.add_hline(y=1, line_dash="dash", line_color="gray", row=2, col=1)
        
        fig.update_layout(height=800, showlegend=True)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
        fig.update_yaxes(title_text="Relative Performance", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly returns heatmap
        st.subheader("📅 Monthly Returns Heatmap")
        
        monthly_returns = st.session_state.signals['strategy_returns_net'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_pivot = pd.DataFrame(monthly_returns)
        monthly_returns_pivot['Year'] = monthly_returns_pivot.index.year
        monthly_returns_pivot['Month'] = monthly_returns_pivot.index.month
        monthly_returns_pivot = monthly_returns_pivot.pivot(index='Year', columns='Month', values='strategy_returns_net')
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=monthly_returns_pivot.values * 100,
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            y=monthly_returns_pivot.index,
            colorscale='RdYlGn',
            zmid=0,
            text=np.round(monthly_returns_pivot.values * 100, 1),
            texttemplate='%{text}%',
            textfont={"size": 10}
        ))
        
        fig_heatmap.update_layout(
            title='Monthly Returns (%)',
            height=400
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab3:
        st.subheader("🎯 Trading Signals")
        
        # Price chart with signals
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Price & Trading Signals', 'RSI', 'MACD'),
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Price and signals
        fig.add_trace(
            go.Candlestick(
                x=st.session_state.strategy.processed_data.index,
                open=st.session_state.strategy.processed_data['Open'],
                high=st.session_state.strategy.processed_data['High'],
                low=st.session_state.strategy.processed_data['Low'],
                close=st.session_state.strategy.processed_data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Buy signals
        buy_signals = st.session_state.signals[st.session_state.signals['position'] == 1]
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['price'],
                mode='markers',
                marker=dict(symbol='triangle-up', size=10, color='green'),
                name='Buy Signal'
            ),
            row=1, col=1
        )
        
        # Sell signals
        sell_signals = st.session_state.signals[st.session_state.signals['position'] == -1]
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['price'],
                mode='markers',
                marker=dict(symbol='triangle-down', size=10, color='red'),
                name='Sell Signal'
            ),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=st.session_state.strategy.processed_data.index,
                y=st.session_state.strategy.processed_data['RSI'],
                name='RSI',
                line=dict(color='purple')
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(
                x=st.session_state.strategy.processed_data.index,
                y=st.session_state.strategy.processed_data['MACD'],
                name='MACD',
                line=dict(color='blue')
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=st.session_state.strategy.processed_data.index,
                y=st.session_state.strategy.processed_data['MACD_signal'],
                name='Signal',
                line=dict(color='red')
            ),
            row=3, col=1
        )
        
        fig.update_layout(height=900, showlegend=True)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent signals table
        st.subheader("📋 Recent Trading Signals")
        recent_signals = st.session_state.signals[st.session_state.signals['trades'] == 1].tail(10)
        recent_signals_display = recent_signals[['price', 'position', 'probability', 'strategy_returns_net']].copy()
        recent_signals_display['position'] = recent_signals_display['position'].map({1: 'BUY', -1: 'SELL', 0: 'HOLD'})
        recent_signals_display['strategy_returns_net'] = (recent_signals_display['strategy_returns_net'] * 100).round(2).astype(str) + '%'
        recent_signals_display['probability'] = (recent_signals_display['probability'] * 100).round(1).astype(str) + '%'
        recent_signals_display.columns = ['Price', 'Signal', 'Confidence', 'Return']
        st.dataframe(recent_signals_display, use_container_width=True)
    
    with tab4:
        st.subheader("🧠 Model Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Training Accuracy", f"{st.session_state.train_score:.1%}")
            st.metric("Test Accuracy", f"{st.session_state.test_score:.1%}")
        
        with col2:
            # Feature importance chart
            st.subheader("🎯 Top 10 Important Features")
            top_features = st.session_state.feature_importance.head(10)
            
            fig_importance = go.Figure(go.Bar(
                x=top_features['importance'],
                y=top_features['feature'],
                orientation='h'
            ))
            
            fig_importance.update_layout(
                height=400,
                xaxis_title="Importance",
                yaxis_title="Feature"
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Technical indicators
        st.subheader("📊 Current Technical Indicators")
        
        current_data = st.session_state.strategy.processed_data.iloc[-1]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("RSI", f"{current_data['RSI']:.1f}")
            st.metric("MACD", f"{current_data['MACD']:.4f}")
        
        with col2:
            st.metric("SMA 20", f"${current_data['SMA_20']:.2f}")
            st.metric("SMA 50", f"${current_data['SMA_50']:.2f}")
        
        with col3:
            st.metric("Volatility", f"{current_data['volatility']:.4f}")
            st.metric("Volume Ratio", f"{current_data['volume_ratio']:.2f}")
        
        with col4:
            st.metric("BB Position", f"{current_data['BB_position']:.2f}")
            st.metric("Trend Strength", f"{current_data['trend_strength']:.4f}")
    
    with tab5:
        st.subheader("⚡ Live Trading Dashboard")
        
        # Real-time quote
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Last Price", f"${info.get('currentPrice', 'N/A')}")
        
        with col2:
            day_change = info.get('regularMarketChangePercent', 0)
            st.metric("Day Change", f"{day_change:.2f}%")
        
        with col3:
            st.metric("Volume", f"{info.get('volume', 'N/A'):,}")
        
        with col4:
            st.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:.1f}B")
        
        # Trading recommendations
        st.subheader("🎯 Trading Recommendations")
        
        signal = st.session_state.current_signal
        confidence = st.session_state.confidence
        current_price = st.session_state.strategy.processed_data['Close'].iloc[-1]
        
        if signal == "BUY":
            st.success(f"**BUY Signal** - Confidence: {confidence:.1%}")
            entry_price = current_price
            stop_loss_price = entry_price * (1 - stop_loss)
            take_profit_price = entry_price * (1 + take_profit)
            
            st.write(f"**Entry Price:** ${entry_price:.2f}")
            st.write(f"**Stop Loss:** ${stop_loss_price:.2f} (-{stop_loss:.1%})")
            st.write(f"**Take Profit:** ${take_profit_price:.2f} (+{take_profit:.1%})")
            
        elif signal == "SELL":
            st.error(f"**SELL Signal** - Confidence: {confidence:.1%}")
            st.write("Consider closing long positions or opening short positions")
            
        else:
            st.warning(f"**HOLD Signal** - Confidence: {confidence:.1%}")
            st.write("No clear trading opportunity at this time")
        
        # Risk management
        st.subheader("💰 Position Sizing Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            account_value = st.number_input("Account Value ($)", value=initial_capital, step=1000)
            risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0) / 100
        
        with col2:
            risk_amount = account_value * risk_per_trade
            shares = int(risk_amount / (current_price * stop_loss))
            position_value = shares * current_price
            
            st.metric("Risk Amount", f"${risk_amount:.2f}")
            st.metric("Shares to Buy", f"{shares:,}")
            st.metric("Position Value", f"${position_value:,.2f}")

else:
    # Welcome screen
    st.info("👈 Configure your strategy parameters in the sidebar and click 'Run Strategy' to begin!")
    
    st.subheader("📚 How to Use This Dashboard")
    
    st.markdown("""
    1. **Configure Strategy**: Set your preferred stock symbol and parameters in the sidebar
    2. **Run Strategy**: Click the 'Run Strategy' button to train the model and generate signals
    3. **Analyze Results**: Review performance metrics, charts, and trading signals
    4. **Monitor Signals**: Check the current trading signal and confidence level
    5. **Manage Risk**: Use the position sizing calculator to determine appropriate trade sizes
    """)
    
    st.subheader("🎯 Strategy Overview")
    
    st.markdown("""
    This ML trading strategy uses:
    - **Random Forest** machine learning model
    - **20+ Technical Indicators** including RSI, MACD, Bollinger Bands
    - **Risk Management** with position sizing and stop-loss
    - **Backtesting Engine** to evaluate historical performance
    - **Real-time Signals** for current market conditions
    """)
    
    st.subheader("⚠️ Risk Disclaimer")
    
    st.warning("""
    **Important**: This is for educational purposes only. Trading involves substantial risk of loss. 
    Past performance does not guarantee future results. Always do your own research and consider 
    consulting with a financial advisor before making investment decisions.
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Powered by Machine Learning • Data from Yahoo Finance")