import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from itertools import product
import io
import base64
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")

# Import the iterative backtesting classes
class IterativeBase():
    ''' Base class for iterative (event-driven) backtesting of trading strategies.
    '''

    def __init__(self, symbol, start, end, amount, use_spread = True):
        '''
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        start: str
            start date for data import
        end: str
            end date for data import
        amount: float
            initial amount to be invested per trade
        use_spread: boolean (default = True) 
            whether trading costs (bid-ask spread) are included
        '''
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_balance = amount
        self.current_balance = amount
        self.units = 0
        self.trades = 0
        self.position = 0
        self.use_spread = use_spread
        self.results = []  # To store trade results
        self.get_data()
    
    def get_data(self):
        ''' Imports the data.
        '''
        # This will be overridden to use the uploaded or sample data
        pass

    def set_data(self, data):
        ''' Sets the data directly from a dataframe.
        '''
        self.data = data.copy()
        if "returns" not in self.data.columns:
            self.data["returns"] = np.log(self.data.price / self.data.price.shift(1))

    def plot_data(self, cols = None):  
        ''' Plots the closing price for the symbol.
        '''
        if cols is None:
            cols = "price"
        self.data[cols].plot(figsize = (12, 8), title = self.symbol)
    
    def get_values(self, bar):
        ''' Returns the date, the price and the spread for the given bar.
        '''
        date = str(self.data.index[bar].date())
        price = round(self.data.price.iloc[bar], 5)
        spread = 0  # Default in case spread column doesn't exist
        if "spread" in self.data.columns:
            spread = round(self.data.spread.iloc[bar], 5)
        return date, price, spread
    
    def print_current_balance(self, bar):
        ''' Prints out the current (cash) balance.
        '''
        date, price, spread = self.get_values(bar)
        print("{} | Current Balance: {}".format(date, round(self.current_balance, 2)))
        
    def buy_instrument(self, bar, units = None, amount = None):
        ''' Places and executes a buy order (market order).
        '''
        date, price, spread = self.get_values(bar)
        if self.use_spread:
            price += spread/2 # ask price
        if amount is not None: # use units if units are passed, otherwise calculate units
            units = int(amount / price)
        self.current_balance -= units * price # reduce cash balance by "purchase price"
        self.units += units
        self.trades += 1
        
        # Save trade details
        self.results.append({
            "date": self.data.index[bar],
            "price": price,
            "type": "buy",
            "units": units,
            "value": units * price,
            "balance": self.current_balance,
            "position": self.units
        })
        
        #print("{} |  Buying {} for {}".format(date, units, round(price, 5)))
    
    def sell_instrument(self, bar, units = None, amount = None):
        ''' Places and executes a sell order (market order).
        '''
        date, price, spread = self.get_values(bar)
        if self.use_spread:
            price -= spread/2 # bid price
        if amount is not None: # use units if units are passed, otherwise calculate units
            units = int(amount / price)
        self.current_balance += units * price # increases cash balance by "purchase price"
        self.units -= units
        self.trades += 1
        
        # Save trade details
        self.results.append({
            "date": self.data.index[bar],
            "price": price,
            "type": "sell",
            "units": units,
            "value": units * price,
            "balance": self.current_balance,
            "position": self.units
        })
        
        #print("{} |  Selling {} for {}".format(date, units, round(price, 5)))
    
    def print_current_position_value(self, bar):
        ''' Prints out the current position value.
        '''
        date, price, spread = self.get_values(bar)
        cpv = self.units * price
        print("{} |  Current Position Value = {}".format(date, round(cpv, 2)))
    
    def print_current_nav(self, bar):
        ''' Prints out the current net asset value (nav).
        '''
        date, price, spread = self.get_values(bar)
        nav = self.current_balance + self.units * price
        print("{} |  Net Asset Value = {}".format(date, round(nav, 2)))
        
    def close_pos(self, bar):
        ''' Closes out a long or short position (go neutral).
        '''
        date, price, spread = self.get_values(bar)
        #print(75 * "-")
        #print("{} | +++ CLOSING FINAL POSITION +++".format(date))
        self.current_balance += self.units * price # closing final position (works with short and long!)
        self.current_balance -= (abs(self.units) * spread/2 * self.use_spread) # substract half-spread costs
        
        # Save closing trade details
        if self.units != 0:
            self.results.append({
                "date": self.data.index[bar],
                "price": price,
                "type": "close",
                "units": self.units,
                "value": self.units * price,
                "balance": self.current_balance,
                "position": 0
            })
        
        #print("{} | closing position of {} for {}".format(date, self.units, price))
        self.units = 0 # setting position to neutral
        self.trades += 1
        perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        #self.print_current_balance(bar)
        #print("{} | net performance (%) = {}".format(date, round(perf, 2) ))
        #print("{} | number of trades executed = {}".format(date, self.trades))
        #print(75 * "-")
        
        # Return performance metrics
        return {
            "final_balance": self.current_balance,
            "performance_pct": perf,
            "trades": self.trades
        }
    
    def get_performance_summary(self):
        '''Returns a summary of the strategy performance'''
        if not self.results:
            return None
        
        trades_df = pd.DataFrame(self.results)
        if trades_df.empty:
            return None
        
        initial = self.initial_balance
        final = self.current_balance
        
        return {
            "initial_balance": initial,
            "final_balance": final,
            "return_pct": (final - initial) / initial * 100,
            "trades": self.trades,
            "trades_df": trades_df
        }


class IterativeBacktest(IterativeBase):
    ''' Class for iterative (event-driven) backtesting of trading strategies.
    '''

    # helper method
    def go_long(self, bar, units = None, amount = None):
        if self.position == -1:
            self.buy_instrument(bar, units = -self.units) # if short position, go neutral first
        if units:
            self.buy_instrument(bar, units = units)
        elif amount:
            if amount == "all":
                amount = self.current_balance
            self.buy_instrument(bar, amount = amount) # go long

    # helper method
    def go_short(self, bar, units = None, amount = None):
        if self.position == 1:
            self.sell_instrument(bar, units = self.units) # if long position, go neutral first
        if units:
            self.sell_instrument(bar, units = units)
        elif amount:
            if amount == "all":
                amount = self.current_balance
            self.sell_instrument(bar, amount = amount) # go short

    def test_sma_strategy(self, SMA_S, SMA_L):
        ''' 
        Backtests an SMA crossover strategy with SMA_S (short) and SMA_L (long).
        
        Parameters
        ----------
        SMA_S: int
            moving window in bars (e.g. days) for shorter SMA
        SMA_L: int
            moving window in bars (e.g. days) for longer SMA
        '''
        
        # reset 
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.results = []  # reset results
        
        # prepare data
        self.data["SMA_S"] = self.data["price"].rolling(SMA_S).mean()
        self.data["SMA_L"] = self.data["price"].rolling(SMA_L).mean()
        self.data.dropna(inplace = True)

        # Track positions for plotting
        self.data["position"] = 0

        # sma crossover strategy
        for bar in range(len(self.data)-1): # all bars (except the last bar)
            if self.data["SMA_S"].iloc[bar] > self.data["SMA_L"].iloc[bar]: # signal to go long
                if self.position in [0, -1]:
                    self.go_long(bar, amount = "all") # go long with full amount
                    self.position = 1  # long position
            elif self.data["SMA_S"].iloc[bar] < self.data["SMA_L"].iloc[bar]: # signal to go short
                if self.position in [0, 1]:
                    self.go_short(bar, amount = "all") # go short with full amount
                    self.position = -1 # short position
            
            # Record position for this bar
            self.data.iloc[bar, self.data.columns.get_loc("position")] = self.position
            
        # Record final position
        if bar + 1 < len(self.data):
            self.data.iloc[bar + 1, self.data.columns.get_loc("position")] = self.position
            
        self.close_pos(bar+1) # close position at the last bar
        
        # Calculate strategy returns based on positions
        self.data["strategy"] = self.data["position"].shift(1) * self.data["returns"]
        self.data["cstrategy"] = self.data["strategy"].cumsum().apply(np.exp)
        
        # For comparison with buy-and-hold
        self.data["creturns"] = self.data["returns"].cumsum().apply(np.exp)
        
        return self.get_performance_summary()
        
        
    def test_con_strategy(self, window = 1):
        ''' 
        Backtests a simple contrarian strategy.
        
        Parameters
        ----------
        window: int
            time window (number of bars) to be considered for the strategy.
        '''
        
        # reset 
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.results = []  # reset results
        
        # prepare data
        self.data["rolling_returns"] = self.data["returns"].rolling(window).mean()
        self.data.dropna(inplace = True)
        
        # Track positions for plotting
        self.data["position"] = 0
        
        # Contrarian strategy
        for bar in range(len(self.data)-1): # all bars (except the last bar)
            if self.data["rolling_returns"].iloc[bar] <= 0: #signal to go long
                if self.position in [0, -1]:
                    self.go_long(bar, amount = "all") # go long with full amount
                    self.position = 1  # long position
            elif self.data["rolling_returns"].iloc[bar] > 0: #signal to go short
                if self.position in [0, 1]:
                    self.go_short(bar, amount = "all") # go short with full amount
                    self.position = -1 # short position
            
            # Record position for this bar
            self.data.iloc[bar, self.data.columns.get_loc("position")] = self.position
            
        # Record final position
        if bar + 1 < len(self.data):
            self.data.iloc[bar + 1, self.data.columns.get_loc("position")] = self.position
            
        self.close_pos(bar+1) # close position at the last bar
        
        # Calculate strategy returns based on positions
        self.data["strategy"] = self.data["position"].shift(1) * self.data["returns"]
        self.data["cstrategy"] = self.data["strategy"].cumsum().apply(np.exp)
        
        # For comparison with buy-and-hold
        self.data["creturns"] = self.data["returns"].cumsum().apply(np.exp)
        
        return self.get_performance_summary()
        
        
    def test_boll_strategy(self, SMA, dev):
        ''' 
        Backtests a Bollinger Bands mean-reversion strategy.
        
        Parameters
        ----------
        SMA: int
            moving window in bars (e.g. days) for simple moving average.
        dev: int
            distance for Lower/Upper Bands in Standard Deviation units
        '''
        
        # reset 
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.results = []  # reset results
        
        # prepare data
        self.data["SMA"] = self.data["price"].rolling(SMA).mean()
        self.data["Lower"] = self.data["SMA"] - self.data["price"].rolling(SMA).std() * dev
        self.data["Upper"] = self.data["SMA"] + self.data["price"].rolling(SMA).std() * dev
        self.data.dropna(inplace = True) 
        
        # Track positions for plotting
        self.data["position"] = 0
        
        # Bollinger strategy
        for bar in range(len(self.data)-1): # all bars (except the last bar)
            if self.position == 0: # when neutral
                if self.data["price"].iloc[bar] < self.data["Lower"].iloc[bar]: # signal to go long
                    self.go_long(bar, amount = "all") # go long with full amount
                    self.position = 1  # long position
                elif self.data["price"].iloc[bar] > self.data["Upper"].iloc[bar]: # signal to go Short
                    self.go_short(bar, amount = "all") # go short with full amount
                    self.position = -1 # short position
            elif self.position == 1: # when long
                if self.data["price"].iloc[bar] > self.data["SMA"].iloc[bar]:
                    if self.data["price"].iloc[bar] > self.data["Upper"].iloc[bar]: # signal to go short
                        self.go_short(bar, amount = "all") # go short with full amount
                        self.position = -1 # short position
                    else:
                        self.sell_instrument(bar, units = self.units) # go neutral
                        self.position = 0
            elif self.position == -1: # when short
                if self.data["price"].iloc[bar] < self.data["SMA"].iloc[bar]:
                    if self.data["price"].iloc[bar] < self.data["Lower"].iloc[bar]: # signal to go long
                        self.go_long(bar, amount = "all") # go long with full amount
                        self.position = 1 # long position
                    else:
                        self.buy_instrument(bar, units = -self.units) # go neutral
                        self.position = 0  
            
            # Record position for this bar
            self.data.iloc[bar, self.data.columns.get_loc("position")] = self.position
            
        # Record final position
        if bar + 1 < len(self.data):
            self.data.iloc[bar + 1, self.data.columns.get_loc("position")] = self.position
            
        self.close_pos(bar+1) # close position at the last bar
        
        # Calculate strategy returns based on positions
        self.data["strategy"] = self.data["position"].shift(1) * self.data["returns"]
        self.data["cstrategy"] = self.data["strategy"].cumsum().apply(np.exp)
        
        # For comparison with buy-and-hold
        self.data["creturns"] = self.data["returns"].cumsum().apply(np.exp)
        
        return self.get_performance_summary()

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
    else:
        comparison_pairs = [selected_pair]

# Function to download dataframe as CSV
def get_table_downloa