import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")


class IterativeBase():
    ''' Base class for iterative (event-driven) backtesting of trading strategies.
    '''

    def __init__(self, symbol, start, end, amount, use_spread=True):
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
    
    def get_data(self):
        ''' Imports the data from a CSV file.
        This is a placeholder method that should be overridden or used with set_data.
        '''
        pass
    
    def set_data(self, data):
        ''' Sets the data directly from a dataframe.
        
        Parameters
        ----------
        data: pandas.DataFrame
            DataFrame containing price and optional spread data
        '''
        self.data = data.copy()
        if "returns" not in self.data.columns:
            self.data["returns"] = np.log(self.data.price / self.data.price.shift(1))

    def plot_data(self, cols=None):  
        ''' Plots the closing price for the symbol.
        
        Parameters
        ----------
        cols: str or list
            column name(s) to be plotted
        '''
        if cols is None:
            cols = "price"
        self.data[cols].plot(figsize=(12, 8), title=self.symbol)
    
    def get_values(self, bar):
        ''' Returns the date, the price and the spread for the given bar.
        
        Parameters
        ----------
        bar: int
            index (row number) to get the values from
            
        Returns
        -------
        tuple
            (date, price, spread)
        '''
        date = str(self.data.index[bar].date())
        price = round(self.data.price.iloc[bar], 5)
        spread = 0  # Default in case spread column doesn't exist
        if "spread" in self.data.columns:
            spread = round(self.data.spread.iloc[bar], 5)
        return date, price, spread
    
    def print_current_balance(self, bar):
        ''' Prints out the current (cash) balance.
        
        Parameters
        ----------
        bar: int
            index (row number) for the current bar
        '''
        date, price, spread = self.get_values(bar)
        print("{} | Current Balance: {}".format(date, round(self.current_balance, 2)))
        
    def buy_instrument(self, bar, units=None, amount=None):
        ''' Places and executes a buy order (market order).
        
        Parameters
        ----------
        bar: int
            index (row number) for the current bar
        units: int
            number of units to buy
        amount: float
            amount to invest in monetary terms
        '''
        date, price, spread = self.get_values(bar)
        if self.use_spread:
            price += spread/2  # ask price
        if amount is not None:  # use units if units are passed, otherwise calculate units
            units = int(amount / price)
        self.current_balance -= units * price  # reduce cash balance by "purchase price"
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
        
        print("{} |  Buying {} for {}".format(date, units, round(price, 5)))
    
    def sell_instrument(self, bar, units=None, amount=None):
        ''' Places and executes a sell order (market order).
        
        Parameters
        ----------
        bar: int
            index (row number) for the current bar
        units: int
            number of units to sell
        amount: float
            amount to sell in monetary terms
        '''
        date, price, spread = self.get_values(bar)
        if self.use_spread:
            price -= spread/2  # bid price
        if amount is not None:  # use units if units are passed, otherwise calculate units
            units = int(amount / price)
        self.current_balance += units * price  # increases cash balance by "purchase price"
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
        
        print("{} |  Selling {} for {}".format(date, units, round(price, 5)))
    
    def print_current_position_value(self, bar):
        ''' Prints out the current position value.
        
        Parameters
        ----------
        bar: int
            index (row number) for the current bar
        '''
        date, price, spread = self.get_values(bar)
        cpv = self.units * price
        print("{} |  Current Position Value = {}".format(date, round(cpv, 2)))
    
    def print_current_nav(self, bar):
        ''' Prints out the current net asset value (nav).
        
        Parameters
        ----------
        bar: int
            index (row number) for the current bar
        '''
        date, price, spread = self.get_values(bar)
        nav = self.current_balance + self.units * price
        print("{} |  Net Asset Value = {}".format(date, round(nav, 2)))
        
    def close_pos(self, bar):
        ''' Closes out a long or short position (go neutral).
        
        Parameters
        ----------
        bar: int
            index (row number) for the current bar
            
        Returns
        -------
        dict
            performance metrics dictionary
        '''
        date, price, spread = self.get_values(bar)
        print(75 * "-")
        print("{} | +++ CLOSING FINAL POSITION +++".format(date))
        self.current_balance += self.units * price  # closing final position (works with short and long!)
        self.current_balance -= (abs(self.units) * spread/2 * self.use_spread)  # subtract half-spread costs
        
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
        
        print("{} | closing position of {} for {}".format(date, self.units, price))
        self.units = 0  # setting position to neutral
        self.trades += 1
        perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        self.print_current_balance(bar)
        print("{} | net performance (%) = {}".format(date, round(perf, 2)))
        print("{} | number of trades executed = {}".format(date, self.trades))
        print(75 * "-")
        
        # Return performance metrics
        return {
            "final_balance": self.current_balance,
            "performance_pct": perf,
            "trades": self.trades
        }
    
    def get_performance_summary(self):
        '''Returns a summary of the strategy performance
        
        Returns
        -------
        dict
            performance summary dictionary
        '''
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