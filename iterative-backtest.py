from iterative_base import IterativeBase
import numpy as np

class IterativeBacktest(IterativeBase):
    ''' Class for iterative (event-driven) backtesting of trading strategies.
    '''

    # helper method
    def go_long(self, bar, units=None, amount=None):
        '''
        Go long (buy) at the given bar.
        
        Parameters
        ----------
        bar: int
            index (row number) for the current bar
        units: int
            number of units to buy
        amount: float or str
            amount to invest in monetary terms, or "all" to use all available balance
        '''
        if self.position == -1:
            self.buy_instrument(bar, units=-self.units)  # if short position, go neutral first
        if units:
            self.buy_instrument(bar, units=units)
        elif amount:
            if amount == "all":
                amount = self.current_balance
            self.buy_instrument(bar, amount=amount)  # go long

    # helper method
    def go_short(self, bar, units=None, amount=None):
        '''
        Go short (sell) at the given bar.
        
        Parameters
        ----------
        bar: int
            index (row number) for the current bar
        units: int
            number of units to sell
        amount: float or str
            amount to sell in monetary terms, or "all" to use all available balance
        '''
        if self.position == 1:
            self.sell_instrument(bar, units=self.units)  # if long position, go neutral first
        if units:
            self.sell_instrument(bar, units=units)
        elif amount:
            if amount == "all":
                amount = self.current_balance
            self.sell_instrument(bar, amount=amount)  # go short

    def test_sma_strategy(self, SMA_S, SMA_L):
        ''' 
        Backtests an SMA crossover strategy with SMA_S (short) and SMA_L (long).
        
        Parameters
        ----------
        SMA_S: int
            moving window in bars (e.g. days) for shorter SMA
        SMA_L: int
            moving window in bars (e.g. days) for longer SMA
            
        Returns
        -------
        dict
            performance summary dictionary
        '''
        
        # nice printout
        stm = "Testing SMA strategy | {} | SMA_S = {} & SMA_L = {}".format(self.symbol, SMA_S, SMA_L)
        print("-" * 75)
        print(stm)
        print("-" * 75)
        
        # reset 
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.results = []  # reset results
        
        # prepare data
        self.data["SMA_S"] = self.data["price"].rolling(SMA_S).mean()
        self.data["SMA_L"] = self.data["price"].rolling(SMA_L).mean()
        self.data.dropna(inplace=True)

        # Track positions for plotting
        self.data["position"] = 0

        # sma crossover strategy
        for bar in range(len(self.data)-1):  # all bars (except the last bar)
            if self.data["SMA_S"].iloc[bar] > self.data["SMA_L"].iloc[bar]:  # signal to go long
                if self.position in [0, -1]:
                    self.go_long(bar, amount="all")  # go long with full amount
                    self.position = 1  # long position
            elif self.data["SMA_S"].iloc[bar] < self.data["SMA_L"].iloc[bar]:  # signal to go short
                if self.position in [0, 1]:
                    self.go_short(bar, amount="all")  # go short with full amount
                    self.position = -1  # short position
            
            # Record position for this bar
            self.data.iloc[bar, self.data.columns.get_loc("position")] = self.position
            
        # Record final position
        if bar + 1 < len(self.data):
            self.data.iloc[bar + 1, self.data.columns.get_loc("position")] = self.position
            
        self.close_pos(bar+1)  # close position at the last bar
        
        # Calculate strategy returns based on positions
        self.data["strategy"] = self.data["position"].shift(1) * self.data["returns"]
        self.data["cstrategy"] = self.data["strategy"].cumsum().apply(np.exp)
        
        # For comparison with buy-and-hold
        self.data["creturns"] = self.data["returns"].cumsum().apply(np.exp)
        
        return self.get_performance_summary()
        
        
    def test_con_strategy(self, window=1):
        ''' 
        Backtests a simple contrarian strategy.
        
        Parameters
        ----------
        window: int
            time window (number of bars) to be considered for the strategy.
            
        Returns
        -------
        dict
            performance summary dictionary
        '''
        
        # nice printout
        stm = "Testing Contrarian strategy | {} | Window = {}".format(self.symbol, window)
        print("-" * 75)
        print(stm)
        print("-" * 75)
        
        # reset 
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.results = []  # reset results
        
        # prepare data
        self.data["rolling_returns"] = self.data["returns"].rolling(window).mean()
        self.data.dropna(inplace=True)
        
        # Track positions for plotting
        self.data["position"] = 0
        
        # Contrarian strategy
        for bar in range(len(self.data)-1):  # all bars (except the last bar)
            if self.data["rolling_returns"].iloc[bar] <= 0:  # signal to go long
                if self.position in [0, -1]:
                    self.go_long(bar, amount="all")  # go long with full amount
                    self.position = 1  # long position
            elif self.data["rolling_returns"].iloc[bar] > 0:  # signal to go short
                if self.position in [0, 1]:
                    self.go_short(bar, amount="all")  # go short with full amount
                    self.position = -1  # short position
            
            # Record position for this bar
            self.data.iloc[bar, self.data.columns.get_loc("position")] = self.position
            
        # Record final position
        if bar + 1 < len(self.data):
            self.data.iloc[bar + 1, self.data.columns.get_loc("position")] = self.position
            
        self.close_pos(bar+1)  # close position at the last bar
        
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
        dev: float
            distance for Lower/Upper Bands in Standard Deviation units
            
        Returns
        -------
        dict
            performance summary dictionary
        '''
        
        # nice printout
        stm = "Testing Bollinger Bands Strategy | {} | SMA = {} & dev = {}".format(self.symbol, SMA, dev)
        print("-" * 75)
        print(stm)
        print("-" * 75)
        
        # reset 
        self.position = 0  # initial neutral position
        self.trades = 0  # no trades yet
        self.current_balance = self.initial_balance  # reset initial capital
        self.results = []  # reset results
        
        # prepare data
        self.data["SMA"] = self.data["price"].rolling(SMA).mean()
        self.data["Lower"] = self.data["SMA"] - self.data["price"].rolling(SMA).std() * dev
        self.data["Upper"] = self.data["SMA"] + self.data["price"].rolling(SMA).std() * dev
        self.data.dropna(inplace=True) 
        
        # Track positions for plotting
        self.data["position"] = 0
        
        # Bollinger strategy
        for bar in range(len(self.data)-1):  # all bars (except the last bar)
            if self.position == 0:  # when neutral
                if self.data["price"].iloc[bar] < self.data["Lower"].iloc[bar]:  # signal to go long
                    self.go_long(bar, amount="all")  # go long with full amount
                    self.position = 1  # long position
                elif self.data["price"].iloc[bar] > self.data["Upper"].iloc[bar]:  # signal to go Short
                    self.go_short(bar, amount="all")  # go short with full amount
                    self.position = -1  # short position
            elif self.position == 1:  # when long
                if self.data["price"].iloc[bar] > self.data["SMA"].iloc[bar]:
                    if self.data["price"].iloc[bar] > self.data["Upper"].iloc[bar]:  # signal to go short
                        self.go_short(bar, amount="all")  # go short with full amount
                        self.position = -1  # short position
                    else:
                        self.sell_instrument(bar, units=self.units)  # go neutral
                        self.position = 0
            elif self.position == -1:  # when short
                if self.data["price"].iloc[bar] < self.data["SMA"].iloc[bar]:
                    if self.data["price"].iloc[bar] < self.data["Lower"].iloc[bar]:  # signal to go long
                        self.go_long(bar, amount="all")  # go long with full amount
                        self.position = 1  # long position
                    else:
                        self.buy_instrument(bar, units=-self.units)  # go neutral
                        self.position = 0  
            
            # Record position for this bar
            self.data.iloc[bar, self.data.columns.get_loc("position")] = self.position
            
        # Record final position
        if bar + 1 < len(self.data):
            self.data.iloc[bar + 1, self.data.columns.get_loc("position")] = self.position
            
        self.close_pos(bar+1)  # close position at the last bar
        
        # Calculate strategy returns based on positions
        self.data["strategy"] = self.data["position"].shift(1) * self.data["returns"]
        self.data["cstrategy"] = self.data["strategy"].cumsum().apply(np.exp)
        
        # For comparison with buy-and-hold
        self.data["creturns"] = self.data["returns"].cumsum().apply(np.exp)
        
        return self.get_performance_summary()