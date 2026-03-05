"""
RSI-entry simple backtester
- Buy when RSI < 30
- Exit (sell) when RSI >= 50
- Uses vectorized pandas operations and simulates next-bar fills (open price)
- Includes position sizing, commission (per trade), and slippage (pct)
- Produces basic performance metrics

Requirements:
pip install pandas numpy matplotlib
(You can optionally install TA-Lib, but this code implements RSI directly.)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import floor

# ---------- CONFIG ----------
RSI_PERIOD = 14
RSI_ENTRY = 30        # buy when RSI < this
RSI_EXIT = 50         # sell when RSI >= this
INITIAL_CAPITAL = 10000.0  # starting capital in currency units
RISK_PER_TRADE = 0.02  # fraction of capital to risk per trade (used for sizing guidance)
POSITION_SIZE_PCT = 0.10  # max position size as fraction of portfolio (10% default)
COMMISSION_PER_TRADE = 1.0  # flat commission per trade (adjust to your broker)
SLIPPAGE_PCT = 0.0005  # e.g., 0.05% slippage
MIN_SHARES = 1  # minimum tradable size
# ----------------------------

def compute_rsi(close, period=14):
    # Wilder's smoothing method
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -1 * delta.clip(upper=0.0)
    # first average (simple)
    roll_up = up.rolling(window=period, min_periods=period).mean()
    roll_down = down.rolling(window=period, min_periods=period).mean()
    # then smooth using Wilder's method
    # after the first calculation use exponential smoothing:
    avg_gain = roll_up.copy()
    avg_loss = roll_down.copy()
    for i in range(period, len(close)):
        if i == period:
            # already set the initial SMA-based values
            continue
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + up.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + down.iloc[i]) / period
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def backtest(df, initial_capital=INITIAL_CAPITAL, position_size_pct=POSITION_SIZE_PCT):
    """
    df must contain: index = datetime, columns: ['open','high','low','close','volume']
    Strategy entry/exit uses next-bar open for execution (common realistic assumption)
    """
    df = df.copy().dropna(subset=['open','close'])
    df['rsi'] = compute_rsi(df['close'], period=RSI_PERIOD)
    # signals computed on bar close; we act next bar open
    df['signal'] = 0  # 1 means long, 0 flat
    # buy signal when rsi < RSI_ENTRY
    df.loc[df['rsi'] < RSI_ENTRY, 'signal'] = 1
    # sell/exit when rsi >= RSI_EXIT -> signal 0
    # But we need to convert this into "positions" where a buy after flat and hold until exit
    position = 0
    positions = []
    for i in range(len(df)):
        r = df['rsi'].iat[i]
        if np.isnan(r):
            positions.append(position)
            continue
        if position == 0 and r < RSI_ENTRY:
            position = 1
        elif position == 1 and r >= RSI_EXIT:
            position = 0
        positions.append(position)
    df['position_raw'] = positions
    # we will execute entries/exits at next bar open -- shift position to get "execution" rows
    df['position'] = df['position_raw'].shift(1).fillna(0)  # position held during bar
    # trades: when position changes (execution at that bar's open)
    df['trade'] = df['position'].diff().fillna(0)  # +1 = buy executed at this bar's open, -1 = sell at open

    cash = initial_capital
    shares = 0
    equity_curve = []
    trade_log = []

    for idx, row in df.iterrows():
        # if a trade to execute at this bar's open:
        if row['trade'] == 1:  # BUY at this bar's open
            # position sizing: buy up to position_size_pct of current equity
            equity = cash + shares * row['open']
            max_position_value = equity * position_size_pct
            desired_shares = floor(max_position_value / row['open'])
            if desired_shares < MIN_SHARES:
                desired_shares = 0
            if desired_shares > 0:
                fill_price = row['open'] * (1 + SLIPPAGE_PCT)  # slippage on buy
                cost = desired_shares * fill_price + COMMISSION_PER_TRADE
                if cost <= cash:
                    cash -= cost
                    shares += desired_shares
                    trade_log.append({'datetime': idx, 'type': 'BUY', 'price': fill_price, 'shares': desired_shares, 'cash': cash})
        elif row['trade'] == -1:  # SELL at this bar's open (close entire position)
            if shares > 0:
                fill_price = row['open'] * (1 - SLIPPAGE_PCT)  # slippage on sell
                proceeds = shares * fill_price - COMMISSION_PER_TRADE
                cash += proceeds
                trade_log.append({'datetime': idx, 'type': 'SELL', 'price': fill_price, 'shares': shares, 'cash': cash})
                shares = 0
        # update equity
        current_equity = cash + shares * row['close']  # mark-to-market using close
        equity_curve.append(current_equity)

    df = df.iloc[len(df)-len(equity_curve):].copy()
    df['equity'] = equity_curve

    # Performance metrics
    total_return = df['equity'].iloc[-1] / initial_capital - 1.0
    # CAGR
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25 if days > 0 else 1/252.0
    cagr = (df['equity'].iloc[-1] / initial_capital) ** (1.0 / years) - 1.0
    # Max drawdown
    roll_max = df['equity'].cummax()
    drawdown = (df['equity'] - roll_max) / roll_max
    max_dd = drawdown.min()
    # Sharpe (annualized) assuming daily returns
    df['returns'] = df['equity'].pct_change().fillna(0)
    if df['returns'].std() == 0:
        sharpe = np.nan
    else:
        sharpe = (df['returns'].mean() / df['returns'].std()) * np.sqrt(252)

    results = {
        'initial_capital': initial_capital,
        'final_equity': df['equity'].iloc[-1],
        'total_return': total_return,
        'cagr': cagr,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'trades': trade_log,
        'equity_curve': df['equity'],
        'df': df
    }
    return results

# -------------------------
# Example usage:
# -------------------------
if __name__ == "__main__":
    # Load your historical data CSV (must have open,high,low,close,volume and datetime index)
    # Example CSV format: Date,Open,High,Low,Close,Volume
    datafile = "E:\Trading\TCS.NS_5min_data.csv"  # replace with your file
    df = pd.read_csv(datafile, parse_dates=['Date'], index_col='Date')
    df.columns = [c.lower() for c in df.columns]  # normalize
    # ensure required columns exist
    for col in ['open','high','low','close']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    res = backtest(df)
    print("Initial capital:", res['initial_capital'])
    print("Final equity:", res['final_equity'])
    print("Total return %:", round(res['total_return']*100,2))
    print("CAGR %:", round(res['cagr']*100,2))
    print("Max Drawdown %:", round(res['max_drawdown']*100,2))
    print("Sharpe Ratio (ann):", round(res['sharpe'], 3))
    print("Number of trades:", len(res['trades']))
    # Plot equity curve
    plt.figure(figsize=(10,5))
    res['equity_curve'].plot()
    plt.title("Equity Curve")
    plt.ylabel("Portfolio Value")
    plt.show()
