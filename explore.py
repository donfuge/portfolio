#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
from os import path, listdir
import seaborn as sb
import datetime

matplotlib.rc('figure', figsize=(8, 7))


DATADIR = "data"
RAWDATADIR = path.join(DATADIR,"raw")
stocks_to_drop = ["MKBBANK"]


price_col = "Utolsó ár"
change_col = "Változás"
date_col = "Dátum"
name_col = "Név"

#%%

files = [f for f in listdir(RAWDATADIR) if f.endswith('csv')]

print(f"Loading files: {files}")

stock_data = pd.DataFrame()

for file in files:
    data = pd.read_csv(path.join(RAWDATADIR,file), parse_dates=[date_col])
    stock_data = pd.concat([stock_data, data])

stock_names = set(stock_data[name_col])

stocks = {}


for name in stock_names:
    if name not in stocks_to_drop:
        stocks[name] = stock_data[stock_data[name_col] == name].reset_index(drop=True)
        stocks[name].sort_values(by=date_col, ascending=True, inplace=True)

print(f"Total number of stocks: {len(stocks)}")
print(stocks.keys())

# %% filling NaNs, calculating daily changes

popular_stocks = []

for name, stock in stocks.items():

    # fill NaNs in the price with last value 

    stock[price_col].fillna(method='ffill', inplace=True)

    # calulate normalized change (return)

    stock[change_col] = stock[price_col].pct_change()

    if stock[change_col].isna().sum() <= 1:
        popular_stocks.append(name)

    print(f"""NaNs in {name}: {stock[change_col].isna().sum()}, NaNs-to-total ratio: {stock[change_col].isna().sum() / stock[change_col].size:.3f}""")

print(f"Popular stocks: {popular_stocks}")
# %% filter for a time interval

start_date = pd.to_datetime(datetime.date(2001,1,1))
end_date = pd.to_datetime(datetime.date(2022,1,1))

stock = stocks[list(stocks.keys())[0]]
date_mask = (stock[date_col] >= start_date) & (stock[date_col] <= end_date)  


# %% investigate a selected stock

fig,axs = plt.subplots(2,1, sharex=True)

sel_stock_name = "GARDENIA"

sel_stock = stocks[sel_stock_name]

sel_stock.plot(x=date_col, y=price_col, title=sel_stock_name, ylabel=price_col, ax=axs[0])

sel_stock.plot(x=date_col, y=change_col, title=sel_stock_name, ylabel=change_col, ax=axs[1])

for ax in axs:
    ax.axvspan(start_date, end_date, facecolor='g', alpha=0.2)

plt.tight_layout()

#%% explore returns 

returns = pd.DataFrame()

for name, stock in stocks.items():
    if name in popular_stocks:
        returns.insert(0, name, stock[change_col][date_mask], allow_duplicates=True)

returns.corr()

plt.figure(figsize=(15,15))
sb.heatmap(returns.corr(), cmap="coolwarm", annot=True, center=0, annot_kws={"fontsize":8},
vmax=1, vmin=-1, fmt='.2f')

# %% risk vs return

# from https://towardsdatascience.com/in-12-minutes-stocks-analysis-with-pandas-and-scikit-learn-a8d8a7b50ee7

plt.figure(figsize=(15,15))

def plot_risk_mean(returns):
    plt.scatter(returns.mean(), returns.std())
    plt.xlabel('Expected returns')
    plt.ylabel('Risk')

    for label, x, y in zip(returns.columns, returns.mean(), returns.std()):
        plt.annotate(
            label, 
            xy = (x, y), xytext = (20, -20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

plot_risk_mean(returns)
#%%

from matplotlib import animation
def drawframe_correlation(n):
    pass

# blit=True re-draws only the parts that have changed.
anim = animation.FuncAnimation(fig, drawframe_correlation, frames=100, interval=20, blit=True)

#%% interactive test

%matplotlib notebook
from ipywidgets import interact
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def f(var):
    sns.distplot(np.random.normal(1, var, 1000))
    plt.show()
interact(f, var = (1,10))
# %%
