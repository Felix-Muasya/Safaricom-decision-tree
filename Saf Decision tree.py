# importing the libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# making our dataframe got the data from https://www.marketscreener.com/quote/stock/SAFARICOM-PLC-6500172/quotes/
data = {"Capitalization": [1103803, 1059731, 1452372],
        "Enterprise Value (EV)": [1079763, 1040971, 1440408],
        "P/E ratio": [17.7, 14.4, 21.2],
        "Yield": [4.54, 5.29, 3.78],
        "Capitalize/Revenue": [4.41, 4.04, 5.50],
        "EV/Revenue": [4.31, 3.96, 5.46],
        "EV/EBITDA": [8.69, 7.54, 10.7],
        "Price to Book": [7.65, 7.40, 10.6],
        "N.O of Stocks": [40065428, 40065428, 40065428],
        "Reference Price": [27.6, 26.5, 36.3],
        "EBITDA": [124300, 138040, 264027],
        "Net Sales": [250280, 262560, 264027],
        "Operating Profit": [88970, 101490, 96165],
        "Dividends Per Share": [1.25, 1.40, 1.37],
        "Net Income": [62490, 73660, 68676],
        "OPY?": [1, 0, 1]} # OPY outperform previous year 1 means yes, 0 means no

df = pd.DataFrame(data, index=[2019, 2020, 2021])
df.to_csv("Saf_3_year_financials.csv")

features = ['Capitalization',
            'Enterprise Value (EV)',
            'P/E ratio',
            'Yield',
            'Capitalize/Revenue',
            'EV/Revenue',
            'EV/EBITDA',
            'Price to Book',
            'N.O of Stocks',
            'Reference Price',
            'EBITDA',
            'Net Sales',
            'Operating Profit',
            'Dividends Per Share',
            'Net Income']
df = pd.read_csv("Saf_3_year_financials.csv")

X = df[features]
Y = df['OPY?']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, Y)

# predicting whether saf will outperform the previous year based on the following info
""" 
Capitalization = 1,400,000
Enterprise Value (EV) = 1,410,000
P/E ratio = 19
Yield = 4.4
Capitalize/Revenue = 4.7
EV/Revenue = 4.75
EV/EBITDA = 9.3
Price to Book = 7.8
N.O of Stocks = 40,065,428
Reference Price = 35
EBITDA = 151,027
Net Sales = 297,038
Operating Profit = 111,637
Dividends Per Share = 1.53
Net Income = 74063
"""

info = [[1400000, 1410000, 19, 4.4, 4.7, 4.75, 9.3, 7.8, 40065428, 35, 151027, 297038, 111637, 1.53, 74063]]
print(dtree.predict(info))

print("[0] mean it won't outperform")
print("[1] mean it will outperform")