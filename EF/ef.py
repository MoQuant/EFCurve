tickers = ['MSFT','AAPL','GOOGL','AMZN','NVDA','CSCO',
           'ORCL','AMD','QCOM','BLK','JPM','NFLX','TSLA']


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def musd(x):
    m, n = x.shape
    mu = (1/m)*np.ones(m).dot(x)
    cv = (1/(m-1))*(x - mu).T.dot(x - mu)
    sd = np.sqrt(np.diag(cv))
    return sd, mu, cv

def MinVar(x):
    sd, mu, cvs = musd(x)
    cv = (2.0*cvs).tolist()
    n = len(cv)
    for i in range(n):
        cv[i].append(1.0)
    cv.append(np.ones(n).tolist() + [0])
    B = np.array(np.zeros(n).tolist() + [1.0])
    weights = np.linalg.inv(cv).dot(B)[:-1]
    risk_x = np.sqrt(weights.T.dot(cvs.dot(weights)))
    retz_x = weights.T.dot(mu)
    return risk_x, retz_x

def MaxSharpe(o):
    def Optimize(cv, mu):
        def Objective(x):
            return -(x.T.dot(mu))/(x.T.dot(cv.dot(x)))
        def Constraint(x):
            return np.sum(x) - 1.0

        cons = [{'type':'eq','fun':Constraint}]
        x = np.ones(len(mu))
        res = minimize(Objective, x, method='SLSQP', bounds=None, constraints=cons)
        return res.x
    
    sd, mu, cv = musd(o)
    weights = Optimize(cv, mu)
    risk_x = np.sqrt(weights.T.dot(cv.dot(weights)))
    retz_x = weights.T.dot(mu)

    return risk_x, retz_x

def EF(x):
    
    def Optimize(cov, mu, r):
        cov = (2.0*cov).tolist()
        n = len(cov)
        for i in range(n):
            cov[i].append(mu[i])
            cov[i].append(1.0)
        cov.append(mu.tolist() + [0, 0])
        cov.append(np.ones(n).tolist() + [0, 0])
        B = np.zeros(n).tolist() + [r, 1.0]
        cov, B = np.array(cov), np.array(B)
        return np.linalg.inv(cov).dot(B)[:-2]
    
    sd, mu, cv = musd(x)
    ux, uy = [], []
    for i in np.arange(np.min(mu), np.max(mu)+0.0001, 0.0001):
        weights = Optimize(cv, mu, i)
        ux.append(np.sqrt(weights.T.dot(cv.dot(weights))))
        uy.append(weights.T.dot(mu))

    return ux, uy

def CapitalAllocationLine(risk_x, risk_y):
    w = (0.5, 1, 1.5)
    x, y = [], []
    for weight in w:
        x.append(weight*risk_x)
        y.append(weight*risk_y + (1 - weight)*-risk_y)
    return x, y

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)


data = {tick:pd.read_csv(f'{tick}.csv') for tick in tickers}

close = np.array([data[tick]['adjClose'].values.tolist()[::-1] for tick in tickers]).T

ror = close[1:]/close[:-1] - 1.0

x, y, E = musd(ror)

for tick, xi, yi in zip(tickers, x, y):
    ax.scatter(xi, yi, color='blue', s=7)
    ax.annotate(tick, xy=(xi, yi))

ex, ey = EF(ror)
ax.plot(ex, ey, color='red', linewidth=0.8)

mx, my = MinVar(ror)
ax.scatter(mx, my, color='green', s=12)

sx, sy = MaxSharpe(ror)
ax.scatter(sx, sy, color='limegreen', s=12)

cx, cy = CapitalAllocationLine(sx, sy)
ax.plot(cx, cy, color='black', linewidth=0.8)

ax.set_xlabel('Risk (Standard Deviation)')
ax.set_ylabel('Average Daily Return')
ax.set_title('Effecient Frontier Chart')

plt.show()
    

