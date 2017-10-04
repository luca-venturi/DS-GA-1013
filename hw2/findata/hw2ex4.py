import numpy as np
from findata_tools import *

names, days = load_data('stockprices.csv')
(nDays, nStocks) = days.shape
nReturns = nDays - 1
returns = days[1:] - days[:-1]

# (a)

returnsCentered = returns - np.mean(returns, axis=0)
U, S, returnsPrincipalDirections = np.linalg.svd(returnsCentered)
maxCoeffStocks_index = [np.argmax(np.abs(returnsPrincipalDirections[k])) for k in range(2)]
maxCoeffStocks = [names[k] for k in maxCoeffStocks_index]

print maxCoeffStocks
pretty_print(np.sum(np.abs(returns),axis = 0),names)

# (b)

U, S, returnsPrincipalDirections_stand = np.linalg.svd(np.corrcoef(returnsCentered,rowvar=False))
for k in range(2):
	pretty_print(returnsPrincipalDirections_stand[k],names)

# (c)


