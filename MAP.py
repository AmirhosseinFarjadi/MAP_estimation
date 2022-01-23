import numpy as np
from numpy import random
from numpy.core.fromnumeric import size
from scipy import stats
import scipy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
my_sample=np.random.normal(loc=20,scale=6,size=100)
avg_prior=16
variance_prior=4
for i in range(0,50):
    teta=my_sample[i]+avg_prior
teta=teta/1+100
def gaussian(parameters):
    avg=parameters[0]
    variance=parameters[1]
    sum_log=-np.sum(np.log(stats.norm.pdf(teta,loc=avg_prior,scale=variance_prior) * stats.norm.pdf(my_sample,loc=avg,scale=variance)))
    return sum_log

initial=[5,5]
output=minimize(gaussian,initial).x
sigma = output[0]
mu = output[1]
print('sigma= ',output[0])
print('mu=',output[1])
print('mean - teta= ',output[0]-20)
count, bins, ignored = plt.hist(my_sample, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * 
         np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.show()