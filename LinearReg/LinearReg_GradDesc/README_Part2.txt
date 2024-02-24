README (PART 2)

This is the README file for part2 of assignmnet
Implememtaion of SGDRegressor Linear model (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)


Please see below

1)The code has been written in python programming language

2) Requirement for running the code
a) All the below libraries must be present/installed.
Name                      Version          Build
numpy                     1.20.3           py39h4b4dc7a_0
pandas                    1.3.4            py39h743cdd8_0
scikit-image              0.18.3           py39hae1ba45_0  
scikit-learn              0.24.2           py39hb2f4e1b_0  
scipy                     1.7.1            py39h88652d9_2  
seaborn                   0.11.2             pyhd3eb1b0_0  
scikit-image              0.18.3           py39hae1ba45_0  
scikit-learn              0.24.2           py39hb2f4e1b_0  
matplotlib                3.4.3            py39hecd8cb5_0  
matplotlib-inline         0.1.2              pyhd3eb1b0_2 

3) For running the script, please navigate to directory where script has been saved. After running the script, plots and logs will be saved in the working directory. If using an editor like Spyder, plots and print statements will also be shown as stdout. 

4) we have used Average Localization Error (ALE) in sensor node localization process in WSNs Data Set (https://archive.ics.uci.edu/ml/datasets/Average+Localization+Error+%28ALE%29+in+sensor+node+localization+process+in+WSNs)

Number of instances in this data is 107
This data contains four features and one predictand. 
Features are; 
1. Anchor ratio 
2. Transmission range (measured in meters) 
3. Node density and 
4. Iteration 
The predictand is ALE (measured in meters)

5) We have used SGDRegressor linear model implementation . 
class sklearn.linear_model.SGDRegressor(loss='squared_error', *, penalty='None', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False)

Following parameters were considered
learning rate is invscaling. eta = eta0 / pow(t, power_t) (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html)
penalty = None
alpha = This is a contant term and is used to compute the learning rate, when learning rate is set to optimal. We tried varying this parameter. We found that there are other variables which may effect the computation of learning rate in this model. This is beacuse for learning rate option optimal, following equation is used
‘optimal’: eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a heuristic proposed by Leon Bottou.

6) For final report we have presented 
a) Log file : This logs the number of epochs and loss until convergence.
b) In the log file the last line presents MSE and R^2 for Training and testing data.
c) Residual vs predicted plot: This plot shows residuals vs predicted values
d) The distibution of 4 features have been presented as histogram
e) To study the correlation between different featues, pearson correlation was computed and has been shown as heatmap.


################### END #####################
