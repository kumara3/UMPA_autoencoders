
This is the README file for code



Please see below

1)	The code has been written in python programming language

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

3) For running the script, please navigate to directory where script has been saved. After running the script, plots and text files will be saved in the working directory. If using an editor like Spyder, plots and print statements will also be shown as stdout. 

4) we have used haberman's survival dataset (https://archive.ics.uci.edu/ml/datasets/haberman%27s+survival)

Number of instances in this data is 306
This data contains 3 features and a class.
Features are; 
1. Age of patient at time of operation (numerical) 
2. Patient's year of operation (year - 1900, numerical) 
3. Number of positive axillary nodes detected (numerical) 

The class is 
Survival status (class attribute) 
-- 1 = the patient survived 5 years or longer 
-- 2 = the patient died within 5 year

5) We have used MLP classifier https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
We have used various combination of hyperparameters in the model. The performance of models have been shown as the loss curve (plot of error with iterations). Additional, the code also generates tab separated file which shows the model performance in terms of training and test data accuracy and mean square error. 


6) For final report we have presented 
a) Tab separated files : This logs the accuracy and MSE of each model 
b) The loss curve showing the plot of error with number of iterations
d) The distribution of 3 features have been presented as histogram
e) pair plot to study the relation between dependent and independent variables.


################### END #####################
