##### INSTALLATIONS REQUIRED #####
1. Install the latest version of python available from python.org, however any version of python3.9+ should work well. Ensure the path variables are set for the same.
2. Open the command prompt or terminal. This can be done from run in windows and searching for terminal in mac.
3. Ensure that python is installed using the "python --version" command. After doing so execute the following pip commands:
    . python3 -m pip install pandas
    . python3 -m pip install numpy
    . python3 -m pip install matplotlib
    . python3 -m pip install -U scikit-learn
    . python3 -m pip install seaborn
    . python3 -m pip install wget
    . python3 -m pip install logging
##### RUNNING THE CODE #####
4. Navigate to the directory where the python files are saved. Can be done using the "cd" command.
5. Run the python file using the following command for part1:
    . python3 part1.py
6. Run the python file using the followoing command for part2:
    . python3 part2.py
##### NOTE #####
1. On running the code, multiple graphs will pop up, these are the plots of various parameters.
2. On closing the graphs the code will resume execution and the graphs will be saved in the same location as the .py file.
3. A log file containing parameters will be created in the same location called "logfile.log" for part1 and "logs.invscaling.txt" for part2. This can be opened using a text editor.
4. You must be connected to the internet to download the dataset file (mcs_ds_edited_iter_shuffled.csv).
##### PLOTS #####
1. MSE vs Iterations
2. Heatmap
3. Important feature vs Target
4. R squared value vs Iterations
5. RMSE vs Iterations
##### OUTPUT #####
1. part1.py - Generates the output of the linear regression model using gradient descent without library functions. This output is a dataframe which contains the learning rate, number of iterations, mean squared error, mean absolute error, root mean squared error and the R squared value. The same output is generated in the log file.
2. part2.py - Generates the output of the linear regression model using inbuilt library functions. This output is written into the logfile which contains the Mean Squared Error for the training and testing data as well as the R squared value.


################### END #######################

