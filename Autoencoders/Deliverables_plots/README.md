

**###Dimensionality reduction in single cell data using autoencoders###**


**###Dependencies###**
```

1) The code has been written in python programming language
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

3) For running the script, please navigate to directory where script has been saved. After running the script, plots and logs will be saved in the working directory. If using an editor like Spyder, some plots and print statements will also be shown as stdout.

4) The data is a single cell gene expression dataset. The columns/features are number of genes and rows are numebr of cells/example.
Please download the zipped file; unzip it and give the path to filename in the url argument in __main__ part of script.  

Path to download: ./data/pbmc.counts.zip 
```

**###Name of the output directory:###*
After running you should see 3 folders generated. 
1. PCA_results : This folder has the pca plot. Please refer to the report.
2. TSNE_results: This folder has 
	       - The TSNE visulaization of PCA reduced data. This is named as pbmc.counts.TSNE_on_PCA.TSNE_plot.png
               - The TSNE visualization of latent/coding layer from the test data. Each file is named on the number of layer and learning rate used during creating and training model

3. Autoencoder_results: This folder has all the loss curve plot of training data for different experiments.


4. Deliverables include
   - final report
   - code (written in python: autoencoder.py)
   - The output folders from my run (as zipped file)
   - Inside the output folders, you will find loss curve plots, TSNE plots and PCA plots.Please refer to point 3 above for more information.

