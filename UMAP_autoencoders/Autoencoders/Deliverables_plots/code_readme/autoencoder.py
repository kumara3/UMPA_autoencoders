# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 12:15:24 2022

@author: Ashwani kumar
NETID AXK200017
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 17:03:16 2022

@author: Ashwani
"""
import math
import os 
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


class Autoencoder():
    def __init__(self,layers,activation_name,lr,epochs,dataFile,components):
        """ 
        Modeling Autoencoder class with encoder, decoder and bottle neck layers
        """
        self.dataFile = dataFile
        self.raw_input = pd.read_csv(dataFile,header=None, sep="\t")
        self.layers = layers
        self.activation_name = activation_name
        self.lr = lr
        self.epochs = epochs
        self.W = []
        self.loss = []
        self.components = components
        self.batch_size = 100
        #self.prefix = prefix
        
        for i in np.arange(0,len(self.layers) -2):
            w = self.wt_initialize(self.layers[i] + 1,self.layers[i+1] + 1)
            self.W.append(w / np.sqrt(self.layers[i]))
        w = self.wt_initialize(self.layers[-2]+1,self.layers[-1])
        self.W.append(w / np.sqrt(self.layers[-2]))
    
        
    def preprocessing(self):
        self.preprocessed_data = self.raw_input
        
        #Describe data
        
        print("\n" + "Dimensions of input file: " + str(self.preprocessed_data) + "\n")
        print("\n" + "first few lines of input file: " + "\n")
        print(self.preprocessed_data.iloc[0:4, 0:4])
        print("\n" + "Last column is teh cluster assignment " + "\n")
        print(self.preprocessed_data.iloc[0:4, (self.preprocessed_data.shape[1]-4):self.preprocessed_data.shape[1]])
    
        X = self.preprocessed_data.values[1:,1:(self.preprocessed_data.shape[1])]
        X = X.astype(np.float)
    
        ## Log transform
        X = np.float32(np.log(X + 1) )
        
        return X  
    
     
    def dimension_reduction_PCA(self):
        # Peform the dimesnionality reduction using PCA
        
        #X_train, Y_label = self.preprocessing()
        X = self.preprocessing()
        print("\n"+ "Principal component Analysis in progress (PCA)...." + "\n")
        pca_x = PCA(self.components)
        pca_x_train = pca_x.fit_transform(X)
        
        ## Genearting the pca results in tabular form ##
        pca_df = pd.DataFrame(data = pca_x_train)
        
        #make directory and save files
        try:
            if not os.path.exists("PCA_results"):
                os.makedirs("PCA_results")
        except OSError:
            print('Error : creating directory'+"PCA_results")
        path_wo_ext = self.dataFile.rsplit('.',1)[0]
        path_prefix = os.path.basename(path_wo_ext)
        dir_pca = os.path.join("PCA_results",path_prefix+".principal_components.csv")
        pca_out_file = open(dir_pca,"w")
        pca_df.to_csv(pca_out_file)
        
        dir_pca = os.path.join("PCA_results",path_prefix+".PCA_plot.png")
        plt.figure(figsize=(20, 15))
        plt.tight_layout()
        plt.scatter(pca_x_train[:, 0], pca_x_train[:, 1], cmap = 'viridis', s = 1)
        plt.title('PCA, plot')
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.savefig(dir_pca,dpi=150)
        #plt.show()
        
        ## Min Max scaling
        pca_x_train = MinMaxScaler().fit_transform(pca_x_train)
        X_train, X_test = train_test_split(pca_x_train, test_size=0.33, random_state=42)
        print("\n")
        print("Dimenison of reduced data="+str(pca_x_train.shape))
        return X_train, X_test 
    
    def dimension_TSNE(self,X,outprefix):
        
        tsne =TSNE(n_components=2).fit_transform(X)
        print(tsne.shape)
        tsne_df = pd.DataFrame(data = tsne)
        #make directory and save files
        try:
            if not os.path.exists("TSNE_results"):
                os.makedirs("TSNE_results")
        except OSError:
            print('Error : creating directory'+"TSNE_results")
        path_wo_ext = self.dataFile.rsplit('.',1)[0]
        path_prefix = os.path.basename(path_wo_ext)
        dir_tsne = os.path.join("TSNE_results",path_prefix+"."+outprefix+".components.csv")
        #if os.path.exists(dir_tsne):
        #    dir_tsne = os.path.join("TSNE_results",outprefix+"."+outprefix+".components.csv")
        #else:
        #    dir_tsne = os.path.join("TSNE_results",path_prefix+"."+outprefix+".components.csv")
        tsne_out_file = open(dir_tsne,"w")
        tsne_df.to_csv(tsne_out_file)
        
        
        
        dir_tsne = os.path.join("TSNE_results",path_prefix+"."+outprefix+".TSNE_plot.png")
        #if os.path.exists(dir_tsne):
        #    dir_tsne = os.path.join("TSNE_results",outprefix+"."+outprefix+".TSNE_plot.png")
        #else:
        #    dir_tsne = os.path.join("TSNE_results",path_prefix+"."+outprefix+".TSNE_plot.png")
        fig, ax = plt.subplots(1)
        plt.figure(figsize=(20, 15))
        plt.tight_layout()
        sns.scatterplot(tsne[:, 0], tsne[:, 1], data = tsne_df, palette=sns.color_palette("hls", 3)).set(title = "TSNE plot")
        plt.title('TSNE, plot')
        plt.xlabel("tsne1")
        plt.ylabel("tsne2")
        plt.savefig(dir_tsne,dpi=150)
        #plt.show()
        
    def __repr__(self):
        return "Layers in sequential NeuralNetwork(including input layer):{}".format("-".join(str(i) for i in self.layers))
        #f"SequentialModel n_layer: {len(self.layers)}"
        
    
  
    def wt_initialize(self,r1,r2):
        #initialize the weights connections between layers in autoencoders
        
        wt  = np.random.uniform(-1.0,1.0,(r1, r2))
        return wt 
    
    def fit(self,X,y,prefixout):
        # Add a column of bias term to X matrix
        X = np.c_[X, np.ones((X.shape[0]))]
        #y = np.c_[y, np.ones((y.shape[0]))]
        loss_plot = []
        for ep in range(0,self.epochs):
            idx = np.random.randint(0, X.shape[0], self.batch_size)
            X=X[idx]
            y=np.delete(X,np.s_[-1:],axis=1)
            
            
            for (X_train,y_train) in zip(X,y):
                
                
                
                tr_au = self.train_autoencoders(X_train,y_train,self.activation_name)
                mean_sq_loss = sum(tr_au)/len(tr_au)
            loss_plot.append(mean_sq_loss)
        try:
            if not os.path.exists("Autoencoder_results"):
                os.makedirs("Autoencoder_results")
        except OSError:
            print('Error : creating directory'+"Autoencoder_results")
        path_wo_ext = self.dataFile.rsplit('.',1)[0]
        #prefix = os.path.basename(path_wo_ext)
        dir_autoen = os.path.join("Autoencoder_results",prefixout+"."+".Autoencoder.png")
        df = pd.DataFrame({'Training':loss_plot})
        ax = df.plot(kind = 'line', title = 'Loss curve')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        fig = ax.get_figure()
        fig.savefig(dir_autoen,dpi=150) 
        dir_autoen = os.path.join("Autoencoder_results",prefixout+"."+".Autoencoder.txt")
        auto_out_file = open(dir_autoen,"w")
        df.to_csv(auto_out_file)
                
    def train_autoencoders(self,X,y,name):
        
        train_data = y
        A=[np.atleast_2d(X)]
        for ly in np.arange(0, len(self.W)):
            net = self.feedforward(A[ly],self.W[ly])
            
            net_activation = self.Relu(net)           
            
            A.append(net_activation)    
        
       ## if the layer is the output layer
        out_error = A[-1] - train_data
        
        loss_ep = 0.5 * np.sum((A[-1] - train_data) ** 2)
        #loss_ep =  np.sum((A[-1] - train_data) ** 2)
        self.loss.append(loss_ep)
        accumulate_delta = [out_error * self.Sigmoid_gradient(A[-1])]
        
        ## if the layer is hidden
        for lay in range(len(A) -2,0,-1):
            delta_wt = accumulate_delta[-1].dot(self.W[lay].T)
            delta_gard = delta_wt * self.Sigmoid_gradient(A[lay])
            
            accumulate_delta.append(delta_gard)
            
        ### Updates weight
        accumulate_delta = accumulate_delta[::-1] 
        for ly in np.arange(0, len(self.W)):
            self.W[ly] += -self.lr * A[ly].T.dot(accumulate_delta[ly])
            #W_new.append(self.W[ly])
        return self.loss
    
    def feedforward(self,X,W):
        return X.dot(W)
    
    
    def predict_autoencoders(self,layers,X,prefix_out):
        test_data=np.atleast_2d(X)
        new_prefix = prefix_out+"TSNE_ON_AUTOENCODER"
        test_data = np.c_[test_data, np.ones((test_data.shape[0]))]
        for ly in np.arange(0, len(self.W)):
            test_data = self.Sigmoid(np.dot(test_data, self.W[ly]))
            
        
        ##Plot dimensionality reduction
        self.dimension_TSNE(test_data,new_prefix)
        #plt.scatter(test_data[:,0],test_data[:,1], s=1, cmap='tab20')
        df = pd.DataFrame(test_data)
        #plt.show()

    
    #### all activation functions

    def Sigmoid(self,X):
        return 1 / (1 + np.exp(-X))
    
    def Sigmoid_gradient(self,X):
        return self.Sigmoid(X) * (1 - self.Sigmoid(X))
    
    def Relu(self,X):
        return np.where(X >= 0, X, 0)
    
    def Relu_gradient(self,X):
        return np.where(X > 0, 1, 0)
    
    def Elu(self,X):
        return np.where(X >= 0, X, self.lr*(np.exp(X) - 1))

    def Elu_gradient(self,X):
        return np.where(X >= 0, 1, self.Elu(self.lr) + self.lr)

if __name__ == '__main__':
    
    ##Download the file from below URL.
    #url = "https://github.com/kumara3/Computational_genomics/blob/master/pbmc.counts.zip"
    url = "path_to_filename/pbmc.counts.txt"
    
    layers = [[10,5,2,5,10],[15,10,5,2,5,10,15],[20,15,10,8,2,8,10,15,20]]
    learn_rate = [0.1,0.01]
    for each in layers:
        for lr in learn_rate:
            prefix_name = "layer_"+str(len(each))+"_"+str(lr)+"_"+str(each[0])
            
            encoder = Autoencoder(each,'Sigmoid',lr,500,url,each[0])
            
            #encoder = Autoencoder([20,10,10,2,10,10,20],'Sigmoid',0.01,500,url,20)
            #encoder = Autoencoder([20,5,2,5,20],'Sigmoid',0.01,500,url,20)
            #encoder = Autoencoder([10,5,1,5,10],'Sigmoid',0.01,500,url,10)
            #encoder = Autoencoder([10,5,1,5,10],'Elu',0.01,500,url,10)
            #encoder = Autoencoder([10,5,2,5,10],'Sigmoid',0.1,500,url,10)
    
            ##Autoencoder execution
            encoder.preprocessing()
            X,y = encoder.dimension_reduction_PCA()
            encoder.dimension_TSNE(X,"TSNE_on_PCA")
            encoder.fit(X,X,prefix_name)
    
            ## Bottlenck layer feature extraction
            bottle_neck_layer = [i for i in each if i > min(each)]
            bottle_neck_pos = len(bottle_neck_layer)/2
            bottle_neck_pos = int(bottle_neck_pos)
            bottle_neck_layer = bottle_neck_layer[0:bottle_neck_pos]
            #encoder.predict_autoencoders([10,5,2],X)
            print(bottle_neck_layer)
            
            encoder.predict_autoencoders(bottle_neck_layer,y,prefix_name)
            
            