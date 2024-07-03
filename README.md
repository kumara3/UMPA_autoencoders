**Dimensionality reduction in Single cell sequencing using Autoencoder**
Single cell sequencing technology refers to the sequencing of genome or transcriptome at the single cell resolution, so as to obtain the cell population difference. Single-cell technologies have the advantage of detecting heterogeneity among individual cells compared to other traditional sequencing methods where we get the average of overall cells and less to no resolution of the cellular heterogeneity. Although this technology has emerged as one of the most sought method for biologist, the bioinformatics of single cell data presents a challenge. This is due to high dimensional nature of data and dropout events during the sequencing.
Single cell RNA sequencing technology enables the capture of expression of genes of the individual cells thus giving better resolution of cellular heterogeneity. Traditional RNA sequencing methods gives an average of gene expression in many cells but loose the cellular heterogeneity [1]. An important steps in single cell sequencing analysis is the clustering of cells into groups (known and novel cell type) on the basis of expression levels of signature genes. However, clustering step has been limited by dimensionality of the single cell data. This is because clustering methods in high dimensional data could lead to misleading results as the distance between most of the pair of genes is similar. Other properties of single cell data such as sparsity, dropout, and batch effect adds further complexity. Therefore, finding the accurate low dimensional representation of the data becomes more crucial than downstream analysis [2] Earlier works have applied various dimension reduction methods to the single cell dataset. PCA is used for an initial dimension reduction on the basis of highly variable genes. Another method called t-SNE (t-distributed stochastic neighbor embedding) is a nonlinear dimension reduction technique that preserves the local structure in the data. The points which are closer to one another in the high dimension data set will tend to be close to one another in the low dimension. The t-SNE [3] algorithm works by modeling the probability distribution of the neighbors around each point. In the high dimensional space this is modeled as Gaussian distribution, whereas in low dimensional space probability distribution are modeled as t-distribution. The algorithms works by finding a mapping onto 2-dimension space that minimizes the difference between two distributions between all points. However t-SNE suffers from limitation such as loss of information on inter cluster relationship, slow computation time. Another method which has recently been widely used in the single cell sequencing is UMAP [4] (uniform Manifold Approximation and Projection). UMPA offers advantages over t-SNE such as increase in speed and better preservation of the global structure. The above mentioned dimensionality reduction approach achieves good performance. However, robust approaches are needed to account and adjust for nature of single cell sequencing dataset. In this project I propose one such approach; the autoencoders.The above mentioned dimensionality reduction approach achieves good performance. However, robust approaches are needed to account and adjust for nature of single cell sequencing dataset. In this project I propose one such approach; the autoencoders.

 *Detailed description at* : <object data="files/14732870/ProjectReport_Autoencoders_AshwaniKumar_AXK200017_Report.pdf" width="1000" height="1000" type='application/pdf'></object>
