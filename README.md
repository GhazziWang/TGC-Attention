# TGC-Attention
Abstract: Efficient prediction of bike-sharing usage is pivotal for enhancing user satisfaction and optimizing operational costs for service providers. In this paper, we propose a novel model, the Temporal Graph Convolutional Attention Network (TGC-Attention), which leverages Temporal Convolution Network (TCN) to capture node correlations and updates the adjacency matrix of the Graph Convolutional Network (GCN). The model integrates attention prediction with channel embedding based on shapelet-kmeans and incorporates a residual block based on series decomposition. Through real-world data comparison among different time series prediction models, our study yields intriguing insights for further exploration in this domain.
### Overview
![image](https://github.com/GhazziWang/TGC-Attention/assets/49545379/1cabac85-a07e-4e7d-be8b-aa867ad0be71)
### Data Discription
The Divvy dataset, which includes both electric and classic bicycles from Chicago's bike-sharing system, was chosen for experimentation. Spanning from January 1, 2022, to February 28, 2024, it comprises 5,505,718 classic bicycle trips. To provide a comprehensive understanding of the dataset, cumulative probability distributions of distance and duration were computed and visualized in Figures 3 and 4. Leveraging these insights, the dataset underwent spatio-temporal pruning, excluding trips with durations less than 120 minutes and distances less than 10 kilometers. Consequently, the final dataset contained 5,422,713 trips. This system encompasses 756 stations, each of which the model will abstract into a graph.

### Training
For training, 80% of the dataset is randomly shuffled and used as the training set to train the model, while the remaining 20% is reserved for testing the prediction performance. The Adam optimizer is used during training with the RMSE loss function. Figure 5 illustrates the average loss curves for each epoch across different input and output configurations. 
![Uploading image.png…]()
