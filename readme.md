# BRIDGE
### This repository houses data for the BRIDGE paper under review in the the ACM Conference on Computer and Communications Security 2023. 

More ease of review, our code are in either in jupiter notebooks or python code. 


We organized as follows:

## SCADA Folder: 
Here you will find the python code used to perform the statistical (i.e., frequency and timing) analysis of SCADA physical-bound API calls (i.e., the WRITE and READ calls). There is a README in this folder that will describe how to use the provided code implementation to ingest the SCADA API traces to output the derivations outlined in the paper.  Due to the size of the input-data/All-API-Calls.CSV (826.97MB), which cannot be uploaded in git, we included a reduced size. Also, the output-data-traces/ScadaBR-Dependencies was split into 3 files, ScadaBR-Dependencies-1, ScadaBR-Dependencies-2, and ScadaBR-Dependencies-3, to fit into github. 

## PHYSICS Folder: 
Here, you will find our implementation of the Transformer-based Autoencoder. This is in a jupyter notebook format, which can be easily opened in Google colab online, using the chemical dosing level control process data, which can be found exclusively in the directory. 

Detailed Implementation Information (using the chemical dosing level control as example)

We trained our model on the Level Controls dataset which had 157579 records, all benign (no attacks). In order to train the model, first we need to convert the data into sequences so that, a single input of the data, has x number of datapoints, where x is the sequence size. We found that the model would perform the best in term of metrics when the sequence size was chosen as 2, i.e. the model would look at two previous datapoints, to make prediction about the state of the system, in the next timestamp.

Window Size and Reconstruction Losses:
After converting the data into sequences, we standardized the data, such that the mean of each feature would be 0 and variance would be 1.

Like most other variational autoencoders, ours also had a encoder first, then a bottleneck layer which would learn the latent representations of the data, and finally a decoder, which would try to learn to reconstruct the input from the latent representations. 

We structured the encoder such that, first there were 8 transformer blocks and each transformer block would have 8 heads, followed by a dropout layer and then 2 fully connected layers with another dropout layer between them.  After the transformer block, we have 2 fully connected layers with ReLU (https://doi.org/10.48550/arxiv.1803.08375) activation, the first with 5 neurons and the second with 3 neurons (the number of neurons were decided after analyzing the number of features in the data which was 8), decreasing in size like as in most other autoencoders. 

For the bottleneck layer, we added two fully connected layers without any activations, and then sampled the mean and the variance from them. 

The sampled data was then passed to the decoder, which again consisted of a transformer block, with the same specifications as before. Then, we had two fully connected layers with ReLU activation function, the first with 5 neurons, and the second with 7 neurons. Finally, we had a final fully connected layer, with no activation which would serve as our output layer.

We train the model using the loss function which is used in most VAEs, the sum of reconstruction loss, which penalizes higher mean squared error between the input data and the reconstruction by the model and the KL divergence (https://doi.org/10.1214/aoms/1177729694) loss which penalizes difference in probability distributions of the input data and the reconstruction. The optimizer for our model was ADAM (https://doi.org/10.48550/arxiv.1412.6980). Training our final model took about 20 minutes on a 4 Core CPU with 15 GB RAM and a single Nvidia T4 GPU with 16 GB VRAM. 

In order to convert the reconstructions to binary predictions, we took a threshold of 0.75 quantile of the training loss, and if the reconstruction loss was greater than this threshold, we'd consider the data to be indicating an attack scenario. We performed several experiments with different thresholds and found that a threshold of 0.75 quantile was ideal.

## PLC Devices Analysis: 
Here you will find python scripts used to statically analyze the Actuator connections based on the Statement List (STL) and Function Block Diagrams (FBD)


## FACTORYIO: 
Here you will find information on our testbed setup based on the FactoryIO ICS physical process emulation environment. Ofcourse, FactoryIO can be downloaded and installed for free. 




