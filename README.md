# GSP_FKT_MI_decoding

 This script is an implementation of the method proposed in:
 
 > Miri, M., et al., "Spectral representation of EEG data using learned graphs with application to motor imagery decoding." Biomedical Signal Processing and Control 87 (2024): 105537.   (https://www.sciencedirect.com/science/article/pii/S1746809423009709)


![A schematic overview of the proposed methodology for spectral representation of EEG data on the harmonic basis of learned graphs](figs/overview.jpg?raw=true)
A schematic overview of the proposed methodology for spectral representation of EEG data on the harmonic basis of learned graphs

The main function is `demo.m`. Further description of which is given in the following. 

The general idea is to transform EEG data into a spectral representation. The training and test EEG signal sets for each subject are initially preprocessed, and then fed into the training and test phases, respectively. As temporal preprocessing, for each trial, we used the time points within the 0.5–2.5 s interval after the visual cue to construct graph signals. Motor activity, be it real or imagined, modulates the sensorimotor mu rhythm (8–13 Hz) and beta rhythm (13–30 Hz), therefore, we filtered the extracted signal with a third order Butterworth filter with a passband of 8–30 Hz.
The proposed method is comprised of four stages. First, we modeled the structure of the brain of each subject as a graph, in which vertices corresponded to the EEG electrodes and edges were defined by estimating the graph’s weighted adjacency matrix using the log-penalized and 𝓁2-penalized graph learning frameworks. As a means of comparison, we also defined a fully connected correlation graph in which edge weights were defined based on the absolute value of the Pearson correlation coefficient between the time courses. Second, by interpreting EEG maps as time-dependent graph signals on the graphs, we transform the data into a spectral representation. For each graph, the eigenvectors of the Laplacian matrix were used to compute the GFT of each graph signal. Third, using FKT, a transformation matrix 𝐖 that maps the GFT coefficients into a discriminative graph spectral subspace was then derived. Fourth, we treat the logarithm of variance of representations within the subspace as features, which is in turn used to train and test a SVM classifier.



- run 'demo.m' to classify MI-BCI data contained in folder 'data' 

- fkt is the function that computes the projection matrix by using Fukunaga-Koontz transform

- demean_norm is the function that computes the de-meaned and normalized graph signals 


-------------------------------------------------------------------------------------------------------------------------------------------------------
 Maliheh Miri, December 2024.

