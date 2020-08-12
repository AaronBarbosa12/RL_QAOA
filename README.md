# RL_QAOA
Optimizing the performance of the Quantum Approximate Optimizaton Algorithm (QAOA) using Policy Gradients. 

The QAOA (https://arxiv.org/abs/1411.4028) is a variational quantum algorithm for approximating the ground state of some Hamiltonian, H. The quality of the approximation obtained from the QAOA depends on several input parameters, &gamma; and &beta; . In this project, I used reinforcement learning in order to find optimial values of &gamma; and &beta; much faster than what was obtained by using classical optimization techniques alone. 

<p align="center">
  <img width="300" height="300" src="images/RL_Model.PNG">
</p>

* The model tries to maximize the average performance of the QAOA on the MaxCut problem across a collection of 3-Regular, 4-Regular, and Erdos-Renyi graphs of varying densities.
* The model is trained to find the optimal distribution from which to select &gamma; and &beta;, as was proposed in https://arxiv.org/pdf/2002.01068.pdf. 
* The model also uses graph convolutions and Spatial Pyramidal Pooling (https://arxiv.org/pdf/2002.01068.pdf) as additional input features to improve the quality of the model predictions.

<p align="center">
  <img width="300" height="300" src="images/Results.PNG">
</p>

# What I've Learned
* How the QAOA works from a physics and a computational perspective
* How reinforcement learning works from a computational and mathematical perspective 
* How to implement Policy Gradients in a continuous action space
* How to implement convolutional neural networks in tensorflow
* How to manage .csv files in python
* How to manage large datasets with pandas
