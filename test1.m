%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the test file for a specific case 
% We can treat this .m file as the main function, here we initialize CNN
% struture with values, and run feedforward given the raw input and CNN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc

input_size = 1024;

% # of neuron nodes
N_neuron = [3,2];
% # of depth in each layer
D = [2,4,2];
% # of subvectors in each layer
% e.g. In our case, for a feature vector with input size of 1024, we cut
% the feature vector into 512 subvectors, with volumn of each subvector
% equals 2.
% P = [512,128,32];
P = [512,128];

% # of pools in each layer
% e.g. In our case, Pp denotes P'. Firstly, we will have 256 pools, with
% output t dimension of 256*1.
% Pp = [256,64,8];
Pp = [256,64];

gamma=1;
% Initialization of the structure of CNN
CNN = initial_CNN( input_size, D,  P, Pp, N_neuron);
% input a random matrix
H_raw = rand(input_size,D(1));
% feedforward, main function

[y1,Z_c,Y_c,Z_nn,Y_nn,H]= feedforward( H_raw, CNN );

[d_nn,Delta_c,Y_nn,H] = CalculateDerivatives(CNN,H_raw,gamma);

[Wnn,W_fin,theta_fin,thetann] = grad_desc(CNN,H_raw,gamma);
