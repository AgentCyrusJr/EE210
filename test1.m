%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the test file for a specific case 
% We can treat this .m file as the main function, here we initialize CNN
% struture with values, and run feedforward given the raw input and CNN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
input_size = 1024;

% # of neuron nodes
N_neuron = [3,2];
% # of depth in each layer
D = [2,2,2];
% # of subvectors in each layer
% e.g. In our case, for a feature vector with input size of 1024, we cut
% the feature vector into 512 subvectors, with volumn of each subvector
% equals 2.
P = [512,128,32];
% # of pools in each layer
% e.g. In our case, Pp denotes P'. Firstly, we will have 256 pools, with
% output t dimension of 256*1.
Pp = [256,64,8];

% Initialization of the structure of CNN
CNN = initial_CNN( input_size, D,  P, Pp, N_neuron);
% input a random matrix
H_raw = [[1:input_size]',[input_size+1:2*input_size]'];
% feedforward, main function
[Z,Y,y1]= feedforward( H_raw, CNN );