%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the test file for a specific case 
% We can treat this .m file as the main function, here we initialize CNN
% struture with values, and run feedforward given the raw input and CNN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
input_size = 1024;

% # of convolutional layers + 1
L = 3;
% # of depth in each layer
D = [2,2,2];
% # of subvectors in each layer
P = [512,128,32];
% # of pools in each layer
Pp = [256,64,8];

% Initialization of the structure of CNN
CNN = initial_CNN(input_size, L,  D, P, Pp);
% input a random matrix
H_raw = [[1:input_size]',[input_size+1:2*input_size]'];
% feedforward, main function
y1= feedforward( H_raw, CNN );