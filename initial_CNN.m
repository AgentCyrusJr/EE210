function [ CNN ] = initial_CNN( input_size, D,  P, Pp, N_neuron)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CNN: Initialize the structure of CNN
% Parameters: size of the raw input vector---input_size, 
% depth vector of CNN---D, # of subvectors in
% each layer---P, % # of pools in each layer---Pp, # of neuron nodes---N_neuron
% Return: CNN structure---CNN
% Details: /
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Creates an empty struct which we will fill next
CNN = struct;

% The input size of each image/audio .. etc. Note: We may not need this as the code can learn it by reading the size of the input vector directly.
CNN.input_size = input_size;

% Depth of the Correlation layers, note the first element in D is D0
CNN.D = D;

% # of Correlation layers
% e.g. In our case, with depth vector = [2,2,2], we assume that we have 2
% convolutional network. which is length(CNN.D)-1
CNN.C = length(CNN.D)-1;

% No. of layers in fully connected NN, e.g. 3
CNN.L = length(N_neuron)+1;

% # of subvectors
CNN.P = P;

% # of groups
CNN.Pp = Pp;

% @Yahya : # size of groups
CNN.pooling_size = P./Pp;

% # of neuron nodes, it is a vector
CNN.N_neuron = N_neuron;

% e.g. for CNN.W and CNN. theta, we create a cell for each to store the
% W or theta we need to compute the output z.
% Also, since the partition matrices and permutation matrices for each layer are
% the same, the dimension of the cell is CNN.C*1
% @Yahya : Since D contains D0, then shouldn't the W cell be of szize CNN.D-1 x CNN.C ?
%Response:max(CNN.D) is the largest element in D, containintg D0 will
%influence the length(CNN.D) but not the max value.
CNN.W = cell(max(CNN.D), CNN.C);
CNN.theta = cell(max(CNN.D), CNN.C);
CNN.partition_matrix = cell(CNN.C, 1);
CNN.permutation_matrix = cell(CNN.C, 1);
% Here, nn denotes neural network
CNN.Wnn = cell(CNN.L-1,1);
CNN.thetann = cell(CNN.L-1,1);

for c = 1 : CNN.C
    for d = 1 : CNN.D(c+1)
% initialize the weights and bias
    if(c == 1)
        CNN.W{d, c}  = 0.5*rand(CNN.input_size/CNN.P(c), CNN.D(c));
    else
        CNN.W{d, c}  = 0.5*rand(CNN.Pp(c-1)/CNN.P(c),CNN.D(c));
    end
        CNN.theta{d, c}  = 0.5*rand(CNN.D(c),1);
    end
end
% initalize the partition matrix and permutation matrix
% partition matrices are not necessarily be square, however, the
% permutation matrices must be square.


for l = 1:CNN.L-1
    if (l == 1)
        CNN.Wnn{l} = rand(CNN.N_neuron(l),CNN.Pp(CNN.C)*CNN.D(CNN.C+1));
    else 
        CNN.Wnn{l} = rand(CNN.N_neuron(l), CNN.N_neuron(l-1));
    end
    CNN.thetann{l} = rand(CNN.N_neuron(l),1);
end

CNN.partition = @(h, c, CNN_)partition_fun(h, c, CNN_);
% The partition function takes two inputs, the feature vector h and the index of the correlation layer c.
% Depending on c, the partition_fun should implement the partitioning corresponding to layer c.
% Example how to call the function. TO do the permutation at the beginning of layer 2, we simply call CNN.partition(h,2)   where h is the featur input we want to partition.

CNN.activation_corr = @(z, c)activation_corr_fun(z, c);
% The activation function, takes the vector z and the index of the correlation layer c.
% Depending on c, we can apply a different activation function or keep the same by ignoring the value c.

CNN.activation_nn = @(z, l)activation_nn_fun(z, l);
% The activation function, takes the vector z and the index of the neural network layer l.
% Depending on l, we can apply a different activation function or keep the same by ignoring the value l.

CNN.pooling = @(y, c, CNN_, pooling_method)pooling_fun(y, c, CNN_, pooling_method);
% The pooling function, takes the vector y and the index of the correlation layer c.
% Depending on c, a different pooling can be applied if needed.


CNN.activation_nn_deriv = @(z,l)activation_nn_deriv_fun(z,l);

CNN.activation_corr_deriv = @(z,c)activation_corr_deriv_fun(z,c);


CNN.pooling_deriv = @(CNN_,y,c,pooling_method)pooling_deriv_fun(CNN_,y,c,pooling_method);

end

