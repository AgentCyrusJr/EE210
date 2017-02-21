function [ CNN ] = initial_CNN( input_size, NN_layer, corr_depth,  num_subvector, num_pools)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CNN: Initialize the structure of CNN
% Parameters: size of the raw input vector---input_size, # of neural
% network---NN_layer, depth vector of CNN---corr_depth, # of subvectors in
% each layer---num_subvector, % # of pools in each layer---num_pools
% Return: CNN structure---CNN
% Details: /
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Creates an empty struct which we will fill next
CNN = struct;

% The input size of each image/audio .. etc. Note: We may not need this as the code can learn it by reading the size of the input vector directly.
CNN.input_size = input_size;

% Depth of the Correlation layers, note the first element in D is D0
CNN.D = corr_depth;

% No. of Correlation layers, e.g. 6
CNN.C = length(CNN.D)-1;

% No. of layers in fully connected NN, e.g. 3
CNN.L = NN_layer;

% # of subvectors
CNN.P = num_subvector;

% # of groups
CNN.Pp = num_pools;

CNN.W = cell(max(CNN.D)+1, CNN.C+1);
CNN.theta = cell(max(CNN.D)+1, CNN.C+1);
CNN.partion_matrix = cell(CNN.C+1, 1);
CNN.permutation_matrix = cell(CNN.C+1, 1);
CNN.Wnn = cell(CNN.L-1,1);
CNN.thetann = cell(CNN.L-1,1);

for i = 1:CNN.L-1
    if (i == 1)
        CNN.Wnn{i} = ones(3,CNN.Pp(CNN.C)*CNN.D(CNN.C+1));
    else 
        CNN.Wnn{i} = ones(3);
    end
    CNN.thetann{i} = ones(3,1);
end

for i = 1 : CNN.C+1
    for j = 1 : CNN.D(i)+1
%%initialize the weights and bias
    if(i==1)
        CNN.W{j, i}  = ones(CNN.input_size/CNN.P(i), CNN.D(i));
    else
        CNN.W{j, i}  = ones(CNN.Pp(i-1)/CNN.P(i),CNN.D(i));
    end
        CNN.theta{j, i}  = ones(CNN.D(i),1);
        
%         CNN.W{j+1, i+1}  = rand(CNN.input_size/P(i), CNN.D(i));
%         CNN.theta{j+1, i+1}  = rand(P(i));
    end
%%initalize the partition matrix and permutation matrix
    if (i ==1) 
        CNN.partion_matrix{i} = eye(CNN.input_size);
    
    else CNN.partion_matrix{i} = eye(CNN.Pp(i-1));
    end
    CNN.permutation_matrix{i} = eye(CNN.P(i));
end

% The partition function helps trim the feature vector into a Hn(d,c) matrix
% %%%%%%%%%%%%%%%%%%%% parameter:, return:
% CNN.partition = @(feature_vector, c, cnn)partition_function;


CNN.partition = @(h,c,CNN)partition_fun(h,c,CNN);
% The partition function takes two inputs, the feature vector h and the index of the correlation layer c.
% Depending on c, the partition_fun should implement the partitioning corresponding to layer c.
% Example how to call the function. TO do the permutation at the beginning of layer 2, we simply call CNN.partition(h,2)   where h is the featur input we want to partition.

CNN.activation_corr = @(z,c)activation_corr_fun(z,c);
% The activation function, takes the vector z and the index of the correlation layer c.
% Depending on c, we can apply a different activation function or keep the same by ignoring the value c.

CNN.activation_nn = @(z,l)activation_nn_fun(z,l);
% The activation function, takes the vector z and the index of the neural network layer l.
% Depending on l, we can apply a different activation function or keep the same by ignoring the value l.

CNN.pooling = @(y,c,CNN)pooling_fun(y,c,CNN);
% The pooling function, takes the vector y and the index of the correlation layer c.
% Depending on c, a different pooling can be applied if needed.
end

