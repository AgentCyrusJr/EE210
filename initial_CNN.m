function [ CNN ] = initial_CNN( input_size, NN_layer, corr_depth,  num_subvector, num_pools)
%%%%%%%%%%%%%%%%%
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

% Creates an empty struct which we will fill next
CNN = struct;

% The input size of each image/audio .. etc. Note: We may not need this as the code can learn it by reading the size of the input vector directly.
CNN.input_size = input_size;

% Depth of the Correlation layers, note CNN.C = length(CNN.D)
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

for i = 1 : CNN.C+1
    for j = 1 : CNN.D(i)+1
        CNN.W{j, i}  = ones(CNN.input_size/CNN.P(i), CNN.D(i));
        CNN.theta{j, i}  = ones(CNN.D(i),1);
        
%         CNN.W{j+1, i+1}  = rand(CNN.input_size/P(i), CNN.D(i));
%         CNN.theta{j+1, i+1}  = rand(P(i));
    end
    if (i ==1) 
        CNN.partion_matrix{i} =  eye(CNN.input_size);
    
    else CNN.partion_matrix{i} =eye(CNN.P(i-1)/CNN.Pp(i)/CNN.Pp(i-1));
    end
    CNN.permutation_matrix{i} = eye(CNN.P(i));
end

% The partition function helps trim the feature vector into a Hn(d,c) matrix
% %%%%%%%%%%%%%%%%%%%% parameter:, return:
% CNN.partition = @(feature_vector, c, cnn)partition_function;


end

