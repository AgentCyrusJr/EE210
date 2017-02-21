
%%% This is just an example construction of how the CNN structure will look like


CNN = struct;   
% Creates an empty struct which we will fill next

CNN.input_size = 1024;
% The input size of each image/audio .. etc. Note: We may not need this as the code can learn it by reading the size of the input vector directly.

CNN.C = 6;   
% No. of Correlation layers, e.g. 6

CNN.L = 3;   
% No. of layers in fully connected NN, e.g. 3

CNN.D = [2 3 8 9 3 2];   
% Depth of the Correlation layers, note CNN.C = length(CNN.D)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Weights and Biases %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CNN.W_corr = cell(CNN.C, max(CNN.D));
% This is cell that contains the weights of each correlation layer. 
% The structure of the weights follows the structure of (47.1224a) and (47.1223a) in the notes.

CNN.Bias_corr = {S_1 , S_2, S_3, S_4, S_5, S_6};
% This is cell that contains the biases of each correlation layer. 
% The structure of the weights follows the structure of (47.1224b) and (47.1223b) in the notes.

CNN.W_nn = {W_1 , W_2, W_3};   
% This is cell that contains the weights of each neural network layer. 
%Structure similar to (47.1221g)

CNN.Bias_nn = {S_1 , S_2, S_3};   
% This is cell that contains the biases of each neural network layer. 
% Each element is a vector specifying the bias for each neural network in layer (l).  
%Structure similar to (47.1221h)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Function handlers in the CNN structure %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CNN.partition = @(h,c)parition_fun;
% The partition function takes two inputs, the feature vector h and the index of the correlation layer c.
% Depending on c, the partition_fun should implement the partitioning corresponding to layer c.
% Example how to call the function. TO do the permutation at the beginning of layer 2, we simply call CNN.partition(h,2)   where h is the featur input we want to partition.

CNN.activation_corr = @(z,c)activation_corr_fun;
% The activation function, takes the vector z and the index of the correlation layer c.
% Depending on c, we can apply a different activation function or keep the same by ignoring the value c.

CNN.activation_nn = @(z,l)activation_nn_fun;
% The activation function, takes the vector z and the index of the neural network layer l.
% Depending on l, we can apply a different activation function or keep the same by ignoring the value l.

CNN.pooling = @(y,c)pooling_fun;
% The pooling function, takes the vector y and the index of the correlation layer c.
% Depending on c, a different pooling can be applied if needed.

CNN.inv_permutation = @(y,c)inv_permutation_fun;
% The inverse permutation function used when calculating the derivatives. 
% Since this depends on the pooling, then it is part of the structure.

