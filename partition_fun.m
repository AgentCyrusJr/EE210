function [ partition ] = partition_fun( h, c, CNN)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Partition function: helps trim the feature vector into a matrix Hn(d,c)
% Parameters: feature vector---h, index of convolution layer---c, CNN structure---CNN
% Return: feature matrix---Hn(d,c)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

partition = reshape(CNN.partition_matrix{c}*h,length(h)/CNN.P(c), CNN.P(c));


end

