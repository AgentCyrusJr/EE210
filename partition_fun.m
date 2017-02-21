function [ partition ] = partition_fun( feature_vector, c, CNN)
%%%%%%%%%%%%%%%%%%%%
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
% partition = 1;

% The partition function helps trim the feature vector into a Hn(d,c) matrix
% %%%%%%%%%%%%%%%%%%%% parameter:, return:

partition = reshape(CNN.partion_matrix{c}*feature_vector,length(feature_vector)/CNN.P(c), CNN.P(c));


end

