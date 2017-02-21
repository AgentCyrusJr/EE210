function [ partition ] = partition_function( feature_vector, c, CNN)
%%%%%%%%%%%%%%%%%%%%
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
% partition = 1;

% The partition function helps trim the feature vector into a Hn(d,c) matrix
% %%%%%%%%%%%%%%%%%%%% parameter:, return:

partition = reshape(CNN.partion_matrix{c}*feature_vector,CNN.input_size/CNN.P(c), CNN.P(c));


end

