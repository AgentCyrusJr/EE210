function [ partition ] = partition_fun( h, c, CNN)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Partition function: helps trim the feature vector into a matrix Hn(d,c)
% Parameters: feature vector---h, index of convolution layer---c, CNN structure---CNN
% Return: feature matrix---Hn(d,c)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if (c ==1) 
        partition_matrix = eye(CNN.input_size);
    else
        partition_matrix = eye(CNN.Pp(c-1));
    end


partition = reshape(partition_matrix*h,length(h)/CNN.P(c), CNN.P(c));


end

