function [ partition,partition_matrix ] = partition_fun( h, c, CNN,mode)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Partition function: helps trim the feature vector into a matrix Hn(d,c)
% Parameters: feature vector---h, index of convolution layer---c, CNN structure---CNN
% Return: feature matrix---Hn(d,c)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
partition = [];
if strcmp(mode,'init')
    image_size = sqrt(length(h))-2*CNN.padding_margin(c);
    partition_matrix = partition_matrix_image(image_size, CNN.stride(c), CNN.padding_margin(c), CNN.filtersize(c));%eye(CNN.input_size);
else
    
    partition = reshape(CNN.partition_matrix{c}*h,CNN.filtersize(c)^2, CNN.P(c));
end
   
end
function [partition_matrix] = partition_matrix_image(image_size, stride, padding_margin, filter_size)
%min_size : the length or width of imput image
%stride : the stride
%padding margin: the padding margin 
%filtersize: size of filter, e.g. 3, then the filter is 3*3
K= (filter_size-1)/2;
sparse_index = zeros((floor((image_size-1)/stride)+1)^2*(2*K+1)^2,2);
count =1;
for c = padding_margin+1 : stride : image_size+padding_margin
    for r = padding_margin+1 :stride: image_size + padding_margin
        for l = 1 : 2*K+1
            for k = 1 : 2*K+1
                %%% Yahya: I checked this, the computation is perfect.
                sparse_index(count,:) = [count, (c-padding_margin+l-1-1)*(image_size+2*padding_margin)+r+k-1-padding_margin];
                count =count+1;
            end
        end
    end
end
partition_matrix = sparse(sparse_index(:,1),sparse_index(:,2),ones((floor((image_size-1)/stride)+1)^2*(2*K+1)^2,1),(floor((image_size-1)/stride)+1)^2*(2*K+1)^2,(image_size+2*padding_margin)^2);
end