%   convert the input image into output image associated with the weight
function [ res ] = outputImg( input, weight, k)
% parameter:
% input : input image, weight: mask, k: half length of the input image,(2*k+1 = size)

% return:
% output image
res = zeros(2*k+1);
for r = 1:2*k+1
    for c = 1:2*k+1
        for i = -k:k
            for j = -k:k
                if (r+i > 0 && c+j > 0 && r+i <= 2*k+1 && c+j<= 2*k+1)
                    res(r,c) = res(r,c) + input(r+i, c+j)*weight(k+1+i, k+1+j);
                end
            end
        end
    end
end
end

