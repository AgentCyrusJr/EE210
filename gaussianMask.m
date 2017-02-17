% Implement a gaussian mask
function [ weight ] = gaussianMask(k, l, sigma)
% parameter:
% k : row, l: column, sigma: standard deviation

% return:
% weight with Gaussian distributed value 

weight = zeros(k, l);

for i = 1:2*k+1
    for j = 1:2*l+1
        weight(i,j) = 1/((sqrt(2*pi))*sigma^2)*exp(-((i-k-1)^2+(j-k-1)^2)/(2*sigma^2));
    end
end

end

