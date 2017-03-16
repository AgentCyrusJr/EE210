function y = activation_corr_fun(z,c);
% Activation function: enhance performance in case of convolutional network
% Parameters: output vector---z, index of convolution layer---c
% Return: feature map---y
% Details: depending on c, a different activation function can be applied
% if needed. Here we only implement y = tanh(z), more comes later.
    y = tanh(z);
    %y = sigmfb(z);
    %y=z;
end