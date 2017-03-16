function y = activation_corr_deriv_fun(z,c)
% Activation function: enhance performance in case of convolutional network
% Parameters: output vector---z, index of convolution layer---c
% Return: feature map---y
% Details: depending on c, a different activation function can be applied
% if needed. Here we only implement y = tanh(z), more comes later.
    y = 1-tanh(z).^2;
    %y =(1-sigmfb(z)).* sigmfb(z);
    %y=1;
end