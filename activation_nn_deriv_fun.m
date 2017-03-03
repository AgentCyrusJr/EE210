function y = activation_nn_deriv_fun(z,l)
%derivative of tanh    
y = 1.14393 * (1- tanh( 2/3 * z).^2);  
end