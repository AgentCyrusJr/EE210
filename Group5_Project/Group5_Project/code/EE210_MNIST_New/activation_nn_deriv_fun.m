function y = activation_nn_deriv_fun(z,l)
%derivative of tanh    
if(l~=3)
y = 1.14393 * (1- tanh( 2/3 * z).^2);  
else
    y = sigmfb(z).*(1-sigmfb(z));
end
end