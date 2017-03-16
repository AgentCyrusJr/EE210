function [ output_args ] = plotPic( img )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
subplot(131); 
imagesc(img(:,:,1)); 
title('Red'); 

subplot(132); 
imagesc(img(:,:,2)); 
title('Green'); 

subplot(133); 
imagesc(img(:,:,3)); 
title('Blue'); 

end