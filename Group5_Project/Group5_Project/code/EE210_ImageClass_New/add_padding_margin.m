function [vector_added_padding_margin, padding_matrix]= add_padding_margin(v,padding_margin0)
v = v(:);
l = length(v);
size_img = sqrt(l);
temp = reshape(v,size_img,size_img);
temp = [zeros(size_img, padding_margin0) , temp , zeros(size_img, padding_margin0)];

temp = [zeros(padding_margin0, size_img+2*padding_margin0) ; temp ; zeros(padding_margin0, size_img+2*padding_margin0)];

vector_added_padding_margin = reshape(temp, (size_img+2*padding_margin0)*(size_img+2*padding_margin0), 1);


size_img = sqrt(length(v));
% list all pixel_locations

new_size_img = 2*padding_margin0 + size_img;
indexes = zeros(size_img^2,2);
count = 0;
for r = 1:size_img
    for c = 1:size_img
        count = count + 1;
        old_indx = (c-1)*size_img + r;
        new_indx = (c+padding_margin0-1)*new_size_img + (r + padding_margin0);
        indexes(count,:) = [new_indx old_indx];
    end
end

padding_matrix = sparse(indexes(:,1), indexes(:,2), ones(size_img^2,1),new_size_img^2,size_img^2);


end