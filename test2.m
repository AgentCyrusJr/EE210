sobel_mask_x = [-1, 0, 1; -2, 0, 2; -1, 0, 1];
sobel_mask_y = [-1, -2, -1; 0, 0, 0; 1, 2, 1];
K =1;
path = 'seven.png';
image = imread(path);
image = [1,2,3;4,5,6;7,8,9];
[r,c] = size(image);
min_size = min(r,c);

padding_margin = 1;

image = reshape(image, min_size, min_size);

image = [zeros(min_size, padding_margin) , image , zeros(min_size, padding_margin)];

image = [zeros(padding_margin, min_size+2*padding_margin) ; image ; zeros(padding_margin, min_size+2*padding_margin)];

image = reshape(image, (min_size+2*padding_margin)*(min_size+2*padding_margin), 1);
sparse_index = zeros(min_size^2*(2*K+1)^2,2);
count =1;
for r = padding_margin+1 : min_size+padding_margin
    for c = padding_margin+1 : min_size + padding_margin
        for l = 1 : 2*K+1
            for k = 1 : 2*K+1
                sparse_index(count,:) = [count, (c-1+l-1-K)*(min_size+2*padding_margin)+r+k-1-K];
                count =count+1;
            end
        end
    end
end
partitionMatrix = sparse(sparse_index(:,1),sparse_index(:,2),ones(min_size^2*(2*K+1)^2,1),min_size^2*(2*K+1)^2,(min_size+2*padding_margin)^2);


