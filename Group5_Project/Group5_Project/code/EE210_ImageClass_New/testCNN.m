function [accuracy , store_decisions] = testCNN(images,labels,CNN)

[N_valid,~] = size(labels);
n_correct = 0;
store_decisions = zeros(1,N_valid);
for i = 1:N_valid
    
    H_raw = squeeze(images(:,i,:));
    [y1,~,~,~,~,~]= feedforward(H_raw, CNN );
%     [~,store_decisions(i)] = max(y1);
%     if (store_decisions(i) ~= find(y1==max(y1)))
%         x = 0;
%     end
    
     n_correct = n_correct+ max(double(find(y1==max(y1))) == (labels(i)+1));
    
end

%sum((store_decisions) == (labels_valid+1).') / N_valid

accuracy = n_correct/N_valid;

end