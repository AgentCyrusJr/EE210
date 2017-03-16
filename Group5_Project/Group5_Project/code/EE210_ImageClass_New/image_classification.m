clear
clc

%% Data Preprocessing
image_size = 256;
Files1 = dir(strcat('image_class1/','*.jpg'));
Files2 = dir(strcat('image_class2/','*.jpg'));
Files3 = dir(strcat('image_class3/','*.jpg'));
Files4 = dir(strcat('image_class4/','*.jpg'));
LengthFiles1 = length(Files1);
LengthFiles2 = length(Files2);
LengthFiles3 = length(Files3);
LengthFiles4 = length(Files4);

%%% total size of the image from class 1 and 2
images=zeros(image_size^2,LengthFiles1+LengthFiles2,3);
images_labels  = zeros(LengthFiles1+LengthFiles2,1);
%%%load the image of class 1
for i = 1:LengthFiles1;
    for depth = 1:3
        Img = imread(strcat('image_class1/',Files1(i).name));
        images(:,i,depth) = double(reshape(Img(:,:,depth),image_size^2,1))./255;
        images_labels(i) = 0;
    end
end
%%%load the image of class 2
for i = 1:LengthFiles2;
    for depth = 1:3
        Img = imread(strcat('image_class2/',Files2(i).name));
        images(:,i+LengthFiles1,depth) = double(reshape(Img(:,:,depth),image_size^2,1))./255;
        images_labels(i+LengthFiles1) = 1;
    end
end
%%%load the image of class 3
for i = 1:LengthFiles3;
    for depth = 1:3
        Img = imread(strcat('image_class3/',Files3(i).name));
        images(:,i+LengthFiles1+LengthFiles2,depth) = double(reshape(Img(:,:,depth),image_size^2,1))./255;
        images_labels(i+LengthFiles1+LengthFiles2) = 2;
    end
end
%%%load the image of class 4
for i = 1:LengthFiles4;
    for depth = 1:3
        Img = imread(strcat('image_class4/',Files4(i).name));
        images(:,i+LengthFiles1+LengthFiles2+LengthFiles3,depth) = double(reshape(Img(:,:,depth),image_size^2,1))./255;
        images_labels(i+LengthFiles1+LengthFiles2+LengthFiles3) = 3;
    end
end
N = LengthFiles1 + LengthFiles2 + LengthFiles3 + LengthFiles4;
Indices = crossvalind('Kfold', N, 5);
images_train = images(:,find(Indices>=2&Indices<=5),:);
labels_train = images_labels(find(Indices>=2&Indices<=5),:);

images_valid = images(:,find(Indices==1),:);
labels_valid = images_labels(find(Indices==1),:);


%% Determine HyperParameters

% size of filter in each layer
filter_size = [3,3];

% pooling filter size
pool_filtersize = [2,2];

% # of neuron nodes
N_neuron = [200,4];

% value of sride in each layer
stride = [2,1]; 

% value of padding margin in each layerH
%padding_margin = [1,1];
padding_margin = (filter_size - 1) / 2;

% # of depth in each layer
D = [1,5,5];

% step size
mu = 0.01;

% regularization coefficient for Wc
rho1 = 0.01;

% regularization coefficient for Wl
rho2 = 0.01;

% batch size
Batch_size = 1;

% maximum number of epoch
MaxEpoch = 501;


%%%% Automatically calculate the size of P and Pp
P = zeros(1,length(D)-1);
Pp = zeros(1,length(D)-1);
P(1) = floor((image_size + 2*padding_margin(1) - filter_size(1)) / stride(1) + 1 ).^2;
Pp(1) = floor( sqrt(P(1)) / pool_filtersize(1) ).^2;

for i=2:length(P)
    P(i) =  floor((sqrt(Pp(i-1)) + 2*padding_margin(i) - filter_size(i)) / stride(i) + 1 ).^2;
    Pp(i) = floor( sqrt(P(i)) / pool_filtersize(i) ).^2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Initialization of the structure of CNN
CNN = initial_CNN( D , P , Pp , N_neuron , filter_size , padding_margin , stride, pool_filtersize,image_size*ones(1,2));
[~,N_train,~] = size(images_train);
costVal = [];
train_accuracy = [];

CNN_orig = CNN;

images_train_orig = images_train;
labels_train_orig = labels_train;

r_perm = randperm(N_train);
images_train = images_train_orig(:,r_perm,:);
labels_train = labels_train_orig(r_perm);

CNN_update = CNN;

for epoch = 1:MaxEpoch

%       r_sample = datasample(1:N_train,Batch_size);
%       images_train = images_train_orig(:,r_sample);
%       labels_train = labels_train_orig(r_sample);
      
    for index=1:Batch_size;

        gamma = zeros(4,1);
        gamma(labels_train(epoch)+1)=1;
        H_raw =squeeze( images_train(:,epoch,:));

        CNN_update = grad_desc(CNN,H_raw,gamma,mu,rho1,rho2,Batch_size,CNN_update);
        if index==Batch_size
            CNN = CNN_update;
        end
        
    end
       [y1,~,~,~,~,~]= feedforward( H_raw, CNN );
       cost = - sum(gamma .* log(y1)+(1-gamma).* log(1-y1));
       costVal = [costVal , cost];
       [train_accuracy1 , ~] = testCNN(images_train_orig,labels_train_orig,CNN);
       train_accuracy = [train_accuracy , train_accuracy1];
       epoch

end
%% Plot
figure;
plot(1:MaxEpoch,costVal);
xlabel('epoch');
ylabel('Cost Funcation Value'); 
title('Cost Funcation Value - epoch');
figure;
plot(1:MaxEpoch,train_accuracy);
xlabel('epoch');
ylabel('Training Accuracy');
title('Training Accuracy - epoch');
%%

[test_accuracy , store_decisions] = testCNN(images_valid,labels_valid,CNN);

