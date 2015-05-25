function [ prediction ] = VisualizeFeedForward ( net )

m = 29;
patches_edge = 7;
num_patches = 300;
patches = MakeTestPatches(m,num_patches);
size(patches)

% loading 'mean_image' variable
load('exp/train_params.mat');

% subtracting the average
patches = bsxfun(@minus, patches, mean_image);
net = net.makePass({single(patches); single(zeros(2, num_patches))});
prediction = net.getBlob('softmax');

[~, order] = sort(prediction(1,:));

for i = 1:patches_edge^2
    subplot(patches_edge,patches_edge,i), imshow(reshape(patches(:,:,:,order(i)),m,m), [-150,150]);
    title(prediction(1,order(i)));
end

end