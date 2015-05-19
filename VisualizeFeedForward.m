function [ prediction ] = VisualizeFeedForward ( net )

m = 29;
patches_edge = 7;
num_patches = patches_edge^2;
patches = MakeTestPatches(m,num_patches);
net = net.makePass({single(patches); single(zeros(2, num_patches))});
prediction = net.getBlob('prediction');

[~, order] = sort(prediction(1,:));

for i = 1:size(patches,4)
    subplot(patches_edge,patches_edge,i), imshow(reshape(patches(:,:,:,order(i)),m,m), [0 255]);
    title(prediction(1,order(i)));
end

end