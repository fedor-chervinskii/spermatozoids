function prob_map = PrepareProbMap_conv( net )
m = 29;
d = (m-1)/2;

% loading 'mean_image' variable
load('exp/train_params.mat');

% 1. Open the new unlabeled image
name = 'C001H001S0001000003';
filename = ['images/raw/' name '.tif'];
image = single(imread(filename));
imsize = size(image);
figure
imshow(filename);

image = image - 122;
net = net.makePass({single(image); single(zeros(250, 250, 2, 1))});
prediction = net.getBlob('softmax');
prob_map = prediction(:,:,2,1);
imshow(prob_map, 'Colormap',copper);

save('exp/heat_map_data.mat', 'prob_map');

hold off;

end