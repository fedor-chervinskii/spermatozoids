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
net = net.makePass({single(image); single(zeros(250, 250, 5, 1))});
prediction = net.getBlob('softmax');
prob_map = prediction(:,:,:,1);
map = [[0 0 0];[1 0 0];[0 1 0];[0 0 1];[1 0 1]]
prob_map = reshape(mtimes(reshape(prob_map,[],5),map),250,250,3);
imshow(prob_map);

save('exp/heat_map_data.mat', 'prob_map');

hold off;

end