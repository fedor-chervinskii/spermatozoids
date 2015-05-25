function PrepareProbMap( net )
m = 29;
d = (m-1)/2;

% loading 'mean_image' variable
load('exp/train_params.mat');

% 1. Open the new unlabeled image
name = 'C001H001S0001000002_3';
filename = ['images/' name '.tif'];
image = single(imread(filename));
figure
imshow(filename);
hold on;

% 2. Iterate over the internal image points, call the feedforward 
% and get the value. For each value plot a red point if it exceeds the
% threshold. Mockup: output point and value to terminal.
num_patches_in_row = size(image, 2)-m+1;
num_cols = 10;
prob_map = zeros(num_cols,num_patches_in_row);

for i = 1:size(image, 1)-m+1
    patches = zeros(num_patches_in_row,m*m);
    for j = 1:num_patches_in_row
        patches(j,1:end) = reshape(image(i:i+m-1,j:j+m-1),1,m*m);
    end
    patches = permute(reshape(patches, [num_patches_in_row m m 1]), [2 3 4 1]);
    patches = bsxfun(@minus, patches, mean_image);
    net = net.makePass({single(patches); single(zeros(2, num_patches_in_row))});
    prediction = net.getBlob('prediction');
    prob_map(i,:) = prediction(1,:);
    for j = 1:num_patches_in_row
        prediction_for_point = prediction(1,j);
        if prediction_for_point < -1.7
            fprintf('i = %d, j = %d, prediction = %f\n',i,j,prediction_for_point);
            plot(i+d,j+d,'r.');
        end
    end
end

save('exp/heat_map_data.mat', 'prob_map');

hold off;

end
