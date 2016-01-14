function prob_map = PrepareProbMap()
m = 28
d = floor(m/2);

load('exp/baseline_net.mat')

num_classes = 37;

mean_image = 122;

% 1. Open the new unlabeled image
name = 'C001H001S0001000002_4';
filename = ['images/val/' name '.tif'];
%name = 'C001H001S0001000002_3';
%filename = ['images/' name '.tif'];
image = single(imread(filename));
figure
imshow(filename);
hold on;

% 2. Iterate over the internal image points, call the feedforward 
% and get the value. For each value plot a red point if it exceeds the
% threshold. Mockup: output point and value to terminal.
num_patches_in_row = size(image, 2)-m+1;
num_cols = 10;
prob_map = ones(num_cols,num_patches_in_row,num_classes);

for i = 1:size(image, 1)-m+1
    i
    patches = zeros(num_patches_in_row,m*m);
    for j = 1:num_patches_in_row
        patches(j,1:end) = reshape(image(i:i+m-1,j:j+m-1),1,m*m);
    end
    patches = permute(reshape(patches, [num_patches_in_row m m 1]), [2 3 4 1]);
    patches = bsxfun(@minus, patches, mean_image);
    det_net = det_net.makePass({single(patches); single(zeros(1,1,num_classes, num_patches_in_row))});
    x = det_net.getBlob('prediction');
    probs = squeeze(x);
    prob_map(i,:,1) = probs;
    %y = exp(bsxfun(@minus,x,max(x,[],3)));
    %probs = bsxfun(@rdivide,y,sum(y,3));
    %probs = squeeze(probs);
    %prob_map(i,:,:) = probs';
%    for j = 1:num_patches_in_row
%        prediction_for_point = prediction(1,j);
%        if prediction_for_point < 0.5
%            fprintf('i = %d, j = %d, prediction = %f\n',i,j,prediction_for_point);
%            plot(i+d,j+d,'r.');
%        end
%    end
end

save('exp/heat_map_data.mat', 'prob_map');
save('exp/test_filename', 'filename');

hold off;

end
