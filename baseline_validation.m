m = 28;
d = m/2;

% Open the validation image
img_filename = 'images/val/C001H001S0001000002_4.tif'
labels_filename = 'labels/orientations/val/C001H001S0001000002_4.csv'

load('exp/baseline_net.mat')
net = baseline_net;
num_classes = size(baseline_net.layers{end-2, 1}.ex,1);

% 1. Open the image
image = single(imread(img_filename));
imsize = size(image);

height = imsize(1);
width = imsize(2);
%calculate output dimensions
out_height = floor((floor((height - 3 - 4)/2) - 4)/2) - 3;
out_width = floor((floor((width - 3 - 4)/2) - 4)/2) - 3;

image = image - 122; %mean subtraction

% 2. Iterate over the internal image points, call the feedforward 
% and get the predictions for classes.
num_patches_in_row = size(image, 2)-m+1;
num_rows = size(image, 1)-m+1;
prob_map = zeros(num_rows, num_patches_in_row, num_classes);

for i = 1:num_rows
    patches = zeros(num_patches_in_row,m,m);
    for j = 1:num_patches_in_row
        patches(j,:,:) = image(i:i+m-1,j:j+m-1);
    end
    patches = permute(reshape(patches, [num_patches_in_row m m 1]), [2 3 4 1]);
    net = net.makePass({single(patches); single(zeros(num_classes, num_patches_in_row))});
    prediction = net.getBlob('prediction');
    prob_map(i,:,:) = prediction';
    i
end

%take maximum over all classes, storing the matrix of corresponding classes
[segment_map, class_idx] = max(prob_map,[],3);
segment_map(class_idx == 1) = 0;

[y, x] = nonmaxsuppts(segment_map(:,:), 5, 10);

im_x = x + 14;
im_y = y + 14;

num_found = numel(im_x)

angles = zeros(num_found,1);
for i = 1:num_found
    angles(i) = 360/(num_classes-1)*(class_idx(y(i), x(i))-1.5);
end

f = fopen(labels_filename,'r');
gt_centers = zeros(5000,2);
gt_angles = zeros(5000,1);
tline = fgetl(f);
counter = 1;

while ischar(tline)
    A = sscanf(tline, '%f,%f;%f,%f'); 
    Xc = round(A(3));
    Yc = round(A(1));
    angle = atan2d(A(4)-A(3),A(1)-A(2))+180;
    gt_centers(counter,:) = [Xc Yc];
    gt_angles(counter,:) = angle;
    tline = fgetl(f);
    counter = counter + 1;
end

gt_centers = gt_centers(1:counter-1,:);
gt_angles = gt_angles(1:counter-1,:);
gt_y = gt_centers(:,1);
gt_x = gt_centers(:,2);
num_pos = numel(gt_x)
gt_centers = [gt_x, gt_y];

nearest_thresh = 5

pred_centers = [im_x, im_y];
D = pdist2(gt_centers,pred_centers);
D(D > nearest_thresh) = Inf;
[assignment, cost] = assignmentoptimal(D);
fp = sum(assignment == 0);
tp = sum(assignment > 0);
fn = num_pos - tp;
recall = tp/num_pos;
precision = tp/num_found;
f_measure = 2*recall*precision/(recall+precision);

angle_errors = zeros(num_found,1);
for k = 1:num_pos
    if assignment(k)
        diff = abs(gt_angles(k) - angles(assignment(k)));
        angle_errors(assignment(k)) = ...
            min(diff, 360-diff);
    end
end

%angle_errors = angle_errors(angle_errors ~= 0);