img_filename = 'images/val/C001H001S0001000002_4.tif'
labels_filename = 'labels/orientations/val/C001H001S0001000002_4.csv'

m = 28
d = floor(m/2)

load('exp/det_net.mat')
load('exp/regr_net.mat')
load('exp/bi_net.mat')

% 1. Open the new unlabeled image
image = single(imread(img_filename));
imsize = size(image);
%imshow(img_filename);
%hold on;

height = imsize(1);
width = imsize(2);

faster = false;

%calculate output dimensions
if faster
    out_height = floor((floor((height - 4)/2) - 4)/2) - 3;
    out_width = floor((floor((width - 4)/2) - 4)/2) - 3;
    prob_map = zeros(out_height,out_width);
else
    out_height = floor((floor((height - 3 - 4)/2) - 4)/2) - 3;
    out_width = floor((floor((width - 3 - 4)/2) - 4)/2) - 3;
    prob_map = zeros(out_height*4,out_width*4);
end
    
image = image - 122; %mean subtraction

if ~faster
    for i = 1:4
        for j = 1:4
            det_net = det_net.makePass({single(image(i:end-4+i,j:end-4+j));
                                         single(zeros(out_height, out_width, 2, 1))});
            x = det_net.getBlob('prediction');
            prob_map(i:4:end-4+i,j:4:end-4+j) = ...
                exp(x(:,:,2))./(exp(x(:,:,1))+exp(x(:,:,2)));  %hand-crafted softmax
            fprintf('%d/16\n',4*(i-1) + j)
        end
    end
    [y, x] = nonmaxsuppts(prob_map, 6, 0.97);

    im_x = x + 14;
    im_y = y + 14;
else
    det_net = det_net.makePass({single(image);single(zeros(out_height, out_width, 2, 1))}); x = det_net.getBlob('prediction');
    prob_map(:,:) = exp(x(:,:,2))./(exp(x(:,:,1))+exp(x(:,:,2)));
    [y, x] = nonmaxsuppts(prob_map, 1, 0.5);
    im_x = x.*4 + 12;
    im_y = y.*4 + 12;
end

num_found = numel(im_x)

patches = zeros(m,m,1,num_found);

for i = 1:num_found
    Xc = im_x(i);
    Yc = im_y(i);
    patches(:,:,1,i) = image(Yc-d:Yc+d-1,Xc-d:Xc+d-1);
end

regr_net = regr_net.makePass({single(patches); single(zeros(1,1,2,num_found))});
prediction = regr_net.getBlob('prediction');
angles = atan2d(-prediction(1,:),-prediction(2,:));
angles = (angles'+180)./2;

centers = [im_y, im_x];
biases = [0, 0];
rotated_patches = augment_patches(m, image, 1, true, biases,...
    centers, angles);
rotated_patches = reshape(rotated_patches',m,m,1,num_found);

bi_net = bi_net.makePass({single(rotated_patches); single(zeros(2,num_found))});
prediction = bi_net.getBlob('prediction');
k = (prediction(1,:) < prediction(2,:));
%k = (squeeze(prediction) > 0);
angles = angles + k'*180;

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
fn = sum(assignment == 0);
angle_errors = zeros(num_found,1);
for k = 1:num_pos
    if assignment(k)
        diff = abs(gt_angles(k) - angles(assignment(k)));
        angle_errors(assignment(k)) = ...
            min(diff, 360-diff);
    end
end
tp = sum(assignment > 0);
fp = num_found - tp;
fn = num_pos - tp;
recall = tp/num_pos;
precision = tp/num_found;
f_measure = 2*recall*precision/(recall+precision);

figure
subplot(1,2,1)
h = histogram(angle_errors(angle_errors>0),180)
subplot(1,2,2)
plot(cumsum(h.Values)./num_pos)
%angle_errors = angle_errors(angle_errors ~= 0);