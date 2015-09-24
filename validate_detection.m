img_filename = 'images/val/C001H001S0001000002_4.tif'
labels_filename = 'labels/val/C001H001S0001000002_4.csv'

m = 28
d = floor(m/2)

load('exp/det_net.mat')
load('exp/regr_net.mat')

% 1. Open the new unlabeled image
image = single(imread(img_filename));
imsize = size(image);
%imshow(img_filename);
%hold on;

height = imsize(1);
width = imsize(2);
out_height = floor((floor((height - 3 - 4)/2) - 4)/2) - 3;
out_width = floor((floor((width - 3 - 4)/2) - 4)/2) - 3;
prob_map = zeros(out_height*4,out_width*4);

image = image - 122;
for i = 1:4
    for j = 1:4
        det_net = det_net.makePass({single(image(i:end-4+i,j:end-4+j));
                                        single(zeros(out_height, out_width, 1, 1))});
        x = det_net.getBlob('prediction');
        prob_map(i:4:end-4+i,j:4:end-4+j) = squeeze(x);
        fprintf('%d/16\n',4*(i-1) + j)
    end
end

f = fopen(labels_filename,'r');
centers = zeros(5000,2);
tline = fgetl(f);
counter = 1;

while ischar(tline)
    A = sscanf(tline, '%f,%f;%f,%f');
    centers(counter,:) = [round(A(3)) round(A(1))];
    tline = fgetl(f);
    counter = counter + 1;
end
gt_centers = centers(1:counter-1,:);
gt_y = gt_centers(:,1);
gt_x = gt_centers(:,2);
num_pos = numel(gt_x)
gt_centers = [gt_x, gt_y];

n_steps = 10;
tp = zeros(n_steps,1);
fp = zeros(n_steps,1);
fn = zeros(n_steps,1);
recall = zeros(n_steps,1);
precision = zeros(n_steps,1);
f_measure = zeros(n_steps,1);
costs = zeros(n_steps,1);
for i = 1:n_steps
    [y, x] = nonmaxsuppts(prob_map(:,:,1), 4, i/n_steps);
    y = y + d;
    x = x + d;
    num_found = numel(x)
    pred_centers = [x, y];
    figure
    imshow(img_filename);
    hold on;
    scatter(x, y, 4,'r');
    scatter(gt_x, gt_y, 4,'b');
    D = pdist2(gt_centers,pred_centers);
    D(D > 10) = Inf;
    [assignment, cost] = assignmentoptimal(D);
    costs(i) = cost;
    fn(i) = sum(assignment == 0);
    is_assigned = zeros(num_found,1);
    for j = 1:num_pos
        if assignment(j)
            is_assigned(assignment(j)) = 1;
        end
    end
    tp(i) = sum(is_assigned == 1);
    fp(i) = num_found - tp(i);
    recall(i) = tp(i)/num_pos;
    precision(i) = tp(i)/num_found;
    f_measure(i) = 2*recall(i)*precision(i)/(recall(i)+precision(i));
end
