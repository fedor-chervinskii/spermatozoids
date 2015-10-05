img_filename = 'images/val/C001H001S0001000002_4.tif'
labels_filename = 'labels/centers/val/C001H001S0001000002_4.csv'

m = 28
d = floor(m/2)

load('exp/det_net.mat')
%load('exp/regr_net.mat')

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
    A = sscanf(tline, '%f,%f'); 
    Xc = round(A(2));
    Yc = round(A(1));
    centers(counter,:) = [Xc Yc];
    tline = fgetl(f);
    counter = counter + 1;
end
gt_centers = centers(1:counter-1,:);
gt_y = gt_centers(:,1);
gt_x = gt_centers(:,2);
num_pos = numel(gt_x)
gt_centers = [gt_x, gt_y];

n_max_steps = 10
n_rad_steps = 5
nearest_thresh = 5

tp = zeros(n_max_steps,n_rad_steps);
fp = zeros(n_max_steps,n_rad_steps);
fn = zeros(n_max_steps,n_rad_steps);
recall = zeros(n_max_steps,n_rad_steps);
precision = zeros(n_max_steps,n_rad_steps);
f_measure = zeros(n_max_steps,n_rad_steps);
costs = zeros(n_max_steps,n_rad_steps);
for i = 1:n_max_steps
    for j = 1:n_rad_steps
        fprintf('radius = %i, max threshold = %.2f\n', j, i/n_max_steps);
        [y, x] = nonmaxsuppts(prob_map(:,:,1), j, i/n_max_steps);
        y = y + d;
        x = x + d;
        num_found = numel(x)
        pred_centers = [x, y];
        D = pdist2(gt_centers,pred_centers);
        D(D > nearest_thresh) = Inf;
        [assignment, cost] = assignmentoptimal(D);
        costs(i,j) = cost;
        fn(i,j) = sum(assignment == 0);
        is_assigned = zeros(num_found,1);
        for k = 1:num_pos
            if assignment(k)
                is_assigned(assignment(k)) = 1;
            end
        end
        tp(i,j) = sum(is_assigned == 1);
        fp(i,j) = num_found - tp(i,j);
        recall(i,j) = tp(i,j)/num_pos;
        precision(i,j) = tp(i,j)/num_found;
        f_measure(i,j) = 2*recall(i,j)*precision(i,j)/(recall(i,j)+precision(i,j));
    end
end
metrics = {fn,fp,tp,recall,precision,f_measure};
titles = {'False Negatives','False Positives','True Positives','Recall',...
          'Precision','F-measure'};
legends = cell(n_rad_steps,1);
for i = 1:n_rad_steps
    legends{i} = sprintf('radius = %i',i);
end
for i = 1:numel(metrics)
    figure
    plot(metrics{i});
    title(titles(i));
    legend(legends);
    savefig(['images/results/val/' titles{i} '.fig']);
end