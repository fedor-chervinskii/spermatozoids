max_steps = 0.9:0.01:0.99
rad_steps = 2:2:10
nearest_thresh = 6

arr_size = [numel(max_steps),numel(rad_steps)];

tp = zeros(arr_size);
fp = zeros(arr_size);
fn = zeros(arr_size);
recall = zeros(arr_size);
precision = zeros(arr_size);
f_measure = zeros(arr_size);
costs = zeros(arr_size);
for i = 1:numel(max_steps)
    for j = 1:numel(rad_steps)
        fprintf('radius = %i, max threshold = %.3f\n', rad_steps(j), max_steps(i));
        [y, x] = nonmaxsuppts(prob_map, rad_steps(j), max_steps(i));
        %cent = FastPeakFind(prob_map, max_steps(i), fspecial('gaussian', [20, 20], rad_steps(j)));
        %x = cent(1:2:end-1);
        %y = cent(2:2:end);
        y = y + d;
        x = x + d;
        num_found = numel(x)
        pred_centers = [x, y];
        D = pdist2(gt_centers,pred_centers);
        D(D > nearest_thresh) = Inf;
        [assignment, cost] = assignmentoptimal(D);
        costs(i,j) = cost;
        tp(i,j) = sum(assignment > 0);
        fp(i,j) = num_found - tp(i,j);
        fn(i,j) = num_pos - tp(i,j);
        recall(i,j) = tp(i,j)/num_pos;
        precision(i,j) = tp(i,j)/num_found;
        f_measure(i,j) = 2*recall(i,j)*precision(i,j)/(recall(i,j)+precision(i,j));
    end
end
metrics = {fn,fp,tp,recall,precision,f_measure};
titles = {'False Negatives','False Positives','True Positives','Recall',...
          'Precision','F-measure'};
legends = cell(numel(rad_steps),1);
for i = 1:numel(rad_steps)
    legends{i} = sprintf('radius = %i',rad_steps(i));
end
for i = 1:numel(metrics)
    figure
    plot(metrics{i});
    title(titles(i));
    legend(legends);
end