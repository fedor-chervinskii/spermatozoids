max_steps = 0:0.05:0.85
rad_steps = 2:2:10
nearest_thresh = 5

tp = zeros(n_max_steps,n_rad_steps);
fp = zeros(n_max_steps,n_rad_steps);
fn = zeros(n_max_steps,n_rad_steps);
recall = zeros(n_max_steps,n_rad_steps);
precision = zeros(n_max_steps,n_rad_steps);
f_measure = zeros(n_max_steps,n_rad_steps);
costs = zeros(n_max_steps,n_rad_steps);
for i = 1:numel(max_steps)
    for j = 1:numel(rad_steps)
        fprintf('radius = %i, max threshold = %.2f\n', rad_steps(j), max_steps(i));
        [y, x] = nonmaxsuppts(prob_map, rad_steps(j), max_steps(i));
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
legends = cell(n_rad_steps,1);
for i = 1:n_rad_steps
    legends{i} = sprintf('radius = %i',i);
end
for i = 1:numel(metrics)
    figure
    plot(metrics{i});
    title(titles(i));
    legend(legends);
end