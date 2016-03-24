labels_dir = 'labels/centers/train/';
imgs_dir = 'images/train/';

fileList = dir([labels_dir '*.csv']);
fileList = {fileList.name}';

for i = 5:numel(fileList)
    name = fileList{i}(1:end-4);
    fprintf('processing %s\n',name);
    image = single(imread([imgs_dir name '.tif']));
    [pred_centers, angles] = apply([imgs_dir name '.tif']);
    %close all;
    hold on;
    
    
    f = fopen([labels_dir name '.csv'],'r');
    gt_centers = zeros(5000,2);
    tline = fgetl(f);
    counter = 1;

    while ischar(tline)
        A = sscanf(tline, '%f,%f,%d'); 
        if A(3) 
            Xc = round(A(1));
            Yc = round(A(2));
            gt_centers(counter,:) = [Yc Xc];
            counter = counter + 1;
        end
        tline = fgetl(f);
    end
    fclose(f);
    
    gt_centers = gt_centers(1:counter-1,:);

    nearest_thresh = 5;

    D = pdist2(pred_centers, gt_centers);
    D(D > nearest_thresh) = Inf;
    [assignment, cost] = assignmentoptimal(D);
    hn_centers = pred_centers(assignment == 0, :);
    scatter(hn_centers(:,2),hn_centers(:,1));
    hold on;
    scatter(gt_centers(:,2),gt_centers(:,1))
    fprintf('%d false positives (hard negatives) found\n',size(hn_centers,1));

    f = fopen([labels_dir name '.csv'],'a+');
    
    for j = 1:size(hn_centers,1)
        h = figure;
        im_x = hn_centers(j,2);
        im_y = hn_centers(j,1);
        imshow(image(im_y-14:im_y+13,im_x-14:im_x+13),[0,255],'InitialMagnification',200);
        hold on;
        plot(15,14,'or');
        a = input('Is this a cell at the center (1/0)? ','s');
        if strcmpi(a,'0')
            fprintf(f,'%.2f,%.2f,0\n',im_y,im_x);
        elseif strcmpi(a,'1')
            fprintf(f,'%.2f,%.2f,1\n',im_y,im_x);
        end
        close(h);
    end
    fclose(f);
end