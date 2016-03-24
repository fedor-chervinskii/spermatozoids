function [ patches, labels ] = collect_patches(labels_dir, imgs_dir, m, ...
                                num_rotations, biases, getAngle, firstZero, neg)
    fileList = dir([labels_dir '*.csv']);
    fileList = {fileList.name}';

    patches = [];
    labels = [];
    
    for i = 1:numel(fileList)
        name = fileList{i}(1:end-4);
        fprintf('processing %s\n',name);
        image = imread([imgs_dir name '.tif']);
        [batch_centers, batch_labels] = ...
            GetCentersOfPatches([labels_dir name], m, getAngle, neg);
        [new_patches, new_labels] = ...
            augment_patches(m, image, num_rotations, firstZero,...
            biases, batch_centers, batch_labels);
        patches = [patches; new_patches];
        labels = [labels; new_labels];
    end
    
    %random permutation
    n = size(patches, 1);
    [patches, idx] = datasample(patches, n);
    labels = labels(idx);
end


function [centers, labels] = GetCentersOfPatches(name, m, getAngle, neg)
    d = floor(m/2);

    f = fopen([name '.csv'],'r');
    centers = zeros(5000,2);
    labels = zeros(5000,1);
    tline = fgetl(f);
    cur_counter = 1;
    while ischar(tline)
        if getAngle
            A = sscanf(tline, '%f,%f;%f,%f'); 
            Xc = round(A(3));
            Yc = round(A(1));
            angle = atan2d(A(4)-A(3),A(1)-A(2))+180;
            point = [Xc Yc];
        else
            A = sscanf(tline, '%f,%f,%d'); 
            Xc = round(A(2));
            Yc = round(A(1));
            if ~neg && ~A(3)
                tline = fgetl(f);
                continue
            end
            angle = A(3) - 1;
            point = [Xc Yc];
        end
        if IsInternalPoint(m, point)
            centers(cur_counter,:) = point;
            labels(cur_counter) = angle;
            cur_counter = cur_counter + 1;
        end
        tline = fgetl(f);   
    end
    if neg
        %collect negative centers
        centersKDT = KDTreeSearcher(centers);
    
        for Xc = round(linspace(1+d,512-d,60))
            for Yc = round(linspace(1+d,512-d,60))
                idx = knnsearch(centersKDT, [Xc, Yc]);
                if (centers(idx, 1) - Xc >= 4) && (centers(idx, 2) - Yc >= 4)
                    centers(cur_counter,:) = [Xc, Yc];
                    labels(cur_counter) = -1;
                    cur_counter = cur_counter + 1;
                end        
            end
        end
    end
    centers = centers(1:cur_counter-1,:);
    labels = labels(1:cur_counter-1,:);
end

function [result] = IsInternalPoint(m, point)
d = m/2;
result = (point(1)+d <= 512) && (point(1)-d >= 1) && ...
    (point(2)+d <= 512) && (point(2)-d >= 1);
end