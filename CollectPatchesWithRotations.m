function [ patches ] = CollectPatchesWithRotations(m, num_rotations)
    fileList = dir('labels/*.csv');
    fileList = {fileList.name}';

    patches = [];

    for i = 1:numel(fileList)
        name = fileList{i}(1:end-4);
        fprintf('processing %s\n',name);
        image = imread(['images/' name '.tif']);

        [centers, labels] = GetCentersOfTrainPatches(m, name);
        patches = [patches; GetAllRotationsForTrainPatches(m, image,...
            num_rotations, centers, labels)];
    end
end


function [centers, labels] = GetCentersOfTrainPatches(m, name)
    d = (m-1)/2;

    f = fopen(['labels/' name '.csv'],'r');
    centers = zeros(5000,2);
    labels = zeros(5000,1);
    tline = fgetl(f);
    cur_counter = 1;
    while ischar(tline)
        A = sscanf(tline, '%f,%f;%f,%f');
        Xc = round(A(3));
        Yc = round(A(1));
        point = [Xc Yc];
        if IsInternalPoint(m, point)
            centers(cur_counter,:) = point;
            labels(cur_counter) = 1;
            cur_counter = cur_counter + 1;
        end
        tline = fgetl(f);   
    end
    
    %collect negative centers
    centersKDT = KDTreeSearcher(centers);
    
    for Xc = round(linspace(1+d,512-d,50))
        for Yc = round(linspace(1+d,512-d,50))
            idx = knnsearch(centersKDT, [Xc, Yc]);
            if (centers(idx, 1) - Xc >= 4) && (centers(idx, 2) - Yc >= 4)
                centers(cur_counter,:) = [Xc, Yc];
                labels(cur_counter) = 0;
                cur_counter = cur_counter + 1;
            end        
        end
    end
    
    centers = centers(1:cur_counter-1,:);
    labels = labels(1:cur_counter-1,:);
end


function [patches] = GetAllRotationsForTrainPatches(m, image,...
        num_rotations, centers, labels)
    patches = zeros(size(centers, 1) * num_rotations, m^2+1);
    num_collected_patches = 0;
    d = (m-1)/2;
    for pair = [centers'; labels']
        center = pair(1:2);
        label = pair(3);
        for i = 0:num_rotations-1
            angle = 360 / num_rotations * i;
            rotated_image = imrotate(image, angle, 'bilinear', 'crop');
            rotated_center = FindPointAfterImrotate(image, angle, center(1), center(2));
            if IsInternalPoint(m, rotated_center)
                num_collected_patches = num_collected_patches + 1;
                patches(num_collected_patches, 1:m^2) = reshape(rotated_image(rotated_center(1)-d:rotated_center(1)+d, rotated_center(2)-d:rotated_center(2)+d)', m^2, 1);
                patches(num_collected_patches, m^2+1) = label;
            end
        end
    end
    patches = patches(1:num_collected_patches, :);
end


function [result] = IsInternalPoint(m, point)
d = (m-1)/2;
result = (point(1)+d <= 512) && (point(1)-d >= 1) && ...
    (point(2)+d <= 512) && (point(2)-d >= 1);
end