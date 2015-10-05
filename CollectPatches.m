function [ patches ] = CollectPatches(labels_dir, imgs_dir, m, ...
                                num_rotations, getAngle, firstZero)
    fileList = dir([labels_dir '*.csv']);
    fileList = {fileList.name}';

    patches = [];

    for i = 1:numel(fileList)
        name = fileList{i}(1:end-4);
        fprintf('processing %s\n',name);
        image = imread([imgs_dir name '.tif']);

        [centers, labels] = GetCentersOfPatches(m, getAngle, [labels_dir name]);
        patches = [patches; GetAugmentedPatches(m, image,...
            num_rotations, firstZero, centers, labels)];
    end
end


function [centers, labels] = GetCentersOfPatches(m, getAngle, name)
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
            A = sscanf(tline, '%f,%f'); 
            Xc = round(A(2));
            Yc = round(A(1));
            angle = 0;
            point = [Xc Yc];
        end
        if IsInternalPoint(m, point)
            centers(cur_counter,:) = point;
            labels(cur_counter) = angle;
            cur_counter = cur_counter + 1;
        end
        tline = fgetl(f);   
    end
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
    
    centers = centers(1:cur_counter-1,:);
    labels = labels(1:cur_counter-1,:);
end


function [patches] = GetAugmentedPatches(m, image,...
        num_rotations, firstVertical, centers, labels)
    [X, Y] = meshgrid(-2:1:2,-2:1:2);
    biases = [X(:) Y(:)];
    %biases = [0 0];
    num_translations = size(biases, 1);
    patches = zeros(size(centers, 1) * num_rotations * num_translations, m^2+1);
    num_collected_patches = 0;
    d = floor(m/2);
    for pair = [centers'; labels']
        center = pair(1:2);
        label = pair(3);
        for i = 0:num_rotations-1
            angle = 360 / num_rotations * i;
            if firstVertical
                angle = angle - label;
            end
            rotated_image = imrotate(image, angle, 'bilinear', 'crop');
            rotated_center = FindPointAfterImrotate(image, angle, center(1), center(2));
            for j = 1:num_translations
                updated_center = rotated_center + biases(j,:);
                if IsInternalPoint(m, updated_center)
                    num_collected_patches = num_collected_patches + 1;
                    patches(num_collected_patches, 1:m^2) = reshape(rotated_image(updated_center(1)-d:updated_center(1)+d-1, updated_center(2)-d:updated_center(2)+d-1), m^2, 1);
                    if label >= 0
                        % has a spermatozoon, label = angle
                        orig_angle = label;
                        new_angle = rem(orig_angle+angle+360,360);
                        new_label = new_angle;
                    else
                        new_label = -1;
                    end
                    patches(num_collected_patches, m^2+1) = new_label;
                end
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