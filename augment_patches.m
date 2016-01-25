function [patches, labels] = augment_patches(m, image,...
        num_rotations, firstZero, biases, centers, labels)

    num_translations = size(biases, 1);
    patches = zeros(size(centers, 1) * num_rotations * num_translations, m^2);
    new_labels = zeros(size(centers, 1) * num_rotations * num_translations, 1);
    num_collected_patches = 0;
    d = floor(m/2);
    for pair = [centers'; labels']
        center = pair(1:2);
        label = pair(3);
        for i = 0:num_rotations-1
            angle = 360 / num_rotations * i;
            if firstZero
                angle = angle - label;
            end
            rotated_image = imrotate(image, angle, 'bilinear');
            rotated_center = FindPointAfterImrotate(size(image),...
                    size(rotated_image), angle, center(1), center(2));
            for j = 1:num_translations
                updated_center = rotated_center + biases(j,:);
                if IsInternalPoint(size(rotated_image), m, updated_center)
                    num_collected_patches = num_collected_patches + 1;
                    patches(num_collected_patches, :) = reshape(rotated_image(updated_center(1)-d:updated_center(1)+d-1, updated_center(2)-d:updated_center(2)+d-1), m^2, 1);
                    if label >= 0
                        % has a spermatozoon, label = angle
                        orig_angle = label;
                        new_angle = rem(orig_angle+angle+360,360);
                        new_label = new_angle;
                    else
                        new_label = -1;
                    end
                    new_labels(num_collected_patches) = new_label;
                end
            end  
         end
    end
    patches = patches(1:num_collected_patches, :);
    labels = new_labels(1:num_collected_patches, :);
    patches = int16(patches);
end

function [result] = IsInternalPoint(size, m, point)
    d = m/2;
    result = (point(1)+d <= size(1)) && (point(1)-d >= 1) && ...
        (point(2)+d <= size(2)) && (point(2)-d >= 1);
end

function [new_orig] = FindPointAfterImrotate(old_size, new_size, angle, orig_x, orig_y)
    old_center = (old_size + 1) / 2;
    new_center = (new_size + 1) / 2;
    orig_x = orig_x - old_center(2);
    orig_y = orig_y - old_center(1);
    rot_mat = [cosd(angle), -sind(angle); sind(angle), cosd(angle)];
    old_orig = [orig_y orig_x];
    mid_orig = old_orig * rot_mat;
    new_orig(1) = round(mid_orig(2) + new_center(2));
    new_orig(2) = round(mid_orig(1) + new_center(1));
end