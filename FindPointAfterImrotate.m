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