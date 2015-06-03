function [new_orig] = FindPointAfterImrotate(image, angle, orig_x, orig_y)
sz = (size(image) + 1) / 2;
orig_x = orig_x - sz(2);
orig_y = orig_y - sz(1);
rot_mat = [cosd(angle), -sind(angle); sind(angle), cosd(angle)];
old_orig = [orig_y orig_x];
mid_orig = old_orig * rot_mat;
new_orig(1) = round(mid_orig(2) + sz(2));
new_orig(2) = round(mid_orig(1) + sz(1));
end