function DrawTestSpermatozoids()

load('exp/heat_map_data.mat');
load('exp/test_filename');
imshow(filename);
hold on;

%num_orient_classes = size(prob_map,3) - 1
% angle = atan2d(A(2)-A(1),A(4)-A(3))+180;
%angles = (((1:num_orient_classes) - 0.5) * 360/(num_orient_classes*2) - 180)
%directions = [cosd(angles)' sind(angles)']

[y, x] = nonmaxsuppts(prob_map(:,:,1), 4, 0.8);

im_x = x + 14;
im_y = y + 14;
numel(x)
%for i = 1:numel(x)
%    probs = squeeze(prob_map(y(i),x(i),:));
%    [m, idx] = max(probs);
%    vector = 5*directions(idx-1,:);
%    %vector = 5*probs(2:num_orient_classes+1)'*directions;
%    line([im_x(i)-vector(1) im_x(i)+vector(1)],[im_y(i)-vector(2) im_y(i)+vector(2)]);
%end
scatter(im_x, im_y, 4,'r')

save('test_centers.mat','im_x', 'im_y')

%p = FastPeakFind(prob_map(:,:,1));
%p = p + 14;
%plot(p(1:2:end),p(2:2:end),'r+');
end

