function DrawTestSpermatozoids()
%name = 'C001H001S0001000003';
%filename = ['images/raw/' name '.tif'];
name = 'C001H001S0001000002_3';
filename = ['images/' name '.tif'];
imshow(filename);
hold on;
load('exp/heat_map_data.mat');
p = FastPeakFind(prob_map);
p = p + 14;
plot(p(1:2:end),p(2:2:end),'r+');
figure()
imshow(-prob_map,[-1 0]);
end

