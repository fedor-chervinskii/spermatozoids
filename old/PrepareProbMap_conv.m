function prob_map = PrepareProbMap_conv()
m = 28;
d = floor(m/2);

load('exp/det_net.mat')

% 1. Open the new unlabeled image
name = 'C001H001S0001000003';
filename = ['images/raw/' name '.tif'];
image = single(imread(filename));
imsize = size(image);
height = imsize(1)
width = imsize(2)
out_height = floor((floor((height - 3 - 4)/2) - 4)/2) - 3
out_width = floor((floor((width - 3 - 4)/2) - 4)/2) - 3
prob_map = zeros(out_height*4,out_width*4);

image = image - 122;
for i = 1:4
    for j = 1:4
        det_net = det_net.makePass({single(image(i:end-4+i,j:end-4+j));
                                        single(zeros(out_height, out_width, 1, 1))});
        x = det_net.getBlob('prediction');
        prob_map(i:4:end-4+i,j:4:end-4+j) = squeeze(x);
        fprintf('%d/16\n',4*(i-1) + j)
    end
end

figure;
imshow(prob_map);

save('exp/heat_map_data.mat', 'prob_map');
save('exp/test_filename', 'filename');

hold off;

end