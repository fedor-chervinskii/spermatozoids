function apply(filename)

m = 28
d = floor(m/2)

load('exp/det_net.mat')
load('exp/regr_net.mat')

% 1. Open the new unlabeled image
image = single(imread(filename));
imsize = size(image);
imshow(filename);
hold on;

height = imsize(1);
width = imsize(2);
out_height = floor((floor((height - 3 - 4)/2) - 4)/2) - 3;
out_width = floor((floor((width - 3 - 4)/2) - 4)/2) - 3;
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

[y, x] = nonmaxsuppts(prob_map(:,:,1), 4, 0.8);

im_x = x + 14;
im_y = y + 14;

n_centers = numel(im_x)

patches = zeros(m,m,1,n_centers);

for i = 1:n_centers
    Xc = im_x(i);
    Yc = im_y(i);
    patches(:,:,1,i) = image(Yc-d:Yc+d-1,Xc-d:Xc+d-1);
end

mean_image = 122;
patches = bsxfun(@minus, patches, mean_image);
regr_net = regr_net.makePass({single(patches); single(zeros(1,1,2,n_centers))});
prediction = regr_net.getBlob('prediction');
angles = atan2d(prediction(1,:),prediction(2,:))+180;
angles = angles./2;

for i = 1:n_centers
    vector = [-cosd(angles(i)) sind(angles(i))]*5;
    line([im_x(i)-vector(1) im_x(i)+vector(1)],[im_y(i)-vector(2) im_y(i)+vector(2)]);
end
scatter(im_x, im_y, 4,'r')

