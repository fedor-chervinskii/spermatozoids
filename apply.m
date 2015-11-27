function apply(filename)

m = 28
d = floor(m/2)

load('exp/det_net.mat')
load('exp/regr_net.mat')
load('exp/bi_net.mat')

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

[y, x] = nonmaxsuppts(prob_map(:,:), 2, 0.1);

im_x = x + 14;
im_y = y + 14;

n_centers = numel(im_x)

patches = zeros(m,m,1,n_centers);

for i = 1:n_centers
    Xc = im_x(i);
    Yc = im_y(i);
    patches(:,:,1,i) = image(Yc-d:Yc+d-1,Xc-d:Xc+d-1);
end

regr_net = regr_net.makePass({single(patches); single(zeros(1,1,2,n_centers))});
prediction = regr_net.getBlob('prediction');
angles = atan2d(-prediction(1,:),-prediction(2,:));
angles = (angles+180)./2;

centers = [im_y, im_x];
labels = angles';
biases = [0, 0];
rotated_patches = GetAugmentedPatches(m, image, 1, true, biases,...
    centers, labels);
rotated_patches = reshape(rotated_patches',m,m,1,n_centers);

bi_net = bi_net.makePass({single(rotated_patches); single(zeros(1,1,1,n_centers))});
prediction = bi_net.getBlob('prediction');
k = [squeeze(prediction) > 0]*2-1;

for i = 1:n_centers
    vector = [cosd(angles(i)) -sind(angles(i))]*k(i)*7;
    line([im_x(i) im_x(i)+vector(1)],[im_y(i) im_y(i)+vector(2)],'Linewidth',4);
end

plot(im_x, im_y,'r.', 'MarkerSize',20)

for i = 1:10 %n_centers
    fh = figure;
    subplot(1,2,1)
    imshow(patches(:,:,1,i),[-127,127]);
    subplot(1,2,2)
    imshow(rotated_patches(:,:,1,i),[-127,127]);
    vector = k(i)*7;
    line([14 14+vector],[14 14],'Linewidth',4);
    waitfor(fh);
end

