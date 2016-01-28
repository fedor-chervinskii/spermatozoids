function apply(filename)

m = 28
d = m/2;

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

% 2. Pass the image through the detection net
image = preprocess(image);
for i = 1:4
    for j = 1:4
        det_net = det_net.makePass({single(image(i:end-4+i,j:end-4+j));
                                        single(zeros(out_height, out_width, 2, 1))});
        x = det_net.getBlob('prediction');
        prob_map(i:4:end-4+i,j:4:end-4+j) = ...
            exp(x(:,:,2))./(exp(x(:,:,1))+exp(x(:,:,2)));  %hand-crafted softmax
        fprintf('%d/16\n',4*(i-1) + j)
    end
end

% 3. Detect maxima of output
[y, x] = nonmaxsuppts(prob_map, 6, 0.97);

im_x = x + d;
im_y = y + d;

n_centers = numel(im_x)

% 4. Cut patches with cells proposals
patches = zeros(m,m,1,n_centers);

for i = 1:n_centers
    Xc = im_x(i);
    Yc = im_y(i);
    patches(:,:,1,i) = image(Yc-d:Yc+d-1,Xc-d:Xc+d-1);
end

% 5. Angle regression on the patches
regr_net = regr_net.makePass({single(patches); single(zeros(1,1,2,n_centers))});
prediction = regr_net.getBlob('prediction');
angles = atan2d(-prediction(1,:),-prediction(2,:));
angles = (angles+180)./2;

% 6. Rotate all patches to a horizontal orientation
centers = [im_y, im_x];
labels = angles';
biases = [0, 0];
rotated_patches = augment_patches(m, image, 1, true, biases,...
    centers, labels);
rotated_patches = reshape(rotated_patches',m,m,1,n_centers);

% 7. Resolve the ambiguity
bi_net = bi_net.makePass({single(rotated_patches); single(zeros(2,n_centers))});
prediction = bi_net.getBlob('prediction');
k = (prediction(1,:) < prediction(2,:)).*2 - 1;

% plot vectors
for i = 1:n_centers
    vector = [cosd(angles(i)) -sind(angles(i))]*k(i)*7;
    line([im_x(i) im_x(i)+vector(1)],[im_y(i) im_y(i)+vector(2)],'Linewidth',4);
end

% plot centers
plot(im_x, im_y, 'r.', 'MarkerSize',20)