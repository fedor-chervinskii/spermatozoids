function apply_then_label_centers(filename, out_name)

m = 28
d = floor(m/2)

load('exp/det_net.mat')

% 1. Open the new unlabeled image
image = single(imread(filename));
f = fopen(out_name,'a+');
imsize = size(image);
%imshow(filename);
%hold on;

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

[y, x] = nonmaxsuppts(prob_map(:,:), 3, 0);

im_x = x + 14;
im_y = y + 14;

startfrom = 427;
for i = startfrom:numel(im_x) %n_centers
    figure;
    imshow(image(im_y(i)-14:im_y(i)+13,im_x(i)-14:im_x(i)+13),[-127,127],'InitialMagnification',200);
    hold on;
    plot(14,14,'or');
    a = input('Is this a cell at the center (1/0)? ','s')
    if strcmpi(a,'1')
        fprintf(f,'%.2f,%.2f,1\n',im_x(i),im_y(i));
    elseif strcmpi(a,'0')
        fprintf(f,'%.2f,%.2f,0\n',im_x(i),im_y(i));
    end
    fprintf('%d out of %d\n',i,numel(im_x));
    close;
end
