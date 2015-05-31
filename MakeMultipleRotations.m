function imgsrot = MakeMultipleRotations(imgs, num_rotations, num_orient_classes);

width = 29;
height = 29;

labels = imgs(end,:);
imgs = imgs(1:end-1,:);
% input  -  array (num_pixels, num_images)
% output  -  array (num_pixels*Num_of_Rotations, num_images)

num_images = size(imgs , 2);

imgsrot = zeros(width, height, num_rotations*num_images);
labelsrot = zeros(1, num_rotations*num_images);

for i = 1:num_rotations
    angle = 360/num_rotations * (i-1);
    for img = 1:num_images
        imgsrot(:,:,(img-1)*num_rotations + i) = imrotate(reshape(imgs(:,img),width,height),angle,'bilinear','crop');
        orig_angle = labels(img);
        new_angle = rem(orig_angle+angle,360);
        labelsrot(1,(img-1)*num_rotations + i) = new_angle*(orig_angle >= 0) - (orig_angle < 0);
    end
end

imgsrot  = cat(1,reshape(imgsrot, [], num_images*num_rotations),labelsrot);