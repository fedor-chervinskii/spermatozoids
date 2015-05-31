function [pos_patches, neg_patches] = CollectPatches(m, plot)

%function collects patches with spermatozoids at the center
% and patches with background

fileList = dir('labels/*.csv');
fileList = {fileList.name}';

%patches m x m pixels

d = (m-1)/2;

fprintf('collecting patches...\n');

bias = 10;

k = (512 - 2*d)/bias;

pos_patches = zeros(5000,m*m + 1);
neg_patches = zeros(5000,m*m + 1);
pos_counter = 1;
neg_counter = 1;

for i = 1:numel(fileList)
    name = fileList{i}(1:end-4);
    fprintf('processing %s\n',name);
    image = imread(['images/' name '.tif']);
    f = fopen(['labels/' name '.csv'],'r');
    centers = zeros(5000,2);
    tline = fgetl(f);
    cur_counter = 1;
    while ischar(tline)
        A = sscanf(tline, '%f,%f;%f,%f');
        Xc = round(A(3));
        Yc = round(A(1));
        angle = atan2d(A(4)-A(3),A(2)-A(1))+180;
        centers(cur_counter,:) = [Xc, Yc];
        if (Xc+d <= 512) && (Xc-d >= 1) && (Yc+d <= 512) && (Yc-d >= 1)
            pos_patches(pos_counter,1:end-1) = reshape(image(Xc-d:Xc+d,Yc-d:Yc+d),1,m*m);
            if orient
                pos_patches(pos_counter,end) = angle;
            else
                pos_patches(pos_counter,end) = 1;
            end
            pos_counter = pos_counter + 1;
            cur_counter = cur_counter + 1;
        end
        tline = fgetl(f);   
    end
    %collect negative patches
    
    centersKDT = KDTreeSearcher(centers);
    
    for Xc = round(linspace(1+d,512-d,k))
        for Yc = round(linspace(1+d,512-d,k))
            idx = knnsearch(centersKDT, [Xc, Yc]);
            if (centers(idx, 1) - Xc >= 4) && (centers(idx, 2) - Yc >= 4)
                neg_patches(neg_counter,1:end-1) = reshape(image(Xc-d:Xc+d,Yc-d:Yc+d),1,m*m);
                neg_patches(neg_counter,end) = -1;
                neg_counter = neg_counter + 1;
            end        
        end
    end
end

pos_patches = pos_patches(1:pos_counter-1,:);
neg_patches = neg_patches(1:neg_counter-1,:);
pos_indices = randsample(pos_counter-1,50);
neg_indices = randsample(neg_counter-1,50);
if plot
    for i = 1:50
        subplot(10,10,i), imshow(reshape(pos_patches(pos_indices(i),1:end-1),m,m),[50, 150])
        subplot(10,10,i+50), imshow(reshape(neg_patches(neg_indices(i),1:end-1),m,m),[50, 150])
    end
end