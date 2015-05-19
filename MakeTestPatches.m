function [ patches ] = MakeTestPatches( m )
%MAKEFRESHTILES Summary of this function goes here
%   Detailed explanation goes here

d = (m-1)/2;
bias = 10;
k = (512 - 2*d)/bias;

num_patches = 10;
patches = zeros(num_patches,m*m);

image = imread('images/C001H001S0001000002_4.tif');
counter = 1;
for Xc = round(linspace(1+d,512-d,k))
    for Yc = round(linspace(1+d,512-d,k))
        patches(counter,1:end) = reshape(image(Xc-d:Xc+d,Yc-d:Yc+d),1,m*m);
        counter = counter + 1;
        if counter == num_patches
            break
        end
    end
    if counter == num_patches
        break
    end
end

patches = permute(reshape(patches, [10 29 29 1]), [2 3 4 1]);

end

