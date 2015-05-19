function [ patches ] = MakeTestPatches( m,num_patches )

d = (m-1)/2;

patches = zeros(num_patches,m*m);

image = imread('images/C001H001S0001000002_4.tif');

for i = 1:num_patches
    Xc = randsample(1+d:512-d,1);
    Yc = randsample(1+d:512-d,1);
    patches(i,1:end) = reshape(image(Xc-d:Xc+d,Yc-d:Yc+d),1,m*m);
end

patches = permute(reshape(patches, [num_patches m m 1]), [2 3 4 1]);

end

