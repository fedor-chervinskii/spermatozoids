% --------------------------------------------------------------------
% Collect patches from raw microscopic images
% Prepare and save the imdb structure

% following parameters are to be changed

imdbPath = 'exp/imdb_bi';
images_dir = 'images/train/';
labels_dir = 'labels/orientations/train/';

rng(0);

m = 28; % patch size, changing this number will influence NN's structure
num_rotations = 2;
[X, Y] = meshgrid(-4:2:4,-4:2:4);  % jittering
biases = [X(:) Y(:)];
%biases = [0 0]; 

getAngle = true;     % collect with labeled angles
firstZero = true;   % collect with certain orientations
neg = false;         % collect negatives
    
% --------------------------------------------------------------------

[ patches, angles ] = collect_patches(labels_dir, images_dir, m, ...
    num_rotations, biases, getAngle, firstZero, neg);
    
n = size(patches,1);

fprintf('number of samples after augmentation %d\n',n);

imdb = struct;
imdb.images.angles = angles;
imdb.images.labels = round(angles./360 + 0.5);
imdb.images.data = reshape(patches',m,m,1,[]) ;

imdb.images.data = preprocess(imdb.images.data);
test_size = 5000;

imdb.images.set = [ones(1,n - test_size) 3*ones(1, test_size)] ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:1,'uniformoutput',false) ;

save(imdbPath, '-struct', 'imdb', '-v7.3') ;