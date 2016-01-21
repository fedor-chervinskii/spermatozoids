% --------------------------------------------------------------------
% Prepare and save the imdb structure

% following parameters are to be changed

imdbPath = 'exp/imdb_det';
images_dir = 'images/train/';
labels_dir = 'labels/centers/train/';

m = 28;
num_rotations = 10;
%[X, Y] = meshgrid(-2:2:2,-2:2:2);  % jittering
%biases = [X(:) Y(:)];
biases = [0 0]; 

getAngle = false;
firstZero = false;
    
% --------------------------------------------------------------------

[ patches, angles ] = CollectPatches(labels_dir, images_dir, m, ...
    num_rotations, biases, getAngle, firstZero);
    
n = size(patches,1);

fprintf('number of samples after augmentation %d\n',n);

imdb = struct;
imdb.images.angles = angles;
imdb.images.labels = round(angles./360 + 0.5);
imdb.images.data = reshape(patches',m,m,1,[]) ;

imdb.images.data = imdb.images.data - 122;

imdb.images.data = single(imdb.images.data);
test_size = 5000;

imdb.images.set = [ones(1,n - test_size) 3*ones(1, test_size)] ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:1,'uniformoutput',false) ;

save(imdbPath, '-struct', 'imdb', '-v7.3') ;