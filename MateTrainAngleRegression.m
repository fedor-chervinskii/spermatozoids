function [regr_net,info,dataset] = MateAngleRegression()
%example is derived from the analogous MatConvNet example

collect_dataset = true;

if collect_dataset
    m = 28;
    num_rotations = 10;

    [X, Y] = meshgrid(-2:1:2,-2:1:2);
    biases = [X(:) Y(:)];

    getAngle = true;
    firstZero = false;
    data = CollectPatches('labels/orientations/train/','images/train/', m, ...
                           num_rotations, biases, getAngle, firstZero);
    data = data(randperm(size(data, 1)), :)';
    n = size(data,2);

    angles = data(end,:);
    data = data(1:end-1,:);

    fprintf('number of samples after augmentation %d\n',n);

    dataset = struct;
    dataset.imdb.images.angles = angles;
    dataset.imdb.images.labels = round(angles./360 + 0.5);
    dataset.imdb.images.data = single(reshape(data,m,m,1,[])) ;

    dataset.imdb.images.data = dataset.imdb.images.data - 122;

    save('exp/orient_dataset.mat', 'dataset');
else
    load('exp/orient_dataset.mat')
end

labels = dataset.imdb.images.labels;
dataset.imdb.images.data = dataset.imdb.images.data(:,:,:,labels > 0);
dataset.imdb.images.angles = dataset.imdb.images.angles(labels > 0);

n = size(dataset.imdb.images.angles,2);

dataset.imdb.images.set = [ones(1,n - 500) 3*ones(1, 500)] ;
dataset.imdb.meta.sets = {'train', 'val', 'test'} ;

[thispath,~,~] = fileparts(mfilename('fullpath'));
expDir = [thispath '/exp'] ;
useGpu = false;

% Define the network

f=1/100 ;

net = MateNet( {
  MateConvLayer(f*randn(5,5,1,20, 'single'), zeros(1, 20, 'single'), ...
                'stride', 1, 'pad', 0, 'name', 'conv1')
  MatePoolLayer('pool',[2 2], 'stride', 2, 'pad', 0)
  MateConvLayer(f*randn(5,5,20,50, 'single'), zeros(1, 50, 'single'), ...
                'stride', 1, 'pad', 0, 'weightDecay', [0.005 0.005])
  MatePoolLayer('pool',[2 2], 'stride', 2, 'pad', 0)  
  MateConvLayer(f*randn(4,4,50,500, 'single'), zeros(1, 500, 'single'), ...
                'stride', 1, 'pad', 0, 'weightDecay', [0.005 0.005])  
  MateReluLayer
  MateConvLayer(f*randn(1,1,500,2, 'single'), zeros(1, 2, 'single'),... 
                'weightDecay', [0.005 0.005])
  MateSqueezeLayer
  MateL2NormalizeLayer('name','prediction')
  MateL2LossLayer('name','loss',...
                'takes',{'prediction','input:2'})
  } );


%subtract mean
dataset.imdb.images.data = bsxfun(@minus, dataset.imdb.images.data,...
            mean(dataset.imdb.images.data,4)) ;
          
%move to GPU         
if useGpu
  net = net.move('gpu');
  dataset.imdb.images.data = gpuArray(dataset.imdb.images.data) ;
  dataset.imdb.images.labels = gpuArray(dataset.imdb.images.labels) ;
end

dataset.train = find(dataset.imdb.images.set == 1);
dataset.val = find(dataset.imdb.images.set == 3);
dataset.batchSize = 100;

[net,info,dataset] = net.trainNet(@getBatch, dataset,...
     'numEpochs',20, 'continue', false, 'expDir', expDir,...
     'learningRate', 0.001, 'monitor', {'loss'},...
     'onEpochEnd', @onEpochEnd) ;

regr_net = net;
save('exp/regr_net.mat', 'regr_net');

%----------------------------------------------------------%

function [x, eoe, dataset] = getBatch(istrain, batchNo, dataset)
eoe=false;
batchStart = (batchNo-1)*dataset.batchSize+1;
batchEnd = batchNo*dataset.batchSize;

if istrain
  if batchEnd >= numel(dataset.train)
    batchEnd = numel(dataset.train);
    eoe = true; %end of epoch
  end
  batch = dataset.train(batchStart:batchEnd);
else
  if batchEnd >= numel(dataset.val)
    batchEnd = numel(dataset.val);
    eoe = true; %end of epoch
  end
  batch = dataset.val(batchStart:batchEnd); 
end

x{1} = dataset.imdb.images.data(:,:,:,batch) ;
labels = mod(dataset.imdb.images.angles(batch),180);
x{2} = zeros([2 numel(batch)],'single');
x{2}(1,:) = sind(2*labels);
x{2}(2,:) = cosd(2*labels);

function [net,dataset,learningRate] = onEpochEnd(net,dataset,learningRate)
1;
%learningRate = 0.9 * learningRate;