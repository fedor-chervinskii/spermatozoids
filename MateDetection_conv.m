function [class_net, info, dataset] = MateDetection_conv

m = 28;
num_rotations = 10;

if 1
    data = CollectPatches('labels/centers/train/','images/train/', m, num_rotations, false, false);
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

    save('exp/dataset.mat', 'dataset');
else
    load('exp/dataset.mat')
end
n = size(dataset.imdb.images.labels,2)

dataset.imdb.images.set = [ones(1,n - 1000) 3*ones(1, 1000)] ;
dataset.imdb.meta.sets = {'train', 'val', 'test'} ;
dataset.imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:1,'uniformoutput',false) ;

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
  MateConvLayer(f*randn(1,1,500,1, 'single'), ...
                zeros(1, 1, 'single'), ...
                'weightDecay', [0.005 0.005], 'name','prediction')
  MateLogisticLossLayer('name','loss',...
                'takes',{'prediction','input:2'})
} );
 
          
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
     'numEpochs',30, 'continue', false, 'expDir', expDir,...
     'learningRate', 0.005,'monitor', {'loss'},...
     'showLayers', 'conv1', 'onEpochEnd', @onEpochEnd) ;

det_net = net;
save('exp/det_net.mat', 'det_net');

%----------------------------------------------------------%

function [x, eoe, dataset] = getBatch(istrain, batchNo, dataset)
eoe=false;
batchStart = batchNo*dataset.batchSize+1;
batchEnd = (batchNo+1)*dataset.batchSize;

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
labels = dataset.imdb.images.labels(batch) ;
x{2} = zeros([1 1 1 numel(batch)],'single');
x{2}(1,1,1,:) = labels(:)*2 - 1;

function [net,dataset,learningRate] = onEpochEnd(net,dataset,learningRate)
1;
%learningRate = 0.9 * learningRate;