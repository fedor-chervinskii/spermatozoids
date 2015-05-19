function [net,info,dataset] = MateDetection
%demonstrates Mate on MNIST
%example is derived from the analogous MatConvNet example

m = 29

[pos_patches, neg_patches] = CollectPatches(m, false);
n_pos = size(pos_patches,1)
n_neg = size(neg_patches,1)

n = n_pos + n_neg;

data = cat(1, pos_patches, neg_patches);
num_rotations = 10;
n = n * num_rotations
data = MakeMultipleRotations( data' , num_rotations );
data = datasample(data, n, 2);
labels = data(end,:);
data = data(1:end-1,:);

dataset = struct;
dataset.imdb.images.data = single(reshape(data,m,m,1,[])) ;
dataset.imdb.images.labels = labels ;
dataset.imdb.images.set = [ones(1,n - 500) 3*ones(1, 500)] ;
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
  MateFlattenLayer
  MateFullLayer(f*randn(2,500, 'single'), zeros(2,1, 'single'),... 
                'weightDecay', [0.005 0.005], 'name','prediction')
  MateSoftmaxLossLayer('name','loss',...
                'takes',{'prediction','input:2'})
  MateMultilabelErrorLayer('name','error',...
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
     'numEpochs',10, 'continue', false, 'expDir', expDir,...
     'learningRate', 0.001,'monitor', {'loss','error'},...
     'showLayers', 'conv1') ;

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
x{2} = zeros([2 numel(batch)],'single');
x{2}(sub2ind(size(x{2}),labels(:) + 1,(1:numel(batch))')) = single(1);

