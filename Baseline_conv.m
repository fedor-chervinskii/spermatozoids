function [net,info,dataset] = Baseline_conv

m = 28

load('exp/orient_dataset.mat')

num_orient_classes = 36;
num_classes = num_orient_classes + 1;

dataset.num_classes = num_classes;
dataset.imdb.images.labels = ...
    round(dataset.imdb.images.angles./(360/num_orient_classes) + 0.5);

n = size(dataset.imdb.images.angles,2);
dataset.imdb.images.set = [ones(1,n - 5000) 3*ones(1, 5000)];
dataset.imdb.meta.sets = {'train', 'val', 'test'};
dataset.imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),...
    0:num_orient_classes,'uniformoutput',false);

[thispath,~,~] = fileparts(mfilename('fullpath'));
expDir = [thispath '/exp'];
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
  MateFullLayer(f*randn(num_classes,500, 'single'), zeros(num_classes,1, 'single'),... 
                'weightDecay', [0.005 0.005], 'name','prediction')
  MateSoftmaxLossLayer('name','loss',...
                'takes',{'prediction','input:2'})
  MateSoftmaxLayer('name','softmax','takes','prediction','skipBackward',1)
  MateMultilabelErrorLayer('name','error',...
                'takes',{'prediction','input:2'})
  } );

%subtract mean
dataset.imdb.images.data = bsxfun(@minus, dataset.imdb.images.data,122) ;
          
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
     'numEpochs',30, 'continue', true, 'expDir', expDir,...
     'learningRate', 0.001,'monitor', {'loss','error'},...
     'showLayers', 'conv1') ;

baseline_net = net;
save('exp/baseline_net.mat', 'baseline_net');
 
%----------------------------------------------------------%

function [x, eoe, dataset] = getBatch(istrain, batchNo, dataset)
eoe=false;
batchStart = batchNo*dataset.batchSize+1;
batchEnd = (batchNo+1)*dataset.batchSize;

num_classes = dataset.num_classes;

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
x{2} = zeros([num_classes numel(batch)],'single');
x{2}(sub2ind(size(x{2}),labels(:) + 1,(1:numel(batch))')) = single(1);

