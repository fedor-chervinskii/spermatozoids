function [net,info,dataset] = MateDetection_conv
%demonstrates Mate on MNIST
%example is derived from the analogous MatConvNet example

m = 29;

data = CollectPatchesWithRotations(m, 24);
data = data(randperm(size(data, 1)), :)';
n = size(data,2);
labels = data(end,:);
data = data(1:end-1,:);
fprintf('number of samples after augmentation %d\n',n);

dataset = struct;
dataset.imdb.images.data = single(reshape(data,m,m,1,[])) ;

% averaging the data
%mean_image = mean(dataset.imdb.images.data,4);
%dataset.imdb.images.data = bsxfun(@minus, dataset.imdb.images.data, mean_image);
dataset.imdb.images.data = dataset.imdb.images.data - 122;
%save('exp/train_params.mat', 'mean_image');
        
for i = 1:49
    j = randi(n);
    subplot(7,7,i), imshow(reshape(dataset.imdb.images.data(:,:,:,j),m,m), [-120,120]), title(labels(j));
end

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
  MateConvLayer(f*randn(1,1,500,2, 'single'), zeros(1, 2, 'single'),... 
                'weightDecay', [0.005 0.005], 'name','prediction')
  MateSoftmaxLossLayer('name','loss',...
                'takes',{'prediction','input:2'})
  MateSoftmaxLayer('name','softmax','takes','prediction','skipBackward',1)
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
     'numEpochs',20, 'continue', false, 'expDir', expDir,...
     'learningRate', 0.001,'monitor', {'loss','error'},...
     'showLayers', 'conv1', 'onEpochEnd', @onEpochEnd) ;

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
x{2} = zeros([1 1 2 numel(batch)],'single');
x{2}(sub2ind(size(x{2}), ones(numel(batch),1), ones(numel(batch),1), labels(:) + 1,(1:numel(batch))')) = single(1);

function [net,dataset,learningRate] = onEpochEnd(net,dataset,learningRate)

learningRate = 0.9 * learningRate;