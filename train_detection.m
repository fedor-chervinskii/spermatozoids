function [net, info] = train_detection(varargin)

expDir = fullfile('exp','det_net');
dataDir = fullfile('exp');
imdbPath = fullfile(dataDir, 'imdb_det_hard.mat');

dataset = struct;

dataset.imdb = load(imdbPath);

dataset.train = find(dataset.imdb.images.set == 1);
dataset.val = find(dataset.imdb.images.set == 3);
dataset.batchSize = 100;

net = init_det_net();

[net,info] = net.trainNet(@getBatch, dataset,...
     'numEpochs', 30, 'continue', true, 'expDir', expDir,...
     'learningRate', 0.001, 'monitor', {'loss','error'},...
     'showLayers', 'conv1') ;

det_net = net;
save('exp/det_net_hard.mat', 'det_net');

% --------------------------------------------------------------------
function [x, eoe, dataset] = getBatch(istrain, batchNo, dataset)
% --------------------------------------------------------------------
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

x{1} = single(dataset.imdb.images.data(:,:,:,batch)) ;
labels = dataset.imdb.images.labels(batch) ;
%x{2} = zeros([1 1 1 numel(batch)],'single');
%x{2}(1,1,1,:) = labels(:)*2 - 1;
x{2} = zeros([1 1 2 numel(batch)],'single');
x{2}(1,1,sub2ind([2 numel(batch)],labels(:) + 1,(1:numel(batch))')) = single(1);


% --------------------------------------------------------------------
