function [regr_net,info,dataset] = train_angle_regression()

expDir = fullfile('exp','regr_net');
dataDir = fullfile('exp');
imdbPath = fullfile(dataDir, 'imdb_orient.mat');

dataset = struct;

dataset.imdb = load(imdbPath);

dataset.train = find(dataset.imdb.images.set == 1);
dataset.val = find(dataset.imdb.images.set == 3);
dataset.batchSize = 100;

% Define the network

net = init_regr_net();

[net,info,dataset] = net.trainNet(@getBatch, dataset,...
     'numEpochs', 25, 'continue', true, 'expDir', expDir,...
     'learningRate', 0.0005, 'monitor', {'loss'},...
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

x{1} = single(dataset.imdb.images.data(:,:,:,batch)) ;
labels = mod(dataset.imdb.images.angles(batch),180);
x{2} = zeros([2 numel(batch)],'single');
x{2}(1,:) = sind(2*labels);
x{2}(2,:) = cosd(2*labels);

function [net,dataset,learningRate] = onEpochEnd(net,dataset,learningRate)
1;
%learningRate = 0.9 * learningRate;