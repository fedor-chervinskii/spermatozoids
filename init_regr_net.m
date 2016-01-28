function net = init_regr_net

rng('default');
rng(0) ;

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
