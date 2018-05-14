function [fm_out,x] = caffe_conv(fm_in, ker_in,stride, pad)
coder.extrinsic('exist');
%% test data
if nargin < 4
  error('Not enough input para!')
  fm_in  = reshape(1:18,[3,3,2]);
  ker_in = zeros(3,3,2,4);
  for i=1:size(ker_in,4)
    ker_in(:,:,:,i) = fm_in+i-1;
  end
  stride = 2;
  pad = 1;
end
%% input parameters
% assert(1==exist('stride', 'var'), 'stride undefined!');
if ~exist('pad','var')  pad=0; end


%% function starts
% data size
fm_in_size = size(fm_in);
ker_size = size(ker_in);
W = fm_in_size(1); assert(W==fm_in_size(2),'square fm_in');
% F = 3;   assert(F==ker_size(2),  'square kernel');
F = ker_size(1);
P = pad;
S = stride;
% W_O = (W-F+2*P)/S + 1;
W_O = floor((W-F+2*P)/S) + 1;
N_IN = fm_in_size(3);
N_OUT = ker_size(4);
fm_pad_size = [W+2*P, W+2*P, N_IN];
fm_out_size = [W_O,W_O,N_OUT];


% padding
fm_pad = zeros(fm_pad_size);
% fm_pad = fi(fm_pad, 1, fm_in.WordLength, fm_in.FractionLength,  'SumMode', fm_in.SumMode, 'SumWordLength',fm_in.SumWordLength);
fm_pad(1+pad:end-pad, 1+pad:end-pad, :) = fm_in;


% preprocess kernels
ker_2d = reshape(ker_in, [F*F*N_IN, N_OUT]);
% preprocess fm_in
fm_in_2d = single(zeros(W_O*W_O, F*F*N_IN));

coder.varsize('fm_in_2d');
for i=1:W_O
  for j=1:W_O
    fm_in_2d(i+(j-1)*W_O,:) = reshape(fm_pad((i-1)*S+1:(i-1)*S+F, (j-1)*S+1:(j-1)*S+F, :), [1,F*F*N_IN]);
  end
end

% fm_out_2d = fm_in_2d*ker_2d + repmat(bias',W_O*W_O,1);
fm_out_2d = fm_in_2d*ker_2d;

fm_out = reshape(fm_out_2d,  fm_out_size);

end