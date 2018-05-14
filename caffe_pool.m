function [fm_out] = caffe_pool(fm_in, ker_size, stride, pad)
coder.extrinsic('exist');
%% test data
if nargin < 3
%   error('Not enough input para!');
  fm_in  = reshape(1:32,[4,4,2]);
  ker_size = 2;
  stride = 2;
end

%% function starts
% data size
fm_in_size = size(fm_in);
W = fm_in_size(1); assert(W==fm_in_size(2),'square fm_in');
F = ker_size;
P = pad; % pad = 0;
S = stride;
% W_O = (W-F+2*P)/S + 1;
W_O = floor((W-F+2*P)/S) + 1;
N_IN = fm_in_size(3);
N_OUT = N_IN;
fm_pad_size = [W+2*P, W+2*P, N_IN];
fm_out_size = [W_O,W_O,N_OUT];

% padding
fm_pad = zeros(fm_pad_size);
% fm_pad = fi(fm_pad, 1, fm_in.WordLength, fm_in.FractionLength,  'SumMode', fm_in.SumMode, 'SumWordLength',fm_in.SumWordLength);
fm_pad(1+pad:end-pad, 1+pad:end-pad, :) = fm_in;

% preprocess fm_in
fm_in_3d = zeros(W_O*W_O, F*F,N_IN);
for i=1:W_O
  for j=1:W_O
    x = (i-1)*S;
    y = (j-1)*S;
%     fm_in_3d(i+(j-1)*W_O,:,:) = reshape(fm_in(x+1:x+F, y+1:y+F, :), [1,F*F,N_IN]);
    fm_in_3d(i+(j-1)*W_O,:,:) = reshape(fm_pad(x+1:x+F, y+1:y+F, :), [1,F*F,N_IN]);
  end
end
fm_out_2d = squeeze(max(fm_in_3d, [], 2));
fm_out = reshape(fm_out_2d,  fm_out_size);

end
