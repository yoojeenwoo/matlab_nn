function [fm_out] = caffe_avg_pool(fm_in, ker_size, stride)
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
P = 0; % pad = 0;
S = stride;
W_O = (W-F+2*P)/S + 1;
N_IN = fm_in_size(3);
N_OUT = N_IN;
fm_out_size = [W_O,W_O,N_OUT];

% preprocess fm_in
fm_in_3d = zeros(W_O*W_O, F*F,N_IN);
for i=1:W_O
  for j=1:W_O
    x = (i-1)*S;
    y = (j-1)*S;
    fm_in_3d(i+(j-1)*W_O,:,:) = reshape(fm_in(x+1:x+F, y+1:y+F, :), [1,F*F,N_IN]);
  end
end
fm_out_2d = squeeze(mean(fm_in_3d, 2));
fm_out = reshape(fm_out_2d,  fm_out_size);

end