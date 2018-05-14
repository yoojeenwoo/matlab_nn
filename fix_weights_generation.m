%fix point weights generation
%weights preprocessing
mydir = 'D:\yyx_workspace\MATLAB_code\yolo2_nobatch\float_weights\';
DIS = dir([mydir,'*.mat']);
n = length(DIS);
for i = 1:n
    data = load([mydir,DIS(i).name]);
    eval([DIS(i).name(1:length(DIS(i).name)-4) ' = single(data.value);']);  
end
kernel_size = [3,3,3,1,3,3,1,3,3,1,3,1,3,3,1,3,1,3,3,3,1];
kernel_depth = [3,32,64,128,64,128,256,128,256,512,256,512,256,512,1024,512,1024,512,1024,1024,1024,70];
layer_name = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21'];
    
    for i = 1:21
%       eval(['conv' layer_name((i-1)*2+1:(i-1)*2+2) '_bias = single(fi_best(conv' layer_name((i-1)*2+1:(i-1)*2+2) '_bias,1,16));']);
      eval(['conv' layer_name((i-1)*2+1:(i-1)*2+2) '_bias = fi_best(conv' layer_name((i-1)*2+1:(i-1)*2+2) '_bias,1,16);']);
      save(['D:\yyx_workspace\MATLAB_code\yolo2_nobatch\fix_8_fix\conv' layer_name((i-1)*2+1:(i-1)*2+2) '_bias'], ['conv' layer_name((i-1)*2+1:(i-1)*2+2) '_bias']);
      eval(['conv' layer_name((i-1)*2+1:(i-1)*2+2) '_weights_reshape = single(reshape(conv' layer_name((i-1)*2+1:(i-1)*2+2) '_weights,' num2str(kernel_size(i)) ',' num2str(kernel_size(i)) ',' num2str(kernel_depth(i)) ',' num2str(kernel_depth(i+1)) '));']);
      eval(['conv' layer_name((i-1)*2+1:(i-1)*2+2) '_weights_reshape = permute(conv' layer_name((i-1)*2+1:(i-1)*2+2) '_weights_reshape,[2,1,3,4]);']);
%       eval(['conv' layer_name((i-1)*2+1:(i-1)*2+2) '_weights_reshape = single(fi_best(conv' layer_name((i-1)*2+1:(i-1)*2+2) '_weights_reshape,1,8));']);
      eval(['conv' layer_name((i-1)*2+1:(i-1)*2+2) '_weights_reshape = fi_best(conv' layer_name((i-1)*2+1:(i-1)*2+2) '_weights_reshape,1,8);']);
      save(['D:\yyx_workspace\MATLAB_code\yolo2_nobatch\fix_8_fix\conv' layer_name((i-1)*2+1:(i-1)*2+2) '_weights_reshape'], ['conv' layer_name((i-1)*2+1:(i-1)*2+2) '_weights_reshape']);

%       x(i) = eval(['conv' layer_name((i-1)*2+1:(i-1)*2+2) '_weights_reshape.FractionLength']);
     %       eval(['conv' layer_name((i-1)*2+1:(i-1)*2+2) '_weights_reshape = single(fi_best(conv' layer_name((i-1)*2+1:(i-1)*2+2) '_weights_reshape,1,8));']);
%       frac(i) = eval(['conv' layer_name((i-1)*2+1:(i-1)*2+2) '_weights_reshape. FractionLength']);
    end
  
