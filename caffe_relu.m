function [fm_in] = caffe_relu(fm_in)
fm_in(fm_in<0) = 0.125*fm_in(fm_in<0);%leaky Relu
end