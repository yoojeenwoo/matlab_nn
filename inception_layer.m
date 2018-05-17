function [fm_out] = inception_layer(fm_in, branch1, branch2, branch3, branch4, EPSILON)

fm_out_1 = caffe_conv(fm_in, branch1.weights./reshape(branch1.batchvar + EPSILON, 1, 1, 1, []), 1, 0);
fm_out_1 = scale_add_bias(fm_out_1, -branch1.batchmean./(branch1.batchvar + EPSILON) + ...
branch1.batchbeta);
fm_out_1 = caffe_relu(fm_out_1);


fm_out_2 = caffe_conv(fm_in, branch2.weights_a./reshape(branch2.batchvar_a + EPSILON, 1, 1, 1, []), 1, 0);
fm_out_2= scale_add_bias(fm_out_2, -branch2.batchmean_a./(branch2.batchvar_a + EPSILON) + ...
branch2.batchbeta_a);
fm_out_2= caffe_relu(fm_out_2);
fm_out_2 = caffe_conv(fm_out_2, branch2.weights_b./reshape(branch2.batchvar_b + EPSILON, 1, 1, 1, []), 1, 1);
fm_out_2 = scale_add_bias(fm_out_2, -branch2.batchmean_b./(branch2.batchvar_b + EPSILON) + ...
branch2.batchbeta_b);
fm_out_2 = caffe_relu(fm_out_2);

fm_out_3 = caffe_conv(fm_in, branch3.weights_a./reshape(branch3.batchvar_a + EPSILON, 1, 1, 1, []), 1, 0);
fm_out_3= scale_add_bias(fm_out_3, -branch3.batchmean_a./(branch3.batchvar_a + EPSILON) + ...
branch3.batchbeta_a);
fm_out_3= caffe_relu(fm_out_3);
fm_out_3 = caffe_conv(fm_out_3, branch3.weights_b./reshape(branch3.batchvar_b + EPSILON, 1, 1, 1, []), 1, 1);
fm_out_3 = scale_add_bias(fm_out_3, -branch3.batchmean_b./(branch3.batchvar_b + EPSILON) + ...
branch3.batchbeta_b);
fm_out_3 = caffe_relu(fm_out_3);

fm_out_4 = caffe_pool(fm_in, 3, 1, 1);
fm_out_4 = caffe_conv(fm_out_4, branch4.weights./reshape(branch4.batchvar + EPSILON, 1, 1, 1, []), 1, 0);
fm_out_4 = scale_add_bias(fm_out_4, -branch4.batchmean./(branch4.batchvar + EPSILON) + ...
branch4.batchbeta);
fm_out_4 = caffe_relu(fm_out_4);

fm_out = cat(3, fm_out_1, fm_out_2, fm_out_3, fm_out_4);

end