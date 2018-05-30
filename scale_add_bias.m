function [input] = scale_add_bias(input,bias,scale)
%     s = size(bias);
    for i = 1:length(bias)
        % input(:,:,i) = input(:,:,i)*scale(i);
        input(:,:,i) = input(:,:,i)+bias(i);
    end
end