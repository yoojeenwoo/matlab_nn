function [input] = batch_normalize(input, mean, variance)
    s = size(input);
%     display(s);
    for i = 1:s(3)
        input(:,:,i) = (input(:,:,i)-mean(i))/(sqrt(variance(i))+0.000001);
    end
end