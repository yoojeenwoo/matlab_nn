function output = fi_best(input, sign, wordlen,b)
global fix_location;
global cnt;
   x = fi(input, sign, wordlen);
   
   frac = x.FractionLength;
%    display(frac);
   error = sum(sum(sum(sum((abs(single(x)-input)).^2))));
   error_best = error;
   frac_best = frac;
   output = x;
   for i = 1:8
      frac = frac+1;
      y = fi(input, sign, wordlen, frac);
      error = sum(sum(sum(sum((abs(single(y)-input)).^2))));
      if error < error_best
          error_best = error;
          frac_best  = frac;
          output = y;
      end
   end
   
   display(frac_best);
   fix_location(cnt) = frac_best;
   cnt = cnt+1;
end