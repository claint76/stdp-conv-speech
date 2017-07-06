function filter = DoG(sz,sigma1,sigma2)
x = repmat([1:sz],sz,1);
y = x';
d2 = (x-sz/2-.5).^2 + (y-sz/2-.5).^2;

filter = 1/sqrt(2*pi) * ( 1/sigma1 * exp(-d2/2/(sigma1^2)) - 1/sigma2 * exp(-d2/2/(sigma2^2)) );

% sum of weight must be 0
filter = filter - mean(filter(:));

% max must be 1
filter = filter / max(filter(:));

