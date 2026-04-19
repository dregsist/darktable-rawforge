function rgb_dem = demosaick(rgb, pattern)

% guided filter epsilon
eps = 1e-10;
% guidede filter window size
h = 5;
v = 5;

% mosaic and mask
[mosaic mask] = mosaic_bayer(rgb, pattern);

% green interpolation
green = green_interpolation(mosaic, mask, pattern, eps);

% red and blue interpolation
red  =  red_interpolation(green, mosaic, mask, h, v, eps);
blue = blue_interpolation(green, mosaic, mask, h, v, eps);

% result image
rgb_dem = zeros(size(rgb));
rgb_dem(:,:,1) = red;
rgb_dem(:,:,2) = green;
rgb_dem(:,:,3) = blue;

end

