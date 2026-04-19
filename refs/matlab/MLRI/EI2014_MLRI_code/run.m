%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%   Implementation of 'Minimized-Laplacian Residual Interpolation for Color Image Demosaicking' 
% 
%   This code is available only for reserch purpose.
%   If you use this code for future publications,
%   please cite the following paper.
% 
%   Daisuke Kiku, Yusuke Monno, Masayuki Tanaka, and Masatoshi Okutomi,
%   'Minimized-Laplacian Residual Interpolation for Color Image Demosaicking',
%   IS&T/SPIE Electronic Imaging, Digital Photography X, 2014.
% 
%   Main funtion
%     rgb_dem = demosaick(rgb, pattern, sigma)
% 
%      Input
%       - rgb		: full RGB image
%       - pattern	: mosaic pattern 
%                       default : 'grbg'
%                       others  : 'rggb','gbrg','bggr'	
%       - sigma		: standard deviation of gaussian filter(default : 1.4)
%                       * For IMAX image dataset, 1 works well.
%                       * For Kodak image dataset, 1e8 works well.
% 
%      Output 
%       - rgb_dem   : result image
% 
%   Copyright (C) 2013 Daisuke Kiku. All rights reserved.
%   dkiku@ok.ctrl.titech.ac.jp 
% 
%   November 26, 2013.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all

% read image
rgb = imread('lena.tiff');

% cast to double
rgb = double(rgb);

% mosaic pattern
% G R ..
% B G ..
% : :   
pattern = 'grbg';

% demosaicking 
sigma = 1.4;
rgb_dem = demosaick(rgb, pattern, sigma);

% save image
imwrite(uint8(rgb_dem), 'lena_MLRI.tiff');

% show image
imshow(rgb_dem/255);

% calculate PSNR and CPSNR
psnr = impsnr(rgb, rgb_dem, 255, 10);
cpsnr = imcpsnr(rgb, rgb_dem, 255, 10);

% print PSNR and CPSNR
fprintf( sprintf( 'Red:%f\n',     psnr(1)     ) );
fprintf( sprintf( 'Green:%f\n',   psnr(2)     ) );
fprintf( sprintf( 'Blue:%f\n',    psnr(3)     ) );
fprintf( sprintf( 'CPSNR:%f\n',   cpsnr       ) );
