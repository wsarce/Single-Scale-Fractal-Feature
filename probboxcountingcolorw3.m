
% Author: Mihai Ivanovici, MIV Research laboratory http://miv.unitbv.ro
% Affiliation: Department of Electronics and Computers, Transilvania University of Brasov, Romania
%
% Date: July 2008
% Last update: November 2019 % May 2017 % September 2016 % June 2010
%
% This Matlab function implements the extension to the color domain of the Voss probabilistic box-counting 
% algorithm for the estimation of the color fractal dimension for a given color image.
% Input parameters: the name of the file containing the colour image
% Outpur parameters: the mean estimated color fractal dimension and variance
% The run of this Matlab script may take up to several minutes for a 500 x 500 image.
%
% You may use the code as you wish for scientific purposes, as long as you acknowledge our work by referencing the following article: 
% M. Ivanovici, N. Richard, "The Colour Fractal Dimension of Colour Fractal Images", IEEE Transactions on Image Processing, January 2011
% https://doi.org/10.1109/TIP.2010.2059032

function [mcfd, vcfd] = probboxcountingcolorw3( imgfilename )
    LMAX = 41; 			% the maximum size of the hyper-cubes
	deltas = 3 : 2 : LMAX; 	% the sizes of the hyper-cubes (delta)
	mmax = power( LMAX, 2 );	% the maximum number m of pixels within a box (hyper-cube)

	P = zeros( mmax, length( deltas ) );	% the probability matrix P[m][delta]
	N = zeros( 1, length( deltas ) );		% the number of boxes N(delta)
	
	img = imread( imgfilename , 'png');
	[ n, m, o ] = size( img );
    % Correct for single channel images
    if o < 3
        im = zeros(n, m, 3, 'uint8');
        im(:,:,1) = img;
        im(:,:,2) = img;
        im(:,:,3) = img;
        img = im;
    end
    
    tic
	skip = floor( LMAX / 2 ) + 1;
	for i = skip : n - skip
	       for j = skip : m - skip
	       		for k = 1 : length( deltas )
				cube_size = deltas( k );
				
				%count how many pixels fall inside the hyper-cube
				cpix = img( i, j, 1 : 3 );
				cs = floor( cube_size / 2 );
				region = img( ( i - cs ) : ( i + cs ), ( j - cs ) : ( j + cs ), 1 : 3 );
				npix = length( find( ...	
						region( :, :, 1 ) >= cpix( 1 ) - cs & region( :, :, 1 ) <= cpix( 1 ) + cs &	...
						region( :, :, 2 ) >= cpix( 2 ) - cs & region( :, :, 2 ) <= cpix( 2 ) + cs &	...
						region( :, :, 3 ) >= cpix( 3 ) - cs & region( :, :, 3 ) <= cpix( 3 ) + cs ...
					) );

				P( npix, k ) = P( npix, k ) + 1;
			end
	       end
	end

	%normalize the probability matrix
	P = P / ( (n - LMAX) * (m - LMAX) );
	
	%compute the statistical moments
	for i = 1 : length( deltas )
		for m = 1 : mmax;
			N( i ) = N( i ) + (1.0 / m )* P( m, i );
		end
    end

    toc
    
	%handle = figure;
	%plot( deltas, N, 'b*-', 'LineWidth', 2 );
	%set( gca, 'FontSize', 14 );
	%xlabel( '\delta' );
	%ylabel( 'N(\delta)' );
	%grid on;
	
    %apply the third weighting function (see article)
    w = deltas.^2;
    
    wdeltas = [];
	wN = [];
	for i = 1 : length(w)
		if w(i) ~= 0
			wdeltas = [wdeltas ones(1,w(i)) * deltas(i)];
			wN = [wN ones(1,w(i)) * N(i)];
		end
	end
    
    %estimate the color fractal dimension D, using the "robust fit" approach with its 9 methods
	methods = { 'ols', ...
       		    'andrews', ...
	     	    'bisquare', ...
                'cauchy', ...
                'fair', ...
                'huber', ...
                'logistic', ...
                'talwar', ...
                'welsch' };
    lines = { 'bo-', 'gx-', 'r*-', 'cs-', 'md-', 'yv-', 'k^-', 'bp--', 'kh--' };

	lndeltas = log( wdeltas );
    lnN = -log( wN );

	for i = 1 : length( methods )
		[res, stat] = robustfit( lndeltas, lnN, char( methods(i) ) );
		cfds(i,:) = res;
    end
    
    %handle = figure;
	%set( gca, 'FontSize', 14 );
	%plot( lndeltas, lnN, 'b*-', 'LineWidth', 2 );
    %mess(1,:) = cellstr( 'N(\delta)' );
	%hold on;
	%for i = 1 : length( methods )
	%	plot( lndeltas, cfds(i,1) + cfds(i,2)*lndeltas, char( lines(i) ) ); 
	%	mess(i+1,:) = cellstr( sprintf( '%4.3f %s', cfds(i,2), char( methods(i) ) ) ); 
	%end
	%grid on;
	%legend( mess , 'Location', 'SouthEast' );
	%xlabel( 'ln(\delta)' );
	%ylabel( '-ln[N(\delta)]' );
    
    %filename = regexprep( imgfilename, '\.png', '' );
    %saveas( gcf, [filename, '_CFDw3.fig'], 'fig' );
	%print( handle, '-dpng', [filename, '_CFDw3.png' ] );
    
    mcfd = mean( cfds( :, 2 ) )
    vcfd = var( cfds( :, 2 ) )

    %img2 = insertText( img, [20 20], num2str( mcfd,'%.2f'), 'FontSize', 18, 'BoxColor', 'white', 'BoxOpacity', 0, 'TextColor', 'white' );
    %figure;
    %imshow( img2 );


