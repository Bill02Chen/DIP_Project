%please modify input image path
path = '/Users/bill/Desktop/test/night.jpg';
Im = imread(path);
dark = count_dark_pixels(Im, 25);

if dark == 1
    lambda = 0.001;
else 
    lambda = 0.01;
end
S = L0Smoothing(Im, lambda);

imwrite(S, 'Smooth.jpg')







