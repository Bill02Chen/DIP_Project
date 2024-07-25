function is_image_dark = count_dark_pixels(image, threshold)

    if nargin < 2
        threshold = 25;
    end
    
    % Convert the image to grayscale if it's not already
    if size(image, 3) == 3
        gray_image = rgb2gray(image);
    else
        gray_image = image;
    end
    
    % Count the number of dark pixels
    dark_pixels = gray_image < threshold;
    dark_pixels_count = nnz(dark_pixels);
    
    total_pixels = numel(gray_image);
    percentage_dark = (dark_pixels_count / total_pixels) * 100;
    
    % Return true if more than 50% of the pixels are dark
    is_image_dark = percentage_dark > 50;
end