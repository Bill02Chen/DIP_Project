import numpy as np
import cv2
from scipy.ndimage.filters import convolve
from matplotlib import pyplot as plt
import sys, os

from transform import transform
from countdark import count_dark_pixels

def rgb_to_grayscale(image):
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

def gaussian_blur(image, kernel_size=5, sigma=1.4):
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = np.outer(kernel, kernel.transpose())
    return convolve(image, kernel)

def sobel_filters(image):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    Ix = convolve(image, Kx)
    Iy = convolve(image, Ky)
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    return G, theta

def non_maximum_suppression(image, D):
    M, N = image.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255
                # angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = image[i, j+1]
                    r = image[i, j-1]
                # angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = image[i+1, j-1]
                    r = image[i-1, j+1]
                # angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = image[i+1, j]
                    r = image[i-1, j]
                # angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = image[i-1, j-1]
                    r = image[i+1, j+1]

                if (image[i,j] >= q) and (image[i,j] >= r):
                    Z[i,j] = image[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z

def threshold(image, lowThresholdRatio=0.02, highThresholdRatio=0.1):
    highThreshold = image.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    
    M, N = image.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(image >= highThreshold)
    zeros_i, zeros_j = np.where(image < lowThreshold)
    
    weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return res, weak, strong

def hysteresis(image, weak, strong=255):
    M, N = image.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (image[i,j] == weak):
                try:
                    if ((image[i+1, j-1] == strong) or (image[i+1, j] == strong) or (image[i+1, j+1] == strong)
                        or (image[i, j-1] == strong) or (image[i, j+1] == strong)
                        or (image[i-1, j-1] == strong) or (image[i-1, j] == strong) or (image[i-1, j+1] == strong)):
                        image[i, j] = strong
                    else:
                        image[i, j] = 0
                except IndexError as e:
                    pass
    return image

def canny_edge_detector(image, low_threshold=0.5, high_threshold=0.1):
    # Step 1: Convert to grayscale
    gray_image = rgb_to_grayscale(image)
    
    # Step 2: Apply Gaussian blur
    blurred_image = gaussian_blur(gray_image)
    # blurred_image = gray_image
    
    # Step 3: Get gradient intensity and direction
    gradient_magnitude, gradient_direction = sobel_filters(blurred_image)
    
    # Step 4: Non-maximum suppression
    non_max_img = non_maximum_suppression(gradient_magnitude, gradient_direction)
    
    # Step 5: Apply double threshold
    threshold_img, weak, strong = threshold(non_max_img, low_threshold, high_threshold)
    
    # Step 6: Edge tracking by hysteresis
    edge_img = hysteresis(threshold_img, weak, strong)
    
    return edge_img


    
def adaptive_moph(image, grid_size=200):
    # Read the image
    count2 = 0
    count1 = 0
    count3 = 0
    height, width = image.shape

    # Calculate the size of each subimage
    subimage_width = width // grid_size
    subimage_height = height // grid_size

    # Create a copy of the original image to place modified subimages
    processed_image = np.zeros_like(image)

    # Loop through grid and process subimages
    for i in range(grid_size):
        for j in range(grid_size):
            left = i * subimage_width
            upper = j * subimage_height

            # Ensure the last subimage in each row/column fits exactly
            right = (i + 1) * subimage_width if i < grid_size - 1 else width
            lower = (j + 1) * subimage_height if j < grid_size - 1 else height

            # Extract subimage
            subimage = image[upper:lower, left:right]

            # Check if more than half of the pixels are non-zero
            non_zero_count = cv2.countNonZero(subimage)
            total_pixels = subimage.shape[0] * subimage.shape[1]
            
            if non_zero_count > total_pixels / 10:
                se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
                count2 +=1
            else:
                se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                count3 +=1
            modified_subimage = cv2.dilate(subimage, se)
                
            processed_image[upper:lower, left:right] = modified_subimage
    # print(count3, count2, count1)
    return processed_image

def adative_merge(image, ref, edge, thresh = 25, size = 3):
    cols, rows, _ = edge.shape
    output = np.zeros_like(image)
    count2 = 0
    count3 = 0
    subimage_width = rows// size
    subimage_height = cols // size
    for i in range(subimage_width):
        for j in range(subimage_height):
            left = i * size
            upper = j * size

            right = (i + 1) * size if i < subimage_width - 1 else rows
            lower = (j + 1) * size if j < subimage_height - 1 else cols

            # Extract subimage
            subimage = image[upper:lower, left:right, :]
            subedge = edge[upper:lower, left:right, :]
            subref = ref[upper:lower, left:right, :]
            dark = count_dark_pixels(subref, thresh)

            if dark:
                subedge = cv2.subtract(subedge, 230)
                suboutput = cv2.add(subimage, subedge)
                count2 +=1
            else:
                suboutput = cv2.subtract(subimage, subedge)
                count3 +=1
                
            output[upper:lower, left:right, :] = suboutput
    # print(count2, count3)
    output = output.astype(np.uint8)
    return output

def gamma_transform(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    
    # Apply the gamma correction using the lookup table
    return cv2.LUT(image, table)




def main(path):
    # Check if the path exists
    if not os.path.exists(path):
        print(f"Error: The file at {path} does not exist.")
        return
    
    image_path = 'Smooth.jpg'  
# Replace with the actual path to your image
    # image_origin = '/Users/bill/Desktop/test/night.jpg'
    image_origin = path
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    origin = cv2.imread(image_origin)
    origin = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
    dark = count_dark_pixels(origin)

# Apply Canny edge detector
    t_image = transform(image, origin)
    if not dark:
        edges = canny_edge_detector(image)
    else: 
        edges = canny_edge_detector(image, 0.5, 0.04)
    edges = edges.astype(np.uint8)
    edges_m = adaptive_moph(edges)
    edges = np.stack([edges, edges, edges], axis=2)
    o_edges = np.stack([edges_m, edges_m, edges_m], axis=2)

    cv2.imwrite('edge_map.jpg', 255-o_edges)
    cv2.imwrite('base.jpg', cv2.cvtColor(t_image,cv2.COLOR_BGR2RGB))
    output = adative_merge(t_image, origin, o_edges)
    gamma = 1.2
    if not dark:
        o_edges = cv2.subtract(o_edges, 130)
        output = cv2.subtract(t_image, o_edges)
        output = gamma_transform(output, gamma)
    else:
        output = adative_merge(t_image, t_image, o_edges, 25)
        output = gamma_transform(output, gamma)

# output = cv2.subtract(t_image, o_edges)
# Display the original and edge-detected images
# plt.figure(figsize=(10, 5))
# plt.subplot(2, 2, 1)
# plt.title('Original Image')
# plt.imshow(t_image)
# plt.subplot(2, 2, 2)
# plt.title('Canny Edges')
# plt.imshow(o_edges, cmap='gray')
# plt.subplot(2, 2, 3)
# plt.imshow(origin)
# plt.subplot(2, 2, 4)
# plt.imshow(output)
# plt.show()

    cv2.imwrite('output.png', cv2.cvtColor(output,cv2.COLOR_BGR2RGB))
    print('Finish processing')

if __name__ == "__main__":
    # Check if the path is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python test.py <path>")
    else:
        main(sys.argv[1])


