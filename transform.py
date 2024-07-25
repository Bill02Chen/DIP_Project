import numpy as np
import cv2
import matplotlib.pyplot as plt

def rgb_to_hsi(image):
    img = image.astype(np.float32) / 255.0
    R, G, B = cv2.split(img)
    I = (R + G + B) / 3.0
    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-6) * min_rgb)
    num = 0.5 * ((R - G) + (R - B))
    denom = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    theta = np.arccos(num / (denom + 1e-6))
    H = np.where(B <= G, theta, 2 * np.pi - theta)
    H = H / (2 * np.pi)
    HSI = cv2.merge([H, S, I])
    return HSI

def hsi_to_rgb(hsi):
    H, S, I = cv2.split(hsi)
    H = H * 2 * np.pi
    R, G, B = np.zeros_like(H), np.zeros_like(H), np.zeros_like(H)
    
    # RG Sector
    idx = (H < 2 * np.pi / 3)
    B[idx] = I[idx] * (1 - S[idx])
    R[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx]) / np.cos(np.pi / 3 - H[idx]))
    G[idx] = 3 * I[idx] - (R[idx] + B[idx])
    
    # GB Sector
    idx = (H >= 2 * np.pi / 3) & (H < 4 * np.pi / 3)
    H[idx] = H[idx] - 2 * np.pi / 3
    R[idx] = I[idx] * (1 - S[idx])
    G[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx]) / np.cos(np.pi / 3 - H[idx]))
    B[idx] = 3 * I[idx] - (R[idx] + G[idx])
    
    # BR Sector
    idx = (H >= 4 * np.pi / 3)
    H[idx] = H[idx] - 4 * np.pi / 3
    G[idx] = I[idx] * (1 - S[idx])
    B[idx] = I[idx] * (1 + S[idx] * np.cos(H[idx]) / np.cos(np.pi / 3 - H[idx]))
    R[idx] = 3 * I[idx] - (G[idx] + B[idx])
    
    rgb = cv2.merge([R, G, B])
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb


def transform(img1, ref):
    hsi1 = rgb_to_hsi(img1)
    hsi2 = rgb_to_hsi(ref)
    
    H1, S1, I1 = cv2.split(hsi1)
    H2, S2, I2 = cv2.split(hsi2)
    
    hsi1_with_s2 = cv2.merge([H1, S2, I1])
    image1_with_s2 = hsi_to_rgb(hsi1_with_s2)
    image1_with_s2.astype(np.uint8)
    return image1_with_s2
    
# Load the images
# image2 = cv2.imread('pflower.jpg')
# image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
# image1 = cv2.imread('Smooth.jpg')
# image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

# # Convert the images from RGB to HSI
# hsi1 = rgb_to_hsi(image1)
# hsi2 = rgb_to_hsi(image2)


# # Extract the saturation component from both images
# H1, S1, I1 = cv2.split(hsi1)
# H2, S2, I2 = cv2.split(hsi2)

# # Replace the saturation component of the first image with that of the second image
# hsi1_with_s2 = cv2.merge([H2, S2, I1])

# # Convert the modified HSI image back to RGB
# image1_with_s2 = hsi_to_rgb(hsi1_with_s2)


# # Display the HSI image
# # plt.subplot(221)
# # plt.imshow(H1)
# # plt.subplot(222)
# # plt.imshow(S1)
# # plt.subplot(223)
# # plt.imshow(I1)
# # plt.subplot(224)
# # plt.imshow(I2)
# # plt.show()
# plt.subplot(131)
# plt.imshow(image1_with_s2)
# plt.subplot(132)
# plt.imshow(image2)
# plt.subplot(133)
# plt.imshow(image1)
# plt.show()
