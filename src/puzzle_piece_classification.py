import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import filters
import numpy as np
from PIL import Image

from IPython.display import display, Markdown
from scipy.ndimage import median_filter
from scipy.spatial.distance import pdist

def showpic(img, width=10):
    """display image"""
    plt.figure(figsize=(width, width / 1000 * 727))
    plt.imshow(img, cmap='gray')
    plt.axis('off')

def my_contrast_stretch(img):
    """contrast stretching/normalization"""
    if img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    M = np.max(img)
    m = np.min(img)
    res = img - m
    res = res * (255 / (M - m))
    return res.astype(np.uint8)

def cart2pol(x, y):
    """cartesian to polar coordinates"""
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    """polar to cartesian coordinates"""
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def smooth(y, box_pts):
    "smoothes the function"
    box = np.ones(box_pts) / box_pts

    padding = box_pts // 2
    # mode same to keep the same number of points
    y_smooth = np.convolve(y[padding:len(y) - padding], box, mode='valid')
    return np.append(y[:padding], np.append(y_smooth, [y[len(y) - padding:]]))

def max_pairwise_distance(contour):
    points = contour[:, 0, :]  # reshape to (N, 2) if needed
    dists = pdist(points)  # pairwise Euclidean distances
    return np.max(dists)


puzzle = np.array(Image.open('../data/multiple.jpg').convert('RGB'))
#showpic(puzzle)

# thresholding/binarization
thresh = cv2.cvtColor(puzzle, cv2.COLOR_RGBA2GRAY)
thresh = my_contrast_stretch(thresh)
thresh = thresh <= thresh.mean()
#showpic(thresh)


# find contours
thresh = cv2.cvtColor(puzzle, cv2.COLOR_RGBA2GRAY)
ret2,th3 = cv2.threshold(thresh,80,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(th3.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

h, w = th3.shape[:2]
vis = np.zeros((h, w, 3), np.uint8)
cv2.drawContours(vis, contours, -1, (255,255,255), -1)
showpic(vis)

# split image in tiles representing 1 piece each

tile_size = 1000
tiles = []
for i in range(len(contours)):
    if len(contours[i]) < 100:
        continue
    x, y, w, h = cv2.boundingRect(contours[i])
    shape, tile = np.zeros(thresh.shape[:2]), np.zeros((tile_size, tile_size), 'uint8')
    cv2.drawContours(shape, [contours[i]], -1, color=1, thickness=-1)

    shape = (vis[:, :, 1] * shape[:, :])[y:y + h, x:x + w]
    tile[(tile_size - h) // 2: (tile_size - h) // 2 + h, (tile_size - w) // 2: (tile_size - w) // 2 + w] = shape
    tiles.append(tile)


def do_classification(puzzle_piece):
    img = puzzle_piece.copy()
    img = median_filter(img.astype('uint8'), size=15)
    showpic(img)
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_length = len(contours[0])

    (center_x, center_y), _ = cv2.minEnclosingCircle(contours[0])
    piece_diameter = max_pairwise_distance(contours[0])

    # get contour points in polar coordinates
    rhos = []
    phis = []
    for i in range(len(contours[0])):
        x, y = contours[0][i][0]
        rho, phi = cart2pol(x - center_x, y - center_y)
        rhos.append(rho)
        phis.append(phi)

    # apply smoothing to get rid of artifacts and paper fibres
    rhos = smooth(rhos, 5)  # adjust the smoothing amount if necessary

    tenth_of_signal = len(rhos) // 10
    print(tenth_of_signal)
    first_tenth_signal_points = rhos[:tenth_of_signal]
    # rhos += first_tenth_signal_points
    print(first_tenth_signal_points)

    rhos = np.append(rhos, [rhos[:tenth_of_signal]])
    rhos.flatten()
    print(rhos.shape)

    # plt.hist(rhos, density=True, bins=len(rhos))  # density=False would make counts
    # plt.ylabel('Dist from piece center (px)')
    # plt.xlabel('Contour point')

    # compute number of "knobs"
    peaks, peak_props = find_peaks(rhos, height=0, prominence=20, distance=30)
    # print(str(len(rhos))))
    # print(str(peaks)))
    # print(str(peak_props)))
    # print(str(len(peaks))))
    peak_points = []
    for peak in peaks:
        # x, y = pol2cart(rhos[peak], phis[peak])
        # peak_points.append([x + center_x, y + center_y])
        peak = peak % contour_length
        peak_points.append(contours[0][peak][0])

    peak_points = np.array(peak_points)
    # print(peaks)
    # print(rhos)
    plt.clf()
    plt.imshow(img)
    plt.plot(center_x, center_y, "og", markersize=5)  # og:shorthand for green circle
    for peak_point in peak_points:
        plt.plot([center_x, peak_point[0]], [center_y, peak_point[1]])
    plt.scatter(peak_points[:, 0], peak_points[:, 1], marker="x", color="red", s=40)
    plt.show()
    plt.clf()

    plt.plot(rhos)
    plt.plot(peaks, rhos[peaks], "x")
    plt.plot(np.zeros_like(rhos), "--", color="gray")
    plt.show()
    plt.clf()

    n_knobs = len(peaks) - 4
    # adjust those cases where the peak is at the borders
    if rhos[-1] >= rhos[-2] and rhos[0] >= rhos[1]:
        n_knobs += 1

    # compute number of "holes"
    rhos[rhos >= piece_diameter / 4] = rhos.max()
    rhos = 0 - rhos + abs(rhos.min())
    peaks_2, peak_props_2 = find_peaks(rhos, prominence=20, distance=30)
    n_holes = len(peaks_2)

    peak_points = []
    for peak in peaks_2:
        # x, y = pol2cart(rhos[peak], phis[peak])
        # peak_points.append([x + center_x, y + center_y])
        peak_points.append(contours[0][peak][0])
    peak_points = np.array(peak_points)

    plt.clf()
    plt.imshow(img)
    plt.plot(center_x, center_y, "og", markersize=5)  # og:shorthand for green circle
    plt.scatter(peak_points[:, 0], peak_points[:, 1], marker="x", color="red", s=40)
    plt.show()
    plt.clf()

    plt.plot(rhos)
    plt.plot(peaks_2, rhos[peaks_2], "x")
    plt.plot(np.zeros_like(rhos), "--", color="gray")
    plt.show()
    plt.clf()

    img_info = f"knobs: {n_knobs}, holes: {n_holes}"

    # classify piece
    n_features = n_knobs + n_holes
    if n_features > 4 or n_features < 0:
        img_info = "ERROR: " + img_info
    if n_features == 4:
        img_info = "Central piece: " + img_info
    if n_features == 3:
        img_info = "Border piece: " + img_info
    if n_features == 2:
        img_info = "Corner piece: " + img_info

    print(img_info)



for puzzle_piece in tiles:
    do_classification(puzzle_piece)
    break

