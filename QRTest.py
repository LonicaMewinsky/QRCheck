import cv2
import numpy as np
import qreader

#### Top marks for QRCode: 0.22 tilt, 109 blur
# Load the image
image = cv2.imread('tqr2.webp')

#Return QR data
def read_qr(image):
    results = []
    #Decode QR using qreader
    try:
        decoded = qreader.decodeQR(image)
    except:
        return None
    for code in decoded:
        results.append(str(code.data))

    return results

def read_and_test_qr(image, url):
    #Decode QR using qreader
    #If the URL is found, return true, else false
    try:
        decoded = qreader.decodeQR(image)
    except:
        return False
    for code in decoded:
        if url in str(code.data):
            return True
    
    return False

#Force perspective skew by image direction
def add_skew(image, l_factor = 0, u_factor = 0, r_factor = 0, d_factor = 0):
    height, width = image.shape[:2]

    top_left = [0 + int(u_factor * width), 0 + int(l_factor * height)]
    top_right = [width - int(u_factor * width), 0 + int(r_factor * height)]
    bot_left = [0 + int(d_factor * width), height - int(l_factor * height)]
    bot_right = [width - int(d_factor * width), height - int(r_factor * height)]

    # Define the source points..
    src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # Define the destination points based on factors
    dst_points = np.float32([top_left, top_right, bot_left, bot_right])
    
    #Apply perspective skew
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    result = cv2.warpPerspective(image, perspective_matrix, (width, height))

    return result

#Force gaussian blur; will adjust to odd number
def add_gblur(image, factor):
    #Ensure odd number
    rounded = round(factor)
    if rounded % 2 == 0:
        if factor < rounded:
            factor = rounded - 1
        else:
            factor = rounded + 1
    else:
        factor = rounded

    return cv2.GaussianBlur(image, (factor, factor), 0)

#Perform 56 progressive blur tests, return float as % score
def test_blur(image, url):
    score = 0
    test_scales = np.linspace(0, 109, 56, dtype=int)
    for scale in test_scales:
        blurred = add_gblur(image, scale)
        if read_and_test_qr(blurred, url): score += 1
    score = round(score/56, 3)

    return score

#Perform 172 progressive tilt tests, return float as % score
def test_tilt(image, url):
    score = 0
    test_scales = np.linspace(0.01, 0.22, 22)
    for scale in test_scales:
        if read_and_test_qr(add_skew(image, scale, 0, 0, 0), url): score += 1
        if read_and_test_qr(add_skew(image, scale, scale, 0, 0), url): score += 1
        if read_and_test_qr(add_skew(image, 0, scale, 0, 0), url): score += 1
        if read_and_test_qr(add_skew(image, 0, scale, scale, 0), url): score += 1
        if read_and_test_qr(add_skew(image, 0, 0, scale, 0), url): score += 1
        if read_and_test_qr(add_skew(image, 0, 0, scale, scale), url): score += 1
        if read_and_test_qr(add_skew(image, 0, 0, 0, scale), url): score += 1
        if read_and_test_qr(add_skew(image, scale, 0, 0, scale), url): score += 1
    score = round(score/172, 3)

    return score

#Test baseline
qr_baseline = read_qr(image)

#Test tilt 0.01 -> 0.22
print(test_tilt(image, ".com"))

#Test blur 0 -> 109
print(test_blur(image, ".com"))