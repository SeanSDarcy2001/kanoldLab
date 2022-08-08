
import numpy as np
import cv2

#enter paths here
templatePath = "/Volumes/Data2/seandarcy/Sean/m794_map_40dB.png"
imgPath = "/Volumes/Data2/seandarcy/Sean/m794_map_40dB.png"

def align_images(image, template, maxFeatures=500, keepPercent=0.2):
	# convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
 
    # use ORB to detect keypoints and extract (binary) local
	# invariant features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
	# match the features
    method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    matcher = cv2.DescriptorMatcher_create(method)
    matches = matcher.match(descsA, descsB, None)
 
 # sort the matches by their distance (the smaller the distance,
	# the "more similar" the features are)
    matches = sorted(matches, key=lambda x:x.distance)
	# keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]
  
  # allocate memory for the keypoints (x, y)-coordinates from the
	# top matches -- we'll use these coordinates to compute our
	# homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")
	# loop over the top matches
    for (i, m) in enumerate(matches):
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt
    
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    #compare computed Freg to Identity for error
    error = np.absolute(H - np.eye(H.shape[0])) * (1/100)
    return error

# load the input image and template from disk
print("[INFO] loading images...")
image = cv2.imread(imgPath)
template = cv2.imread(templatePath)
# align the images
print("[INFO] computing alignment error...")
error = align_images(image, template)
print("Alignment error is:", error)

