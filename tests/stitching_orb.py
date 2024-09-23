import cv2
import numpy as np

head = cv2.imread('/Users/binarii/head_2.png')
body = cv2.imread('/Users/binarii/orig.png')

head_gray = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)
body_gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB.create()

keypoints1, descriptors1 = orb.detectAndCompute(head_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(body_gray, None)

keypoints1 = np.float32([kp.pt for kp in keypoints1])
keypoints2 = np.float32([kp.pt for kp in keypoints2])

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
good_matches = sorted(matches, key=lambda x: x.distance)[:10]

src_pts = np.float32([keypoints1[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)

if len(good_matches) > 4:
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print("Homography matrix calculated successfully.")
else:
    print("Not enough matches found - {}/{}".format(len(good_matches), 4))

height, width, channels = body.shape
head_warped = cv2.warpPerspective(head, M, (width, height))

mask = np.zeros((height, width), dtype=np.uint8)
mask[head_warped[:, :, 0] > 0] = 255

combined = cv2.bitwise_and(body, body, mask=cv2.bitwise_not(mask))
combined += head_warped

cv2.imwrite('combined_image_orb.jpg', combined)

