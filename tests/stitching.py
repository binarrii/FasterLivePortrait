import cv2
import numpy as np

# 读取头像和身体图片
head = cv2.imread('/Users/binarii/head.png')
body = cv2.imread('/Users/binarii/body_2.png')

# 转换为灰度图像
head_gray = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)
body_gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)

# 使用SIFT检测关键点和描述符
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(head_gray, None)
keypoints2, descriptors2 = sift.detectAndCompute(body_gray, None)

# 使用FLANN匹配器进行特征匹配
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# 只保留好的匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
print(f"{len(good_matches)}")

# 提取匹配点的位置
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 使用RANSAC算法计算单应性矩阵
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 应用单应性矩阵将头像变换到身体图像上
height, width, channels = body.shape
head_warped = cv2.warpPerspective(head, M, (width, height))

# 创建一个掩码来合并图像
mask = np.zeros((height, width), dtype=np.uint8)
mask[head_warped[:, :, 0] > 0] = 255

# 使用掩码将头像和身体合并
combined = cv2.bitwise_and(body, body, mask=cv2.bitwise_not(mask))
combined += head_warped

# 保存最终的图像
cv2.imwrite('combined_image.jpg', combined)

