import cv2
import numpy as np

# 读取头像和身体图片
head = cv2.imread('/Users/binarii/head.png')
body = cv2.imread('/Users/binarii/orig.png')

# 转换为灰度图像
head_gray = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)
body_gray = cv2.cvtColor(body, cv2.COLOR_BGR2GRAY)

# 使用CUDA加速的ORB检测关键点和描述符
orb = cv2.ORB_create()
head_gpu = cv2.cuda_GpuMat()
body_gpu = cv2.cuda_GpuMat()
head_gpu.upload(head_gray)
body_gpu.upload(body_gray)

keypoints1, descriptors1 = orb.detectAndCompute(head_gpu, None)
keypoints2, descriptors2 = orb.detectAndCompute(body_gpu, None)

# 下载描述符到CPU
descriptors1 = descriptors1.download()
descriptors2 = descriptors2.download()

# 使用BFMatcher进行特征匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# 只保留好的匹配点
good_matches = sorted(matches, key=lambda x: x.distance)[:10]

# 提取匹配点的位置
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 使用RANSAC算法计算单应性矩阵
if len(good_matches) > 4:
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    print("Homography matrix calculated successfully.")
else:
    print("Not enough matches found - {}/{}".format(len(good_matches), 4))

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

