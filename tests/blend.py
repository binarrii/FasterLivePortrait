import cv2
import numpy as np

# 读取图像
img1 = cv2.imread('/Users/binarii/head.png')
img2 = cv2.imread('/Users/binarii/body_2.png')

# 将图像转换为相同的大小
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# 混合图像
alpha = 0.5
blended = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

# 显示结果
cv2.imshow('Blended Image', blended)
cv2.waitKey(0)
cv2.destroyAllWindows()

