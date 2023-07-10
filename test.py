import numpy as np	
import cv2
import matplotlib.pyplot as plt

# img1=cv2.imread('C:/Users/dbseo/Desktop/codes/test/1.jpg')
# img2=cv2.imread('C:/Users/dbseo/Desktop/codes/test/2.jpg')
# img3=cv2.imread('C:/Users/dbseo/Desktop/codes/test/3.jpg')
# img4=cv2.imread('C:/Users/dbseo/Desktop/codes/test/4.jpg')
# img5=cv2.imread('C:/Users/dbseo/Desktop/codes/test/5.jpg')

# np1 = np.load('./gt/gt_1.npy')
# np2 = np.load('./gt/gt_2.npy')
np3 = np.load('./data/gt/gt_1.npy')
# np4 = np.load('./gt/gt_4.npy')
# np5 = np.load('./gt/gt_5.npy')

# for i in range(2, 3):
#     aa = np.load(f'./mask/{i}.npy')
#     np1 += aa 
# np2 = np.load('./mask1_2.npy')


# overlap = np0*np1 # Logical AND
# union = np0 + np1 # Logical OR

# IOU = overlap.sum()/float(union.sum()) 
# print(img1.shape, np1.shape)

# mask = np.zeros(img1.shape[:2], dtype="uint8")
# print(img1.shape, mask.shape)

# masked = cv2.bitwise_and(img1, img1, mask=np1)
# new = np0 + np1
# print(IOU)

np3[300:,:] = 0

np.save(f'./gt_1_1.npy', np3.astype(int))

plt.imshow(np3, cmap='gray')
# plt.show()
plt.savefig('./test.png', bbox_inches='tight', pad_inches=0)
plt.close()
# cv2.waitKey(0)  # 변수값만큼 사용자의 키입력 시간을 대기시킴
# cv2.destroyAllWindows() 