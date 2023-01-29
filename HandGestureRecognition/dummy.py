from PIL import Image
import cv2

def run_dummy():
    img = cv2.imread('C:\\Users\\rajen\\Documents\\Udhay\\MS Docs\\Classes\\Term3_F2021\\Research Methodology\\Datasets\\8Signs\\Dataset\\Test_Single\\PU_1.jpg', 2)
    ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('Binary Image', bw_img)
    cv2.waitKey(0)  # wait for a keyboard input
    cv2.destroyAllWindows()
    cv2.imwrite('C:\\Users\\rajen\\Documents\\Udhay\\MS Docs\\Classes\\Term3_F2021\\Research Methodology\\Datasets\\8Signs\\Dataset\\Test_single\\greyscale5.jpg', bw_img)