import cv2
import numpy as np
 
if __name__ == '__main__' :
 
    # Read image
    print("Image is read in BGR format using OpenCV ...")
    image = cv2.imread("center.jpg")
    print("The shape of the image is: " + str(image.shape))
     
    # Crop image: row, column
    image_crop = image[71:136, 0:320]
    print("The shape of the cropped image is: " + str(image_crop.shape))
 
    cv2.imshow("Image", image)
    cv2.imshow("Cropped Image", image_crop)

    # cv2.imwrite("image_crop.jpg", image_crop)

    cv2.waitKey(0)