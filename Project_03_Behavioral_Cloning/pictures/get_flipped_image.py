import cv2
import numpy as np
 
if __name__ == '__main__' :
 
    # Read image
    print("Image is read in BGR format using OpenCV ...")
    image = cv2.imread("right.jpg")
    print("The shape of the image is: " + str(image.shape))
     
    # flipped image: row, column
    image_flipped = cv2.flip(image,1)
 
    cv2.imshow("Image", image)
    cv2.imshow("Flipped Image", image_flipped)

    cv2.imwrite("right_flip.jpg", image_flipped)

    cv2.waitKey(0)

