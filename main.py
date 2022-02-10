import cv2
import numpy as np
from PIL import Image

from CV_Func import fuse_color_images
from CV_Func import create_named_window
from CV_Func import get_xy


def main():
    discs = ["images/1.jpg", "images/2.jpg",
             "images/3.jpg", "images/4.jpg",
             "images/5.jpg", "images/6.jpg"]

    images = ['./Mural/mural01.jpg', './Mural/mural02.jpg',
              './Mural/mural03.jpg', './Mural/mural04.jpg',
              './Mural/mural05.jpg', './Mural/mural06.jpg',
              './Mural/mural07.jpg', './Mural/mural08.jpg',
              './Mural/mural09.jpg', './Mural/mural10.jpg',
              './Mural/mural11.jpg', './Mural/mural12.jpg']



    # Icurrent = cv2.imread("Mural/mural01.jpg")

    Icurrent = cv2.imread('images/1.jpg')

    image1_width = Icurrent.shape[1]
    image1_height = Icurrent.shape[0]

    # -> Mural
    # Output image dimensions -> Mural
    # output_image_height = image1_height
    # output_image_width = 8 * image1_width

    # -> Disc Images
    output_image_height = image1_height * 2
    output_image_width = 6 * image1_width

    # Mural
    # src = np.array([(161, 127), (379, 80), (301, 496), (38, 465)])  # Paint
    # dst = np.array([(0, 0), (318, 0), (318, 435), (0, 435)])  # Calculated
    # dst[:, 0] += image1_width  # Shift x coordinates by width

    src = np.array([(9, 17), (785, 5), (8, 582), (795, 586)])
    dst = np.array([(0, 0), (800, 0), (0, 600), (800, 600)])  # Calculated
    dst[:, 0] += image1_width  # Shift x coordinates by width

    '''MOUSE CALL BACK'''
    # Create list.  The (x,y) points go in this list.
    # Create list.  The (x,y) points go in this list.
    # input_points_0 = []
    #
    # # Display image.
    # m_0_0 = Icurrent
    # create_named_window("m_0_0", m_0_0)
    # cv2.imshow("m_0_0", m_0_0)
    #
    # # Assign the mouse callback function, which collects (x,y) points.
    # cv2.setMouseCallback("m_0_0", on_mouse=get_xy, param=("m_0_0", m_0_0, input_points_0))
    # cv2.waitKey(0)
    # # Loop until user hits the ESC key.
    # print("Click on points.  Hit ESC to exit.")
    #
    # print("input_points_01:", input_points_0)  # Print points to the console

    H_current_mosaic, _ = cv2.findHomography(src, dst)
    Imosaic = cv2.warpPerspective(Icurrent, H_current_mosaic, (output_image_width, output_image_height))

    # cv2.imshow("Imos", Imosaic)
    # cv2.waitKey(0)

    # Image Previous = Image Current
    Iprev = Icurrent
    # H_prev = H_current
    H_prev_mosaic = H_current_mosaic

    for image in discs:
        Icurrent = cv2.imread(image)
        # Convert to grayscale
        Iprev_gray = cv2.cvtColor(Iprev, cv2.COLOR_BGR2GRAY)
        Icurrent_gray = cv2.cvtColor(Icurrent, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=5000)  # Amped it up from 2k to 5k with all the noise (carpet) in my images
        kp_prev = orb.detect(Iprev_gray)
        kp_prev, desc_prev = orb.compute(Iprev_gray, kp_prev)
        kp_curr = orb.detect(Icurrent_gray)
        kp_curr, desc_curr = orb.compute(Icurrent_gray, kp_curr)
        matcher = cv2.BFMatcher.create(cv2.NORM_HAMMING, crossCheck=False)

        # Match query image descriptors to the training image.
        # Use k nearest neighbor matching and apply ratio test.
        matches = matcher.knnMatch(desc_curr, desc_prev, k=2)
        good = []
        # m and n are the two best matches
        for m, n in matches:
            # 0.8 for my images and used 0.6 for Mural
            if m.distance < 0.8 * n.distance:
                good.append(m)
        matches = good
        # Reshape not needed
        src_pts = np.float32([kp_curr[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp_prev[m.trainIdx].pt for m in matches])

        H_current_prev, _ = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)
        H_current_mosaic = H_prev_mosaic @ H_current_prev
        Icurrent_warp = cv2.warpPerspective(Icurrent, H_current_mosaic, (output_image_width, output_image_height))
        Imosaic = fuse_color_images(Imosaic, Icurrent_warp)
        create_named_window("Mosaic", Imosaic)
        cv2.imshow("Mosaic", Imosaic)
        cv2.waitKey(0)
        Iprev = Icurrent
        H_prev_mosaic = H_current_mosaic

    # Cross fingers
    cv2.imwrite("discs.jpg", Imosaic)


if __name__ == '__main__':
    main()
