import numpy as np
import cv2, os, sys, shutil

sys.path.insert(0,  r'./machine_vision/external/panorama-image-stitching')
from image_stitching.read_images import read as read_images
from image_stitching.recursion import recurse

def fuse_line(line_stack):
    images_list = read_images(line_stack)
    fusion, _ = recurse(images_list)
    return fusion

def fuse_column(column_stack):
    images_list = read_images(column_stack)
    images_list = [rotate_image(im, angle_degree=90) for im in images_list]
    fusion, _ = recurse(images_list)
    rotated_fusion = rotate_image(fusion, angle_degree=-90)
    return rotated_fusion

def rotate_image(img, angle_degree=90, scale = 1.0):
    # Get the image dimensions
    h, w = img.shape[:2]

    # Define the rotation center
    center = (w // 2, h // 2)

    # Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle_degree, scale)

    # Calculate the new bounding dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Perform the actual rotation and return the image
    rotated_image = cv2.warpAffine(img, M, (new_w, new_h))

    return rotated_image

def stackImages(array = np.ndarray[str]):
    lines, cols = array.shape

    # verify empty cell


    # for each line classic merge columns
    temp_dir = os.path.join("temps")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        line_stack, raw = [], []
        for line in range(lines):
            image_dir_list = array[line,:]
            images_list = read_images(image_dir_list)
            result, _ = recurse(images_list)


            filename = f"partial_line{line}.png"
            cv2.imwrite(os.path.join("temps","_.png"), _)
            cv2.imwrite(os.path.join("temps",filename), result)
            raw.append(os.path.join("temps",filename))

            rotated_img_ = rotate_image(result, angle_degree=90)
            imgname = f"rotated_{filename}"
            cv2.imwrite(os.path.join("temps",imgname), rotated_img_)
            line_stack.append(os.path.join("temps",imgname))

        # for merging line -> rotate inputs and rotate output
        images_list = read_images(line_stack[:])
        fusion, _ = recurse(images_list)
        cv2.imwrite(os.path.join("temps","que_temp.png"), _)
        cv2.imwrite(os.path.join("temps","map_temp.png"), fusion)

        l = cv2.imread(os.path.join("temps","map_temp.png"))
        rotated_fusion = rotate_image(l, angle_degree=-90)
        cv2.imwrite(os.path.join(".","map.png"), rotated_fusion)


    finally:
        print("")
        # Cleanup: remove temp folder and its contents
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

import glob
if __name__ == "__main__":

    nada = ""
    filepaths = glob.glob("machine_vision/db/all_chip//*.png")
    stack = np.array([[filepaths[2],filepaths[1]], [filepaths[3],filepaths[4]]])
    stack = np.array([[filepaths[6],filepaths[7]], [filepaths[3],filepaths[4]]])
    stack = np.array([[filepaths[6],filepaths[7]], [filepaths[5],filepaths[4]]])
    stack = np.array([[filepaths[2],filepaths[1]], [filepaths[5],filepaths[4]]])

    stackImages(stack)