import numpy as np
import cv2, os, sys, shutil

sys.path.insert(0,  r'./machine_vision/external/panorama-image-stitching')
from image_stitching.read_images import read as read_images
from image_stitching.recursion import recurse

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

def fuse_line(line_stack):
    images_list, n_im = read_images(line_stack)
    return recurse(images_list, n_im)

def fuse_column(column_stack):
    images_list, n_im = read_images(column_stack)
    images_list = [rotate_image(im, angle_degree=90) for im in images_list]
    fusion, _ = recurse(images_list, n_im)
    rotated_fusion = rotate_image(fusion, angle_degree=-90)
    return rotated_fusion, rotate_image(_, angle_degree=-90)

def fuse(arr = np.ndarray[str], output_directory:str = ".", name_="panorama.png", show_cache=False):
    lines, cols = arr.shape

    # ----------------------------------------------------------------------
    # Check the dimension of the array
    if arr.ndim != 2:
        raise ValueError("[invalid input]: array should have 2 dimensions")
    
    # Check if all elements are strings
    if arr.dtype == str:
        raise ValueError("[invalid input]: content of array should be str")
    
    # verify empty cell and array dimensions
    if np.any(arr == ""):
        raise ValueError("[invalid input]: empty str detected")

    # Check if all elements are unique
    all_unique = len(np.unique(arr)) == arr.size
    if all_unique is not True:
        raise ValueError("[invalid input]: 2 paths are the same")
    
    # ----------------------------------------------------------------------
    # for each line classic merge columns
    temp_dir = os.path.join("temps")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        if lines == 1:
            fusion, _ = fuse_line(arr[0,:])

        elif cols == 1:
            fusion, _ = fuse_column(arr[:,0])

        else:
            line_stack = []
            for line in range(lines):
                image_dir_list = arr[line,:]
                result, _ = fuse_line(image_dir_list)

                filename = f"partial_line{line}.png"
                cv2.imwrite(os.path.join("temps","_.png"), _)
                cv2.imwrite(os.path.join("temps",filename), result)

                line_stack.append(os.path.join("temps",filename))

            # for merging line -> rotate inputs and rotate output
            fusion, _ = fuse_column(line_stack)

        cv2.imwrite(os.path.join("temps","map.png"), _)
        cv2.imwrite(os.path.join(output_directory,name_), fusion)

    finally:
        # Cleanup: remove temp folder and its contents
        if os.path.exists(temp_dir) and not show_cache:
            shutil.rmtree(temp_dir)