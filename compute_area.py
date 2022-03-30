import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from loguru import logger
import argparse
import json



def extract_circle(src, par1: float, par2: float, debug=True, count=0) -> float:
    """
    par1: it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller)
    par2: it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. 
    """

    height, width = src.shape[:-1]

    logger.info(
        f"Extract circles using parameters: par1: {par1}, par2: {par2}")

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, height / 24,
                               param1=par1, param2=par2,
                               minRadius=width//48, maxRadius=width//24)

    if circles is not None and debug:
        temp_src = src.copy()
        circles = np.uint16(np.around(circles))
        color = np.random.choice(range(256), size=3)
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(temp_src, center, 3, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(temp_src, center, radius,
                       (int(color[0]), int(color[1]), int(color[2])), 3)
        cv2.imshow("detected circles", temp_src)
        cv2.waitKey(0)

    if count > 20:
        logger.info("Can't find representative circles.")
        return 0

    num_circle = len(circles.squeeze())

    if circles is None:
        logger.info("No circles are detected. Parameters will be loose.")
        
        return extract_circle(src, par1=par1*np.random.uniform(low=0.6, high=0.8), par2=par2*np.random.uniform(low=0.6, high=0.8), count=count+1)

    elif num_circle < 4:
        logger.info(
            f"Not enough number of circles are detected. ({num_circle} circles) Parameters will be loose.")
        return extract_circle(src, par1=par1*np.random.uniform(low=0.7, high=0.9), par2=par2*np.random.uniform(low=0.7, high=0.9), count=count+1)

    # if it doesn't converge, increase the value to higher than 2.
    elif np.std(circles.squeeze()[:, 2]) > 2:
        logger.info(
            "Too many circles with different size. Parameters will be strict.")
        return extract_circle(src, par1=par1*np.random.uniform(low=1.1, high=1.3), par2=par2*np.random.uniform(low=1.1, high=1.3), count=count+1)

    else:
        circle_radius = np.median(circles, axis=1)[0, 2]
        logger.info(f"Found the right parameters!: {circle_radius}")
        return circle_radius


def compute_pixel_ratio(radius: float, ACTUAL_DIM: float) -> float:
    """
    radius: radius in pixel unit
    ACTUAL_DIM: actual dimension of diameter in cm unit
    """
    diameter = radius * 2
    ratio = (ACTUAL_DIM/diameter)**2
    logger.info("Ratio of cm2/px is computed: "+str(ratio))

    return round(ratio, 4)


def update_meta(img_name: str, ratio: float, meta_loc: str, separator):

    if meta_loc:
        """ If no meta data, create a new one. """
        if not os.path.isdir(meta_loc):
            os.mkdir(meta_loc)

        loc_meta = os.path.join(meta_loc, "hexa_meta.json")
        key_img_name = separator.join(
            os.path.basename(img_name).split(separator)[:-1])

        meta = {key_img_name: ratio}

        if os.path.exists(loc_meta):
            """ Append to the existing one"""
            with open(loc_meta, "r+") as j:
                data = json.load(j)

            if key_img_name in data.keys():
                logger.info("Already meta of this image")
                if abs(meta[key_img_name] - ratio) > ratio * 0.2:
                    logger.warn(
                        f"ratio of this image is different to old ratio. Old: {data[key_img_name]}, New: {ratio}. Keep the old one.")
            else:
                logger.info(f"Update {key_img_name}")
                data.update(meta)
                with open(loc_meta, "w") as k:
                    json.dump(data, k, indent=4)

        else:
            logger.info(f"Generate a new meta data.")
            with open(loc_meta, 'w') as f:
                json.dump(meta, f, indent=4)


def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
        description="Distort images")

    parser.add_argument("--input",
                        default='input',
                        help="Location of image directory.")

    parser.add_argument("--meta",
                        default="meta",
                        help="Location of meta data.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    images = list(Path().glob(os.path.join(args.input,"*")))
    meta_loc = args.meta

    default_par1 = 200
    default_par2 = 30
    SEPARATOR = "-"
    ACTUAL_DIM = 16  # unit: cm, actual

    logger.info(f"{len(images)}images will be processed.")
    for img in images:

        src = cv2.imread(str(img), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        radius = extract_circle(src, par1=default_par1,
                                par2=default_par2, debug=True)
        if radius == 0:
            logger.info(f"Skip {str(img)}")
            continue 

        ratio = compute_pixel_ratio(radius, ACTUAL_DIM)
        update_meta(img, ratio, args.meta, SEPARATOR)

    logger.info("Work Complete!.")
