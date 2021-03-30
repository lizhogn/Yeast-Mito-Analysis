"""
Cell Segmentation
Use the Mask R-CNN network to detect and segmentation the cell
Then meature the cell morphological parameters

Copyright (c) 2020 Lizhogn, Inc.
Licensed under the MIT License (see LICENSE FOR details)
Written by Lizhongzhong


---------------------------------------------------------
Usage: 

"""

import os
import skimage.draw
import cv2
import csv
from mrcnn.config import Config
from mrcnn import model as modellib

class InferenceConfig(Config): # bai
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "cell" # bai

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + cell

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 200 # 100 bai

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9



def cell_analysis(model, img, calibration=1):
    """
    cell morphological and mitochondrial analysis
    input:
        img: input img
        calibration: the real length 1 pixel(if the calibration equals 1, output length's unit is pixel
    output:
        cell: a cell object contains each measurements
    """

    # STEP3: inference the img
    '''
        return results[0] format: (type: dict)
            masks: [heigth, width, num_cells]
            class_ids: [num_cells,]
            scores: [num_cells,]
            rois: [num_cells, 4]
    '''

    results = model.detect([img], verbose=1)
    results = results[0]

    # STEP4： cell analysis
    # meature the cell morphological parameters
    cell_obj = cell_measure(img, results, calibration)

    return cell_obj



def detect_and_show(model, image_path=None, video_path=None):
    # use the model to inference the cell
    from utils import visualize_cv2
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detction and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)

        # Detect objects
        results = model.detect([image], verbose=1)
        r = results[0]
        '''
        return results[0] format: (type: dict)
            masks: [heigth, width, num_cells]
            class_ids: [num_cells,]
            scores: [num_cells,]
            rois: [num_cells, 4]
        '''
        # # Save output
        import datetime
        file_name = "detected_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())

        visualize_cv2.save_image(image, file_name, r['rois'], r['masks'],
            r['class_ids'], r['scores'], ['BG', 'cell'],
            filter_classs_names=['cell'], scores_thresh=0.7, mode=0)

        return image, r

def cell_measure(image, results, calibration=1):

    '''
    inputs:
        image: the cell image: [height, width, 3]
        results: a dict type contains:
            {
                masks: [heigth, width, num_cells]
                class_ids: [num_cells,]
                scores: [num_cells,]
                rois: [num_cells, 4]
            }
    '''
    from utils.cell_process import CellProcess
    cells = CellProcess(image, results, calibration)
    # measure the cell morphological parameter
    cells.cell_morphological_measure()
    # measure the Mitochondrial parameters
    cells.cell_mitochondrial_measure()

    return cells


if __name__=="__main__":

    '''
    input:
        1. cell image path
        2. save path
    output:
        1. ???
    '''

    # STEP1: access the parameters
    import argparse
    # parse the command line arguments
    parser = argparse.ArgumentParser(description="show the Mask R-CNN detect results")
    parser.add_argument("--image", "-i",
                        default="input_imgs/XY point32-Merge.png")
    parser.add_argument("--weights", required=False,
                        default='last',
                        metavar="logs/cell_1/mask_rcnn_cell.h5",
                        help="Path to weights .h5 file or 'latest")
    args = parser.parse_args()

    # validate arguments
    assert args.image, "Provide --image or -i to show detected result"
    print("image_path: ", args.image)

    config = InferenceConfig()
    config.display()

    # STEP2: create the model and load the weights
    model = modellib.MaskRCNN(mode='inference', config=config, model_dir='logs')
    if args.weights == 'last':
        weights_path = model.find_last()
    else:
        weights_path = args.weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    # STEP3: Inference the image
    '''
    return results[0] format: (type: dict)
        masks: [heigth, width, num_cells]
        class_ids: [num_cells,]
        scores: [num_cells,]
        rois: [num_cells, 4]
    '''
    image, results = detect_and_show(model, image_path=args.image)

    # STEP4: extract every detected cells
    # meature the cell morphological parameters
    cell_obj = cell_measure(image, results)

    # STEP5: save the results
    # # folder create
    folder_name = os.path.basename(args.image).split('.')[0]
    output_dir = os.path.join('output', folder_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # image save
    # 1. origin img save
    origin_img_name = os.path.join(output_dir, 'origin.png')
    cv2.imwrite(origin_img_name, img=image)
    # 2. detect img save
    detect_img_name = os.path.join(output_dir, 'detect.png')
    cv2.imwrite(detect_img_name, img=cell_obj.detect_img)
    # 3. cell morphological and mitochondrial infos save
    # merge_table keys:
    # 'Cell_Id', 'Length', 'Width', 'mitochondrial_numbers',
    # 'mitochondrial_overall_length', 'mitochondrial_each_segment_len'
    cell_infos = cell_obj.merge_measurements()
    cell_infos_name = os.path.join(output_dir, 'cell_infos.csv')
    with open(cell_infos_name, 'w', newline='') as csvfile:
        fieldnames = list(cell_infos[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cell_infos)
