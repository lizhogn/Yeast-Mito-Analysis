import cv2
import os
import numpy as np
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from skimage import morphology
from skimage import measure, color
import csv
import matplotlib.pyplot as plt


class CellProcess():
    '''
    Compute the cell morphological and Mitochondrial parameters
    inputs:
        img: input cell image
        infos: a dict contains
                {
                masks: [heigth, width, num_cells]
                class_ids: [num_cells,]
                scores: [num_cells,]
                rois: [num_cells, 4]
                }
    need to output: every cell height and width
    '''
    def __init__(self, img, infos, calibration, border_cell_filter=True):
        # load the img and detected informations
        self.img = img
        self.infos = infos

        # predefine the result table list
        self.morphological_table = []
        self.mitochondrial_table = []

        # scale setting
        self.pixelsPerMetric = float(calibration)

        # border cell filtration
        self.border_cell_filter = True
        # border cell filtration
        img_height, img_width, _ = self.img.shape
        delete_index = []
        for index, roi in enumerate(self.infos['rois']):
            [x1, y1, x2, y2] = roi
            if (x1 <= 1) or (y1 <= 1) or ((img_height - x2) <= 1) or ((img_width - y2) <= 1):
                delete_index.append(index)

        # delete the border cell
        self.infos['masks'] = np.delete(self.infos['masks'], delete_index, 2)
        self.infos['class_ids'] = np.delete(self.infos['class_ids'], delete_index, 0)
        self.infos['scores'] = np.delete(self.infos['scores'], delete_index, 0)
        self.infos['rois'] = np.delete(self.infos['rois'], delete_index, 0)

    def cell_morphological_measure(self):
        '''
        cell morphological measurement
        :return:
        '''
        # data initialize
        masks = self.infos['masks']
        orig = self.img.copy()

        # pre-define a common used function
        def midpoint(ptA, ptB):
            return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

        for i in range(len(self.infos['scores'])):
            # generate every mask's rectangular
            # 1. split the mask
            cur_mask = np.uint8(masks[:,:,i])
            # 2. Use dilation and erosion operations to close the gap between the edges of objects
            cur_mask = cv2.erode(cur_mask, None, iterations=2)
            cur_mask = cv2.dilate(cur_mask, None, iterations=2)

            # 3. tranlate the mask edge into 2D vector points
            cnts = cv2.findContours(cur_mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            # 4. Calculate the circumscribed rectangular frame according to the contour of the object
            # todo: use the try catch to improve the robust
            try:
                box = cv2.minAreaRect(np.squeeze(cnts[0]))
            except Exception:
                print('--'*10)
                print('the {i}th mask can not find the minAreaRect, just use the origin box')
                morphological_dict = {'Cell_Id': i, 'Length(um)': None, 'Width(um)': None}
                self.morphological_table.append(morphological_dict)
                x1, y1, x2, y2 = self.infos['rois'][i]
                centerX, centerY = midpoint((y1, x1), (y2, x2))
                cv2.rectangle(orig, (y1, x1), (y2, x2), (0, 0, 255), 1)
                cv2.putText(orig, "{}".format(i),
                            (int(centerX), int(centerY)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, (255, 255, 255), 1)
                continue

            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")

            # Sort the contour points in the order of top-left, top-right, bottom-right,
            # and bottom-left, and draw the outer BB, which is represented by a green line
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 1)

            # Draw the 4 vertices of BB, represented by small red circles
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 1, (0, 0, 255), -1)

            # Calculate the center point of top-left and top-right and
            # the center point coordinates of bottom-left and bottom-right respectively
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)

            # Calculate the center point of top-left and top-right and
            # the center point coordinates of top-righ and bottom-right respectively
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            # Calculate the center point of the rectangle
            (centerX, centerY) = midpoint(tl, br)

            # Draw the center points of the 4 sides of BB, represented by small blue circles
            cv2.circle(orig, (int(tltrX), int(tltrY)), 1, (255, 0, 0), -1)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 1, (255, 0, 0), -1)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 1, (255, 0, 0), -1)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 1, (255, 0, 0), -1)

            # Draw a straight line between the center points, represented by a purple-red line
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                     (255, 0, 255), 1)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                     (255, 0, 255), 1)

            # Calculate the Euclidean distance between two center points, that is, the picture distance
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # Initialize the measurement index value, the width of the reference object
            # in the picture has been calculated through the Euclidean distance,
            # and the actual size of the reference object is known

            # Calculate the actual size (width and height) of the target, expressed in feet
            dimA = round(dA * self.pixelsPerMetric,2)
            dimB = round(dB * self.pixelsPerMetric,2)

            # # draw the results in the image
            # cv2.putText(orig, "{:.1f}in".format(dimA),
            #             (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.3, (255, 255, 255), 1)
            # cv2.putText(orig, "{:.1f}in".format(dimB),
            #             (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.3, (255, 255, 255), 1)
            cv2.putText(orig, "{}".format(i),
                        (int(centerX), int(centerY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (255, 255, 255), 1)

            # save the morphological results
            morphological_dict = {'Cell_Id': i, 'Length(um)': max(dimA, dimB), 'Width(um)': min(dimA, dimB)}
            self.morphological_table.append(morphological_dict)

        # add the detect img to the cell object
        self.detect_img = orig

    def cell_mitochondrial_measure(self):
        '''
        compute the cell mitochondrial parameter
        :return:
        '''
        cell_nums = len(self.infos['scores'])
        img = self.img
        masks = self.infos['masks']
        # process each cells
        for i in range(cell_nums):
            # STEP1: extract each cell fontground
            cur_cell = img*masks[:, :, i, None]

            # miss mitocohndrial cell filter
            miss_mito_flag = self._is_miss_mitochondrial(img, masks[:, :, i])
            if miss_mito_flag:
                # if miss the mitochondrial, just continue
                statistic_results = {'mitochondrial_numbers': None,
                                     'mitochondrial_overall_length(um)': None,
                                     'mitochondrial_each_segment_len(um)': None,
                                     'mitochondrial_each_seg_intersection_num': None}

                self.mitochondrial_table.append(statistic_results)
                continue

            # color split and Green - Blue channel
            mito_gray = cur_cell[:, :, 1] - cur_cell[:, :, 2]

            # STEP2: Adjust Brightness and Contrast
            mito_gray = self._adjustBrightness(mito_gray)
            # plt.imshow(mito_gray, cmap='gray')
            # plt.axis('off')
            # plt.show()

            # STEP3: Convert to mask
            mito_mask = self._image_binary(mito_gray)
            # plt.imshow(mito_mask, cmap='gray')
            # plt.axis('off')
            # plt.show()

            # STEP4: Image thining
            mito_skeleton = self._image_skeletonize(mito_mask)
            # plt.imshow(mito_skeleton, cmap='gray')
            # plt.axis('off')
            # plt.show()

            # STEP5: Statistics the mitochondrial
            statistic_results = self._mito_statistics(mito_skeleton)
            self.mitochondrial_table.append(statistic_results)


            #
            # # save the process image
            # filename_contrast = os.path.join('../image_process', '{i}_STEP2_ouput.png'.format(i=i))
            # # cv2.imwrite(filename_contrast, mito_gray)
            # plt.imsave(filename_contrast, mito_gray, cmap='gray')
            #
            # filename_mask = os.path.join('../image_process', '{i}_STEP3_ouput.png'.format(i=i))
            # # cv2.imwrite(filename_mask, mito_mask)
            # plt.imsave(filename_mask, mito_mask, cmap='gray')
            #
            # filename_thin = os.path.join('../image_process', '{i}_STEP4_ouput.png'.format(i=i))
            # plt.imsave(filename_thin, mito_skeleton, cmap='gray')

    def show_specific_cell(self, cell_id):
        # show the specific id cell
        cell_nums = len(self.infos['scores'])
        img = self.img
        masks = self.infos['masks']

        cur_cell = img * masks[:, :, cell_id, None]

        # color split and Green - Blue channel
        mito_gray = cur_cell[:, :, 1] - cur_cell[:, :, 2]

        # STEP2: Adjust Brightness and Contrast
        mito_gray = self._adjustBrightness(mito_gray)

        # STEP3: Convert to mask
        mito_mask = self._image_binary(mito_gray)

        # STEP4: Image thining
        mito_skeleton = self._image_skeletonize(mito_mask)

        # STEP5: Intersection Counter
        Intersections = self.getSkeletonIntersection(mito_skeleton)
        label_mito = measure.label(mito_skeleton, connectivity=2)
        label_mito = color.label2rgb(label_mito, bg_label=0)
        for intersect in Intersections:
            cv2.circle(label_mito, intersect, 1, (255, 255, 255), -1)
        print(label_mito)
        # plot the image
        plt.figure(figsize=(15, 4))

        plt.subplot(1,4,1)
        plt.imshow(cur_cell)
        plt.title('STEP1: cell segmentation')
        plt.axis('off')

        plt.subplot(1,4,2)
        plt.imshow(mito_gray, cmap='gray')
        plt.title('STEP2: split the channel')
        plt.axis('off')

        plt.subplot(1,4,3)
        plt.imshow(mito_mask, cmap='gray')
        plt.title('STEP3: convert to mask')
        plt.axis('off')

        plt.subplot(1,4,4)
        plt.imshow(label_mito)
        plt.title('STEP4: image thinning')
        plt.axis('off')
        plt.tight_layout()

        plt.show()


    def _adjustBrightness(self, gray, clip_hist_percent=10):
        # auto adjust image's brightness and contrast
        # Calculate grayscale histogram
        hist = cv2.calcHist([gray],[0],None,[256],[1,256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index -1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum/100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size -1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha
        auto_result = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

        return auto_result

    def _image_binary(self, image):
        # binary and conver to mask
        _, mask = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return mask

    def _image_skeletonize(self, mask):
        mask[mask==255] = 1
        # image closing
        # kernel = np.ones((3, 3), np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # image thinning
        # mito_skele = morphology.skeletonize(mask)
        mito_thin = morphology.thin(mask)

        return mito_thin

    def _mito_statistics(self, skele):
        # statistics the mitochondrial parameters:
        # 1. overall length
        # 2. mitochondrial numbers
        # 3. the length of different segment
        # 4. degree of polymerization

        labels = measure.label(skele.astype(int), connectivity=2)
        dst = color.label2rgb(labels)

        # print('regions number:', labels.max())
        properties = measure.regionprops(labels)
        valid_label = set()
        mitochondrial_each_segment_len = list()
        mitochondrial_overall_length = 0
        mitochondrial_each_seg_intersection_num = list()
        for prop in properties:
            if prop.area > 1:
                # mito number
                valid_label.add(prop.label)
                # each segment length
                mitochondrial_each_segment_len.append(prop.area)
                # total length
                mitochondrial_overall_length += prop.area
                # intersection number:
                label_copy = np.zeros(labels.shape, dtype=int)
                label_copy[labels==prop.label] = 1
                Intersections = self.getSkeletonIntersection(label_copy)
                mitochondrial_each_seg_intersection_num.append(len(Intersections))

        # mitochondrial total length:
        mitochondrial_overall_length = mitochondrial_overall_length * self.pixelsPerMetric
        mitochondrial_overall_length = round(mitochondrial_overall_length, 2)
        # mitochondrial each segment length:
        mitochondrial_each_segment_len = list(map(lambda x: float(x)*self.pixelsPerMetric, mitochondrial_each_segment_len))
        mitochondrial_each_segment_len = list(map(lambda x: round(x, 2), mitochondrial_each_segment_len))
        mitochondrial_numbers = len(valid_label)
        # mitochondrial each segment intersection number
        statistic_results = {'mitochondrial_numbers': mitochondrial_numbers,
                             'mitochondrial_overall_length(um)': mitochondrial_overall_length,
                             'mitochondrial_each_segment_len(um)': mitochondrial_each_segment_len,
                             "mitochondrial_each_seg_intersection_num": mitochondrial_each_seg_intersection_num}

        return statistic_results

    def getSkeletonIntersection(self, skeleton):
        # get Skeleton Images's Intersection
        """ Given a skeletonised image, it will give the coordinates of the intersections of the skeleton.

            Keyword arguments:
            skeleton -- the skeletonised image to detect the intersections of

            Returns:
            List of 2-tuples (x,y) containing the intersection coordinates
            """
        # A biiiiiig list of valid intersections             2 3 4
        # These are in the format shown to the right         1 C 5
        #                                                    8 7 6
        validIntersection = [[0, 1, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 0, 0],
                             [0, 1, 0, 0, 1, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 0, 1, 0],
                             [0, 1, 0, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0, 1],
                             [0, 1, 0, 1, 0, 0, 0, 1], [0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 1],
                             [1, 0, 1, 0, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0, 1, 0],
                             [1, 0, 0, 0, 1, 0, 1, 0], [1, 0, 0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 1, 1, 1],
                             [1, 1, 0, 0, 1, 0, 0, 1], [0, 1, 1, 1, 0, 0, 1, 0], [1, 0, 1, 1, 0, 0, 1, 0],
                             [1, 0, 1, 0, 0, 1, 1, 0], [1, 0, 1, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0, 1, 1],
                             [1, 1, 0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 1, 0],
                             [0, 0, 1, 0, 1, 0, 1, 1], [1, 0, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1, 0, 1],
                             [1, 0, 1, 0, 1, 1, 0, 0], [1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0, 1, 1],
                             [0, 1, 1, 0, 1, 0, 0, 1], [1, 1, 0, 1, 0, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0],
                             [0, 0, 1, 0, 1, 1, 0, 1], [1, 0, 1, 0, 0, 1, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0],
                             [1, 0, 1, 1, 0, 1, 0, 0]]
        image = skeleton.copy()
        intersections = list()
        for x in range(1, len(image) - 1):
            for y in range(1, len(image[x]) - 1):
                # If we have a white pixel(not 0 value)
                if image[x][y]:
                    neighbours = self.get_neighbours(x, y, image)
                    valid = True
                    if neighbours in validIntersection:
                        intersections.append((y, x))
        # Filter intersections to make sure we don't count them twice or ones that are very close together
        for point1 in intersections:
            for point2 in intersections:
                if (((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) < 5 ** 2) and (point1 != point2):
                    intersections.remove(point2)
        # Remove duplicates
        intersections = list(set(intersections))
        return intersections

    def get_neighbours(self, x, y, image):
        """Return 8-neighbours of image point P1(x,y), in a clockwise order"""
        img = image
        x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1;
        return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1], img[x1][y], img[x1][y_1], img[x][y_1],
                img[x_1][y_1]]

    def merge_measurements(self):
        # merge the mophological and mitochondrial measurements
        merge_table = []
        cell_nums = len(self.morphological_table)
        for i in range(cell_nums):
            merge_measure = self.morphological_table[i]
            merge_measure.update(self.mitochondrial_table[i])
            merge_table.append(merge_measure)
        return merge_table

    def _is_miss_mitochondrial(self, img, mask):
        # judge whether there are mitochondrial in the cell
        mask_int = mask.astype('uint8')
        mask_int[mask_int==1] = 255
        rhist = cv2.calcHist([img], [0], mask_int, [256], [0, 256])
        ghist = cv2.calcHist([img], [1], mask_int, [256], [0, 256])
        bhist = cv2.calcHist([img], [2], mask_int, [256], [0, 256])

        def weight_average(hist):
            weights = list(range(256))
            mean = sum([hist[i]*i for i in range(256)])/sum(hist)
            return mean
        rhist_mean = weight_average(rhist)
        ghist_mean = weight_average(ghist)
        bhist_mean = weight_average(bhist)

        if ghist_mean - (rhist_mean + bhist_mean) / 2 > 20:
            # Mitochondria in the cell are not lost
            return False
        else:
            # Mitochondria in the cell are lsot
            return True


if __name__=='__main__':

    data = np.load('../data.npz')
    image = data['arr_0']
    image_name = 'input_imgs/XY point5-Merge.png'
    results = data['arr_1'].tolist()

    cell_obj = CellProcess(image, results)
    cell_obj.cell_morphological_measure()
    cell_obj.cell_mitochondrial_measure()

    # folder create
    folder_name = os.path.basename(image_name).split('.')[0]
    output_dir = os.path.join('../output', folder_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # result save
    # 1. origin img save
    origin_img_name = os.path.join('../output', output_dir, 'origin.png')
    cv2.imwrite(origin_img_name, img=image)
    # 2. detect img save
    detect_img_name = os.path.join('../output', output_dir, 'detect.png')
    cv2.imwrite(detect_img_name, img=cell_obj.detect_img)
    # 3. cell morphological and mitochondrial infos save
    cell_infos = cell_obj.merge_measurements()
    cell_infos_name = os.path.join('../output', output_dir, 'cell_infos.csv')
    with open(cell_infos_name, 'w', newline='') as csvfile:
        fieldnames = list(cell_infos[0].keys())
        fieldnames = ['Cell_Id', 'Length', 'Width', 'mitochondrial_numbers',
                      'mitochondrial_overall_length', 'mitochondrial_each_segment_len']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cell_infos)



