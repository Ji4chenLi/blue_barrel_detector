'''
ECE276A WI19 HW1
Blue Barrel Detector
'''

import cv2
import numpy as np
import pickle
import os, cv2
from skimage.measure import label, regionprops
from scipy import ndimage
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

def pre_process_img(img):

    img_mean = np.sum(img,axis=2)
    for i in range(img_mean.shape[0]):
        for j in range(img_mean.shape[1]):
            img_mean[i,j] = max(img_mean[i,j],1)
    img_mean = 1 / img_mean
    normal_img = np.zeros(img.shape)
    for i_layer in range(3):
        normal_img[:,:,i_layer] = img[:,:,i_layer] * img_mean * 100
    return normal_img


class Classifier(object):
    def __init__(self,N_class,N_channel):
        self.N_class = N_class
        self.N_channel = N_channel
        self.prior = np.zeros(N_class)
        self.count = np.zeros(N_class).astype(int)
        self.mean,self.var,self.sec_moment  = [],[],[]
        for _ in range(N_class):
            self.mean.append(np.zeros(N_channel).reshape(N_channel,1))
            self.sec_moment.append(np.zeros([N_channel,N_channel]))
            self.var.append(np.zeros([N_channel,N_channel]))

    def observe(self,x,x_class):
        last_mean = self.mean[x_class].copy()
        last_count = self.count[x_class].copy()
        last_sec_moment = self.sec_moment[x_class].copy()
        len_input = x.shape[1]
        mean_input = np.mean(x, axis=1).reshape(3,1)
        sec_moment_input = np.zeros([self.N_channel,self.N_channel])

        for i in range(len_input):
            x_col = x[:,i]
            sec_cur = np.matmul(x_col.reshape(self.N_channel,1),x_col.reshape(1,self.N_channel))
            sec_moment_input += sec_cur
        sec_moment_input = sec_moment_input / len_input

        self.count[x_class] += len_input
        self.mean[x_class] = (mean_input * len_input + last_mean * last_count) / (self.count[x_class])
        self.sec_moment[x_class] = (sec_moment_input * len_input + last_sec_moment * last_count) / (self.count[x_class])
        self.var[x_class] = self.sec_moment[x_class] - np.matmul(self.mean[x_class].reshape(self.N_channel,1),self.mean[x_class].reshape(1, self.N_channel))

        for i_class in range(self.N_class):
            self.prior[i_class] = self.count[i_class] / sum(self.count)

    def classify(self,input):
        input_length = input.shape[1]
        score = []
        log_value = [np.log(max(np.linalg.det(self.var[i]), 0.0001)) for i in range(self.N_class)]
        pro_value = [2 * np.log(self.prior[i]) for i in range(self.N_class)]

        for i in range(self.N_class):
            t_score = np.diag(np.matmul(np.matmul((input - self.mean[i]).T  , np.linalg.inv(self.var[i])),(input - self.mean[i]))).reshape(input_length,1) + log_value[i] -  pro_value[i]
            score.append(t_score)
        score = np.concatenate((score[0],score[1]),axis=1)
        best_fit = np.argmin(score,axis=1)
        return best_fit

    def get_para(self):
        return {'mean': self.mean, 'variance': self.var, 'prior': self.prior, 'count': self.count}


class BarrelDetector():
    def __init__(self, classifier):
        '''
            Initilize your blue barrel detector with the attributes you need
            eg. parameters of your classifier
        '''
        self.classifier = classifier

    def segment_image(self, img):
        '''
            Calculate the segmented image using a classifier
            eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
            call other functions in this class if needed

            Inputs:
                img - original image
            Outputs:
                mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
        '''
        # YOUR CODE HERE
        ROW = img.shape[0]
        COL = img.shape[1]
        img_mask = np.zeros([ROW,COL]).astype(int)
        normal_img = pre_process_img(img)

        for row in range(ROW):
            for col in range(COL):
                x = normal_img[row, col, :].reshape(3, 1)
                result_class = self.classifier.classify(x)
                img_mask[row, col] = 1 - result_class[0]
        return img_mask


    def get_bounding_box(self, img):
        '''
            Find the bounding box of the blue barrel
            call other functions in this class if needed

            Inputs:
                img - original image
            Outputs:
                boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2]
                where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
                is from left to right in the image.

            Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
        '''
        # YOUR CODE HERE
        img_mask = self.segment_image(img)
        open_square = ndimage.binary_opening(img_mask, structure=np.ones((5, 5)))

        # apply threshold
        generate_mask = open_square
        thresh = threshold_otsu(generate_mask)
        bw = closing(generate_mask > thresh, square(3))
        # label image regions
        label_image = label(bw)
        boxes = []
        for region in regionprops(label_image):
            # take regions with large enough areas
            if region.area >= 100:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                rec_square = (maxr - minr) * (maxc - minc)
                bw_float = label_image[minr: maxr, minc: maxc] > 0
                ratio = np.sum(bw_float) / rec_square
                print(ratio)
                if ratio > 0.5 and 1.2 < (maxr - minr) / (maxc - minc) < 2.5:
                    boxes.append([minr, minc, maxr, maxc])
        return boxes


if __name__ == '__main__':
    folder = "trainset"
    with open('classifier.pkl', 'rb') as f:
        stat_into = os.stat('classifier.pkl')
        if stat_into.st_size == 0:
            print('MAKE CLASSIFIER!')
            classifier = Classifier(N_class, N_channel)
        else:
            print('LOAD CLASSIFIER!')
            classifier = pickle.load(f)
        f.close()

    my_detector = BarrelDetector(classifier)
    for filename in os.listdir(folder):
        # read one test image
        img = cv2.imread(os.path.join(folder,filename))
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #Display results:
        #(1) Segmented images
        #	 mask_img = my_detector.segment_image(img)
        #(2) Barrel bounding box
        #    boxes = my_detector.get_bounding_box(img)
        #The autograder checks your answers to the functions segment_image() and get_bounding_box()
        #Make sure your code runs as expected on the testset before submitting to Gradescope

