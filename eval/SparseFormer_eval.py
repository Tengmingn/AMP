import os
import numpy as np
from PIL import Image
import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='SASformer evaluation')
    parser.add_argument('--folder_path', type=str, required=True)
    args = parser.parse_args()
    return args

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,)*2)
 
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() /  self.confusionMatrix.sum()
        return acc
 
    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc
 
    def userAccuracy(self):
        # ua = (TP) / TP + FN
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        return classAcc

    def meanUserAccuracy(self):
        classAcc = self.userAccuracy()
        meanAcc = np.nanmean(classAcc) 
        return meanAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc) 
        return meanAcc 
    
    def meanF1score(self):
        precision = self.classPixelAccuracy()
        recall = self.userAccuracy()
        f1_per_class = 2 * precision * recall / (precision + recall)
        meanF1 = np.nanmean(f1_per_class)
        return meanF1
 
    def KappaConsistency(self):
        # Po = OA
        # Pe = ((TN+FN)(TN+FP)+(FP+TP)(FN+TP))/(TP+FN+FP+TN)^2
        Po = self.pixelAccuracy()
        row_sums = self.confusionMatrix.sum(axis=1)
        col_sums = self.confusionMatrix.sum(axis=0)
        all_sums_sq = self.confusionMatrix.sum()*self.confusionMatrix.sum()
        Pe = (row_sums[1]*col_sums[1]+row_sums[0]*col_sums[0])/all_sums_sq
        Kappa = (Po-Pe)/(1-Pe)
        return Kappa

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix) 
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix) 
        IoU = intersection / union  
        mIoU = np.nanmean(IoU) 
        return mIoU
 
    def genConfusionMatrix(self, imgPredict, imgLabel): 
        # remove classes from unlabeled pixels in gt image and predict
        # imgPredict = imgPredict - 1
        # imgLabel = imgLabel - 1
        # print(np.unique(imgLabel))
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix
 
    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU
 
 
    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)
 
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))
 
    def evaluate_folder(self, folder_path, gt_folder_path):
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            gt_name = gtname = filename[:4]+'_mask.png'
            pred_path = os.path.join(folder_path, filename)
            gt_path = os.path.join(gt_folder_path, gt_name)
            pred_img = np.array(Image.open(pred_path))
            gt_img = np.array(Image.open(gt_path))
            assert pred_img.shape == gt_img.shape, f"Shape mismatch: {pred_img.shape} vs {gt_img.shape}"
            self.addBatch(pred_img, gt_img)
        print('OA is : %f' % self.pixelAccuracy())
        print('PA is :', self.classPixelAccuracy())
        print('MPA is : %f' % self.meanPixelAccuracy())
        print('UA is :', self.userAccuracy())
        print('MUA is : %f' % self.meanUserAccuracy())
        print('mF1 is : %f' % self.meanF1score())
        print('Kappa is : %f' % self.KappaConsistency())
        print('mIoU is : %f' % self.meanIntersectionOverUnion())

        accfilename = "SparseFormer_acc_"+folder_path[-6:]
        with open(accfilename, "w") as file:
            file.write('Model info is :')
            file.write(folder_path)
            file.write('\n OA is : %f' % self.pixelAccuracy())
            file.write('\n PA is : {}\n'.format(self.classPixelAccuracy()))
            file.write(' MPA is : %f' % self.meanPixelAccuracy())
            file.write('\n UA is : {}\n'.format(self.userAccuracy()))
            file.write(' MUA is : %f' % self.meanUserAccuracy())
            file.write(' mF1 is : %f' % self.meanF1score())
            file.write('\n Kappa is : %f' % self.KappaConsistency())
            file.write('\n mIoU is : %f' % self.meanIntersectionOverUnion())

def main(args):
    folder_path = args.folder_path
    gt_folder_path = "/media/llog/Mydata/Tree_canopy/testdata/labels_v2/"
    num_classes = 2  
    metric = SegmentationMetric(num_classes)
    metric.evaluate_folder(folder_path, gt_folder_path)

if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
