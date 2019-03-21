import cv2
import os
import numpy as np


# version 1.2

##### Swich parameter
model = 'video'
# model = 'camera'
# algorithm = ['knn', 'mog2', 'mog2_morph', 'background']
# algorithm = ['mog2_morph']
algorithm = ['mog2_morph', 'background']

expand = 0
standardValue = 150
path = './dataset/image/camera.avi'
generateData = True


#
def modelSelection(model, path4Dict=''):
    # model selection
    if model == 'camera':
        input = cv2.VideoCapture(0) 
        path = ''
    elif model == 'video':
        #read a dictionary for automatically load data
        # ????????????????????????????????????????????????????????????????????

        # path = './dataset/walking/person01_walking_d1_uncomp.avi'
        # path = './dataset/walking/person01_walking_d2_uncomp.avi'
        # path = './dataset/walking/person01_walking_d4_uncomp.avi'
        # path = './dataset/running/person15_running_d1_uncomp.avi'
        # path = './dataset/handclapping/person15_handclapping_d1_uncomp.avi'
        path = './dataset/handwaving/person15_handwaving_d1_uncomp.avi'

        #
        input = cv2.VideoCapture(path)
    else:
        print('Undefinition...')

    # input validate
    if (not input.isOpened):
        print("Video file or Camera is not opened!")
        return 0
    else:
        width=int(input.get(cv2.CAP_PROP_FRAME_WIDTH))
        height=int(input.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps=input.get(cv2.CAP_PROP_FPS)
        print(fps,width,height)

    return input, width, height, fps, path

#
def save2Image(image, path, index, count):

    name_img = path.split('/')[-1].split('.')[0]
    path4Img = './dataset/image/' + name_img + '/image_' + str(index) + '_' + str(count) + '.png'

    if not os.path.exists('./dataset/image/' + name_img):
        os.mkdir('./dataset/image/' + name_img)

    cv2.imwrite(path4Img, image)

def save2File(image, path, index, count):
    name_file = path.split('/')[-1].split('.')[0]
    path4File = './dataset/trainData/' + name_file + '/data_' + str(index) + '_' + str(count) + '.txt'

    if not os.path.exists('./dataset/trainData/' + name_file):
        os.mkdir('./dataset/trainData/' + name_file)

    np.savetxt(path4File, image, fmt='%.0d', delimiter=',')

#
def standardization(image):
    # width = heigh
    if image.shape[0] > image.shape[1]:
        borderSize = int((image.shape[0] - image.shape[1]) / 2)
        if image.shape[0] != image.shape[1] + 2 * borderSize:
            newImage = cv2.copyMakeBorder(image,0,0,borderSize,borderSize+1, cv2.BORDER_CONSTANT,value=[0,0,0])
        else:
            newImage = cv2.copyMakeBorder(image,0,0,borderSize,borderSize, cv2.BORDER_CONSTANT,value=[0,0,0])

    else:
        borderSize = int((image.shape[1] - image.shape[0]) / 2)
        if image.shape[0] + 2 * borderSize != image.shape[1]:
            newImage = cv2.copyMakeBorder(image,borderSize,borderSize+1,0,0, cv2.BORDER_CONSTANT,value=[0,0,0])
        else:
            newImage = cv2.copyMakeBorder(image,borderSize,borderSize,0,0, cv2.BORDER_CONSTANT,value=[0,0,0])

    # zoom
    dim = (standardValue, standardValue)
    resizeImage = cv2.resize(newImage, dim, interpolation=cv2.INTER_AREA)
    
    return resizeImage

# KNN
def knnbs()
    


def backgroundSubtractor(input, width, height, fps, path):
    #
    cv2.namedWindow('Input video')
    cv2.namedWindow('Detection')
    # cv2.namedWindow('debug')

    flag_knn = False
    flag_mog2 = False
    if 'knn' in algorithm:
        cv2.namedWindow('KNN')
        flag_knn = True
    if 'mog2' in algorithm:
        cv2.namedWindow('MOG2')
        flag_mog2 = True
    if 'mog2_morph' in algorithm:
        cv2.namedWindow('MOG2_MORPH')
        flag_mog2 = True
    if 'background' in algorithm:
        cv2.namedWindow('Background')
        flag_knn = True

    #
    # history = 20
    if flag_knn:
        bsmaskKnn = np.zeros([height,width],np.uint8)
        PKNN = cv2.createBackgroundSubtractorKNN(detectShadows=False)
        # PKNN.setHistory(history)
    if flag_mog2:
        bsmaskMOG2 = np.zeros([height,width],np.uint8)
        pMOG2 = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        # pMOG2.setHistory(history)
        #morphological processing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))  ###### MORPH_ELLIPSE

    #
    index_img = 0
    while input.isOpened:
        (flag, frame)=input.read()
        #
        if not flag:
            break
        
        #
        cv2.imshow('Input video', frame)

        if flag_knn:
            bsmaskKnn= PKNN.apply(frame)
            # # 对原始帧进行膨胀去噪
            # th = cv2.threshold(bsmaskKnn.copy(), 244, 255, cv2.THRESH_BINARY)[1]
            # th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
            # bsmaskKnn = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
            background = PKNN.getBackgroundImage()
            if 'knn' in algorithm:
                cv2.imshow('KNN', bsmaskKnn)
            if 'background' in algorithm:
                cv2.imshow('Background', background)
        
        if flag_mog2:
            bsmaskMOG2 = pMOG2.apply(frame)


            bsmaskMOG2_MORPH=cv2.morphologyEx(bsmaskMOG2,cv2.MORPH_OPEN,kernel)

            #
            contours= cv2.findContours(bsmaskMOG2_MORPH, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
            image_bsmaskMOG2_MORPH = bsmaskMOG2_MORPH.copy()

            #
            count = 0
            save2Image(bsmaskMOG2_MORPH, path, 1000 + index_img, count)

            for target in contours: ### 
                x, y, w, h = cv2.boundingRect(target)  ### 
                area = cv2.contourArea(target)            
                if 500 < area < 3000:                
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(image_bsmaskMOG2_MORPH, (x, y), (x + w, y + h), (255, 255, 0), 2)    
  
                    # save frame
                    if generateData:
                        #save2Image(bsmaskMOG2_MORPH, path, index_img)
                        #save2Image(bsmaskMOG2_MORPH[y-expand:y+h+expand, x-expand:x+w+expand], path, index_img, count)
                        feature = standardization(bsmaskMOG2_MORPH[y-expand:y+h+expand, x-expand:x+w+expand])
                        save2Image(feature, path, index_img, count)
                        save2File(feature, path, index_img, count)

                    count += 1

            if 'mog2' in algorithm:
                cv2.imshow('MOG2',bsmaskMOG2)
            if 'mog2_morph' in algorithm:
                cv2.imshow("Detection", frame)        
                cv2.imshow("MOG2_MORPH", image_bsmaskMOG2_MORPH) 

      
        
        index_img += 1

        #
        c = cv2.waitKey(40)
        if c==27:
            break

    input.release()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    #
    input, width, height, fps, path = modelSelection(model)
    backgroundSubtractor(input, width, height, fps, path)

