import cv2 as cv
import xml.etree.ElementTree as ET
import numpy as np

class XML_Augment():
    def __init__(self,annotation_path,image_path):
        self.image = cv.imread(image_path)
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        self.xml = root

    def visualize_annotaitons(self,an_color=(255, 0, 255)):
        all_pts = []
        coords = []
        cnt = 0

        root = self.xml
        image = self.image.copy()
        for object in root.iter('object'):
            label = object.find('name').text
            for polygon in object.iter('polygon'):
                for coord in polygon.iter():
                    if cnt:
                        if len(coords) < 2:
                            point = int(float(coord.text))
                            coords.append(point)
                        else:
                            all_pts.append(coords)
                            point = int(float(coord.text))
                            coords = [point]
                    cnt+=1

                pts_unshape = np.array(all_pts, np.int32)
                pts = pts_unshape.reshape((-1, 1, 2))
                isClosed = True
                
                color = an_color
                thickness = 1

                cv.putText(image, label, pts_unshape[-1], cv.FONT_HERSHEY_SIMPLEX,  
                    0.5, color, thickness, cv.LINE_AA) 
                image = cv.polylines(image, [pts], 
                                    isClosed, color, 
                                    thickness)
                all_pts = []
                coords = []
                cnt = 0
        cv.imshow('image',image)
        cv.waitKey()
        cv.destroyAllWindows()
        return image
    
    def brighten(self,max_bright):
        brightness_val = np.random.randint(max_bright)
        print(brightness_val)
        image = self.image.copy()
        image = cv.addWeighted(image, 1, np.zeros(image.shape, image.dtype), 0, brightness_val)
        return self.xml, image

'''impath = "images\\01.jpg"
anpath = "annotations\\01.xml"

xa = XML_Augment(anpath,impath)

xa.visualize_annotaitons()
an,im = xa.brighten(100)
cv.imshow('image',im)
cv.waitKey()
cv.destroyAllWindows()'''

