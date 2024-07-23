import cv2 as cv
import xml.etree.ElementTree as ET
import numpy as np

class XML_Augment():
    def __init__(self,annotation_path,image_path):
        self.image = cv.imread(image_path)
        self.tree = ET.parse(annotation_path)
        root = self.tree.getroot()
        self.xml = root

    def visualize_annotaitons(self,an_color=(255, 0, 255),text_size=0.5):
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
                    text_size, color, thickness, cv.LINE_AA) 
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
        image = self.image.copy()
        brightness_val = np.random.randint(max_bright)
        print(brightness_val)
        image = cv.addWeighted(image, 1, np.zeros(image.shape, image.dtype), 0, brightness_val)
        self.image = image.copy()
        return self.xml, image
    
    def translate(self,max_translaiton_percent):
        image = self.image.copy()
        tot_y,tot_x = image.shape[:2]
        max_y,max_x = tot_y*max_translaiton_percent,tot_x*max_translaiton_percent
        r1,r2 = np.random.randint(-max_y,max_y),np.random.randint(-max_x,max_x)
        M = np.float32([[1, 0, r1], [0, 1, r2]])
        shifted = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
        self.image = shifted.copy()

        #translate the segmentaiton annotations
        cnt = 0
        coords = 0
        for object in self.xml.iter('object'):
            for polygon in object.iter('polygon'):
                for coord in polygon.iter():
                    if cnt:
                        if coords < 1:
                            point = float(coord.text)
                            coord.text = str(point+r1)
                            coords += 1
                        elif coords ==1:
                            point = float(coord.text)
                            coord.text = str(point+r2)
                            coords = 0
                    cnt+=1
                coords = 0
                cnt = 0
        
        #translate the bounding boxes!
        cnt = 0
        coords = 0
        for object in self.xml.iter('object'):
            for polygon in object.iter('bndbox'):
                for coord in polygon.iter():
                    if cnt:
                        if coords < 2:
                            point = float(coord.text)
                            coord.text = str(point+r1)
                            coords += 1
                        elif coords >= 2:
                            point = float(coord.text)
                            coord.text = str(point+r2)
                            coords +=1
                    cnt+=1
                coords = 0
                cnt = 0

        return self.xml, self.image
    
    def rotate(self,max_rotation_offset=45):

        image = self.image.copy()
        height, width = image.shape[:2]
        center = (width/2, height/2)
        angle = np.random.randint(-max_rotation_offset,max_rotation_offset)
        scale = 1
        rotation_matrix = cv.getRotationMatrix2D(center, angle, scale)
        rotated_image = cv.warpAffine(image, rotation_matrix, (width, height))
        print(rotation_matrix)
        self.image = rotated_image
        

        for object in self.xml.iter('object'):
            label = object.find('name').text
            for polygon in object.iter('polygon'):
                for coord in polygon.iter():
                    if cnt:
                        if len(coords) < 2:
                            point = int(float(coord.text))
                            coords.append(point)
                        else:
                            
                            #do matrix operations here
                            point = int(float(coord.text))
                            coords = [point]
                    cnt+=1

                all_pts = []
                coords = []
                cnt = 0
        return self.xml, rotated_image


'''impath = "images\\01.jpg"
anpath = "annotations\\01.xml"
output_directory = ""
xa = XML_Augment(anpath,impath)


im_test = "annotations\TESTING.xml"
tree = ET.parse(im_test)
root = tree.getroot()

#xa.visualize_annotaitons(text_size=0.3)
xa.visualize_annotaitons(text_size=0.3)
xa.translate(0.2)
xa.brighten(100)
xa.rotate(45)
xa.visualize_annotaitons(text_size=0.3)

'cv.imshow('image',xa.image)
cv.waitKey()
cv.destroyAllWindows()'''

