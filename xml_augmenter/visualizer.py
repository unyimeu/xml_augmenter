import cv2 as cv
import math
import xml.etree.ElementTree as ET
import numpy as np

class XML_Augment():
    def __init__(self,annotation_path,image_path):
        self.image = cv.imread(image_path)
        #parser = ET.XMLParser(encoding="utf-8")
        self.tree = ET.parse(annotation_path)
        root = self.tree.getroot()
        self.xml = root

        
        #print(ETree.tostring(tree.getroot()))

    def visualize_annotaitons(self,an_color=(255, 0, 255),text_size=0.5,show_bbx=False):
        all_pts = []
        coords = []
        cnt = 0

        root = self.xml
        image = self.image.copy()


        for object in root.iter('object'):
            label = object.find('name').text
            
            for polygon in object.iter('polygon'):
                truthyy = list( polygon.iter())
                for coord in polygon.iter():
            
                    if cnt:
                        if len(coords) < 2:
                            point = int(float(coord.text))
                            coords.append(point)
                        if len(coords)==2:
                            all_pts.append(coords)
                            point = int(float(coord.text))
                            coords = []
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
        
        if show_bbx:
            coords1 = []
            coords2 = []
            all_pts = []
            cnt = 0
            for object in root.iter('object'):
                #label = object.find('name').text
                for polygon in object.iter('bndbox'):
                    for coord in polygon.iter():
                        if cnt:
                            if len(coords1) == len(coords2):
                                point = float(coord.text)
                                coords1.append(point)
                            else:
                                point = float(coord.text)
                                coords2.append(point)
                        cnt+=1

                    start_point = (int(coords1[0]), int(coords1[1])) 
    
                    # Ending coordinate, here (220, 220) 
                    # represents the bottom right corner of rectangle 
                    end_point = (int(coords2[0]), int(coords2[1])) 
                    
                    # Blue color in BGR 
                    color = (255, 0, 0) 
                    
                    # Line thickness of 2 px 
                    thickness = 1
                    
                    image = cv.rectangle(image, start_point, end_point, color, thickness) 
                    
                    coords1 = []
                    coords2 = []
                    cnt = 0
        cv.imshow('image',image)
        cv.waitKey()
        cv.destroyAllWindows()
        return image
    
    def brighten(self,max_bright):
        image = self.image.copy()
        brightness_val = np.random.randint(max_bright)
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
                            point = np.float32(coord.text)
                            coord.text = str(point+r1)
                            coords += 1
                        elif coords ==1:
                            point = np.float32(coord.text)
                            coord.text = str(point+r2)
                            coords = 0
                    cnt+=1
                coords = 0
                cnt = 0
        
        self.remove_out_of_bounds()
        self.new_bounding_boxes()
        return self.xml, self.image
    
    def rotate(self,max_rotation_offset=45,scale=1.4):

        image = self.image.copy()
        height, width = image.shape[:2]
        center = (width/2, height/2)
        angle = np.random.randint(-max_rotation_offset,max_rotation_offset)
        #scale = 1
        rotation_matrix = cv.getRotationMatrix2D(center, angle, scale)
        rotated_image = cv.warpAffine(image, rotation_matrix, (width, height))
        self.image = rotated_image
        
        #rotate the segmentationss
        cnt = 0
        coords = []
        all_pts = []
        for object in self.xml.iter('object'):
            for polygon in object.iter('polygon'):
                for coord in polygon.iter():
                    if cnt:
                        if len(coords) < 2:
                            point = np.float32(coord.text)
                            coords.append(point)
                        if len(coords)==2:
                            #do matrix operations here

                            a,b,c = rotation_matrix[0]
                            d, e ,f = rotation_matrix[1]

                            # calculate the dot product of the two matrices
                            x_new = a * coords[0] + b*coords[1]+c
                            y_new = d*coords[0] +e*coords[1] +f

                            all_pts.append(np.float32(x_new))
                            all_pts.append(np.float32(y_new))
                            point = np.float32(coord.text)
                            coords = []
                    cnt+=1
                cnt = 0
        
        cnt = 0
        coords = 0
        tracker = 0
        for object in self.xml.iter('object'):
            for polygon in object.iter('polygon'):
                for coord in polygon.iter():
                    if cnt:
                        try:
                            coord.text = str(all_pts[tracker])
                        except:
                            print('out of range!!!!!!!!')
                        tracker+=1
                    cnt+=1
                coords = 0
                cnt = 0
        
        self.remove_out_of_bounds()
        self.new_bounding_boxes()
        return self.xml, rotated_image
    
    def remove_out_of_bounds(self):
        width = int(self.xml.find('size/width').text)
        height = int(self.xml.find('size/height').text)
        inbounds_cnt = 0
        for object in self.xml.findall('object'):

            #polygons_to_remove = []
            for polygon in object.findall('polygon'):
                coords = list(polygon)
                #print('coords--->',coords)
                all_out_of_bounds = True
                for i in range(0, len(coords), 2):
                    x = float(coords[i].text)
                    y = float(coords[i+1].text)
                    if 0 <= x <= width and 0 <= y <= height:
                        all_out_of_bounds = False
                        inbounds_cnt+=1
                    else:
                        polygon.remove(coords[i])
                        polygon.remove(coords[i+1])
            if all_out_of_bounds or inbounds_cnt<5:
                self.xml.remove(object)
            inbounds_cnt = 0
            

    def new_bounding_boxes(self):
        x_coords = []
        y_coords = []
        all_pts = []
        coords = []
        cnt = 0

        root = self.xml

        for object in root.iter('object'):
            label = object.find('name').text
            for polygon in object.iter('polygon'):
                for coord in polygon.iter():
                    if cnt:
                        if len(coords) < 2:
                            point = int(float(coord.text))
                            coords.append(point)
                        if len(coords)==2:
                            all_pts.append(coords)
                            point = int(float(coord.text))
                            x_coords.append(coords[0])
                            y_coords.append(coords[1])
                            coords = []
                    cnt+=1
                coords = []
                cnt = 0
            

            for box in object.iter('bndbox'):
                #get new bounding coordinates
                xmin = math.floor(min(x_coords))  
                xmax = math.ceil(max(x_coords))   
                ymin = math.floor(min(y_coords)) 
                ymax = math.ceil(max(y_coords))
                
                box.find('xmin').text = str(int(xmin))
                box.find('xmax').text = str(int(xmax))
                box.find('ymin').text = str(int(ymin))
                box.find('ymax').text = str(int(ymax)) 
            x_coords=[]
            y_coords=[]
        return None

        
        
        
        
'''images_output_directory = "C:\\Users\\Unyim\\Downloads\\CSDownloads\\augment_xml\\"
annotations_output_directory = "C:\\Users\\Unyim\\Downloads\\CSDownloads\\augment_xml\\"
cnter= 43
for i in ['01', '02', '03', '05', '06', '07', '09', '10', '11', '12', '13', '14', '15', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42']:

    
    impath = "images\\" +i+".jpg"
    anpath = "annotations\\"+i+".xml"
    for j in range(30):
        xa = XML_Augment(anpath,impath)
        xa.translate(0.2)
        xa.brighten(50)
        xa.rotate(20)
        one = "images\\" +str(cnter)+".jpg"
        two = "annotations\\"+str(cnter)+".xml"
        print(cnter)
        print('**********')
        save_location1 = images_output_directory+  one
        save_location2 = annotations_output_directory +two
        cv.imwrite(save_location1, xa.image)


        xml_str = ET.tostring(xa.xml, encoding='utf-8').decode('utf-8')
        with open(save_location2, 'w', encoding='utf-8') as file:
            file.write(xml_str)
        #tree = ET.ElementTree(xa.xml)
        #tree.write(save_location2, encoding="utf-8", xml_declaration=True)
        cnter+=1'''
        





'''#xa.visualize_annotaitons(text_size=0.3)
#xa.visualize_annotaitons(text_size=0.3)
for i in np.random.randint(100,800,1):
    i=str(i)
    print(i)
    impath = "images\\" +i+".jpg"
    anpath = "annotations\\"+i+".xml"
        
    xa = XML_Augment(anpath,impath)
    xa.visualize_annotaitons(text_size=0.3,show_bbx=True,an_color=(0,0,255))'''

'''i = "01"
impath = "images\\" +i+".jpg"
anpath = "annotations\\"+i+".xml"
    
xa = XML_Augment(anpath,impath)
xa.visualize_annotaitons(text_size=0.3,show_bbx=True,an_color=(0,0,255))'''


'''cv.imshow('image',xa.image)
cv.waitKey()
cv.destroyAllWindows()'''

