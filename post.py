#encoding:utf-8
import numpy as np
import cv2


def read_img(img_path):
    image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    ret,image = cv2.threshold(image,50,255,cv2.THRESH_BINARY)
    
    return image


def find_contours(img):
    #img is two-value image
    cvt_img = img
    binary,contours,hierarchy = cv2.findContours(cvt_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #print('contours:',len(contours))
    return contours
    #cv2.drawContours(img,contours,-1,(0,0,255),3)

def find_rectangles(contours):
    '''
    rect = cv2.minAreaRect(cnt)  
      
    box = np.int0(box)  
    cv2.drawContours(im, [box], 0, (255, 0, 255), 2)
    '''
    rec_t = []
    
    for i in contours:
        #print('contours',i)
        rec_t_tmp = cv2.minAreaRect(i)
        box_tmp = cv2.boxPoints(rec_t_tmp)
        rec_t.append(np.int0(box_tmp))
        #print('np.int0(box_tmp)',np.int0(box_tmp))
    return rec_t


def draw_box(im,box,mode = True):
    for i in box:
        if mode == True:
            #print(im)
            cv2.drawContours(im, [i], -1, (255, 0, 0), 3)
        else:
            cv2.drawContours(im,[i],-1,(0,0,0),-1)
    return im

def get_contours_area(contours):
    #计算轮廓所包含的面积
    #area = cv2.contourArea(cnt)
    area = []
    for i in contours:
        #print('i',i)
        area.append(cv2.contourArea(np.array(i)))
    return area

def run():
    img = read_img('./0.png')
    contours = find_contours(img)
    rec = find_rectangles(contours)
    area = get_contours_area(contours)
    #print(area)
    for index,i in enumerate(rec):
        if area[index] > 2000:
            draw_box(img,[i],mode=True)
            print('less 100')
        else:
            draw_box(img,[i],mode=False)
    cv2.namedWindow('res',cv2.WINDOW_NORMAL)
    cv2.imshow('res',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def lvbo(image,draw,yuzhi=3000):
    contours = find_contours(image)
    rec = find_rectangles(contours)

    area = get_contours_area(contours)
    con_draw_true = []
    con_draw_false = []
    for index,i in enumerate(rec):
        #print('area',area[index])
        if area[index] < yuzhi:
            con_draw_false.append(contours[index])
        else:
            
            [vx,vy,x,y] = cv2.fitLine(contours[index], cv2.DIST_L2,0,0.01,0.01)
            max_x = np.max(np.squeeze(contours[index]),axis=0)[0]
            min_x = np.min(np.squeeze(contours[index]),axis=0)[0]
            length_con = (max_x - min_x) * np.sqrt(1 + np.square(vy / vx))
            #print('length_con',length_con)
            if length_con < 400:
                con_draw_false.append(contours[index])
            else:
                con_draw_true.append(rec[index])
                #print(i)
    image = draw_box(image,con_draw_false,mode=False)
    draw = draw_box(draw,con_draw_true,mode=True)
    image = draw_box(image,con_draw_true,mode=True)
    print(len(con_draw_false),len(con_draw_true))
    
#    if length_con < 300:
#                 image = draw_box(image,[i],mode=False)
#             else:
#                 image = draw_box(image,[i],mode=True)
#                 draw = draw_box(draw,[i],mode=True)
    
    return (image,draw)

def get_contours_afterLV(image_pred,yuzhi = 3000):
    con = []
    con1 = []
    contours = find_contours(image_pred)
    rec = find_rectangles(contours)
    area = get_contours_area(contours)
    for index,i in enumerate(rec):
        
        if area[index] > yuzhi:
            con.append(contours[index])
            print('area',area[index])
        else:
            pass
    for indexj,j in enumerate(con):
        [vx,vy,x,y] = cv2.fitLine(j[0], cv2.DIST_L2,0,0.01,0.01)
        max_x = np.max(np.squeeze(j),axis=0)[0]
        min_x = np.min(np.squeeze(j),axis=0)[0]
        length_con = (max_x - min_x) * np.sqrt(1 + np.square(vy / vx))
        print('length_con',length_con)
        if length_con > 400:
            con1.append(j)
    
    return con1

def get_y_true_contours_center(y_ture):
    c = find_contours(y_ture)
    #print(c)
    point = []
    for cc in c:
        M = cv2.moments(cc)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        point.append((cX,cY))
    return point

def is_point_in_conts(point,conts):
    for cont in conts:
        if cv2.pointPolygonTest(cont,point,False) == 1:
            return True
    return False
def get_iou_by_box(y_true,y_pred):
    cont_pred = get_contours_afterLV(y_pred)
    #cont_true = get_contours_afterLV(y_true)
    cont_true = find_contours(y_true)
    rec1 = find_rectangles(cont_pred)
    rec2 = find_rectangles(cont_true)
    countT = len(rec1)
    count_true = 0 
    for i in rec1:
        for j in rec2:
            #print(np.array([i]))
            im = np.zeros([2048,2592], dtype = "uint8")
            im1 =np.zeros([2048,2592], dtype = "uint8")
            original_grasp_mask = cv2.fillPoly(im, np.array([i]), 255)
            prediction_grasp_mask = cv2.fillPoly(im1,np.array([j]),255)
            masked_and = cv2.bitwise_and(original_grasp_mask,prediction_grasp_mask , mask=im)
            masked_or = cv2.bitwise_or(original_grasp_mask,prediction_grasp_mask,mask=im1)

            or_area = np.sum(np.float32(np.greater(masked_or,0)))
            and_area =np.sum(np.float32(np.greater(masked_and,0)))
            IOU = and_area / or_area
            print('iou',IOU)
            if IOU >0.7:
                count_true += 1
    if countT == 0:
        return(0,0,len(rec2))
    return (count_true/countT,countT,len(rec2))
def mean_iou_by_box(y_true,y_pred):
    true_count = 0
    total_count = 0
    for index,i in enumerate(y_true):
        ture_rate,count = get_iou_by_box(i,y_pred[index])
        true_count += count * ture_rate
        total_count += count
    return true_count * 1.0 /total_count



