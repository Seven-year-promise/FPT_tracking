import os
import numpy as np 
import cv2
import tensorflow as tf
from numpy.random import uniform, randn
import time

from particle_filters_VGG import roi, particle_filter_VGG_1, particle_filter_VGG_2, particle_filter_VGG_3, particle_filter_VGG_4
from faster_part.part_demo import Faster_RCNN as Faster_part
from faster_full.faster_demo import Faster_RCNN as Faster_full
    
cwd = os.getcwd()    

INITIAL_FRAME_NUM = 5
INITIAL_TRACKER_NUM = 3
PROB_THRESHOLD = 0.9
SCALE_THRESHOLD = 5

EVAL = 0

def create_uniform_particles(x_range, y_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], N)
    particles[:, 1] = uniform(y_range[0], y_range[1], N)
    particles[:, 2] = np.ones((N), dtype = np.float32)/N
    return particles
    
def create_Gaussian_particles(mean, std, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N)*std[0])
    particles[:, 1] = mean[1] + (randn(N)*std[1])
    particles[:, 2] = np.ones((N), dtype = np.float32)/N
    return particles
    
def get_initial_localization(img, img_size, Faster_RCNN_ins):
    img_height = img_size[1]
    img_width = img_size[0]
    ratioX = img_width*1.0/512
    ratioY = img_height*1.0/512
    ratio_inf = [ratioY, ratioX]
    objects, feature_map, rpn_bbox, rpn_scores, scales = Faster_RCNN_ins.Faster_run(image = img, is_init=True)
    #print "#=================output information==========================#"
    #print objects, feature_map, rpn_scores.shape
    #print scales
    #if len(objects)==0:
        #print img
    localizations = []
    for ind, obj in enumerate(objects):
        localizations.append({'name':obj['name'],
                             'centerX':obj['boxes'][0], 
                             'centerY':obj['boxes'][1], 
                             'height':obj['boxes'][3], 
                             'width':obj['boxes'][2]})
    return localizations

def initial_tracker(first_trackers, threshold=5):
    roi_threshold = 0.01
    object_n_v = {}#the names and votes for each object
    final_tracker = {}
    bounding_boxes = {}
    
    for tracker_ind, first_tracker in enumerate(first_trackers):
        for object_tracker in first_tracker:
            object_name = object_tracker['name']
            box_X = object_tracker['centerX']
            box_Y = object_tracker['centerY']
            box_height = object_tracker['height']
            box_width = object_tracker['width']
            bbox = np.array([box_X, box_Y, box_height, box_width])
            box_xmin = box_X - box_width/2.0
            box_ymin = box_Y - box_height/2.0
            box_xmax = box_X + box_width/2.0
            box_ymax = box_Y + box_height/2.0
            box = np.array([box_xmin, box_ymin, box_xmax, box_ymax])
            if object_n_v.has_key(object_name):
                if roi(box, final_tracker[object_name])>roi_threshold:
                    object_n_v[object_name] = object_n_v[object_name]+1
                    final_tracker[object_name] = box
                    bounding_boxes[object_name] = bbox
            else:
                object_n_v.update({object_name:1})
                final_tracker.update({object_name:box})
                bounding_boxes.update({object_name:bbox})
    initial_trackers = []
    for name, vote in object_n_v.items():
        if vote > threshold:
            mu_x = bounding_boxes[name][0]
            mu_y = bounding_boxes[name][1]
            init_particles = create_Gaussian_particles(mean=[mu_x, mu_y], std=[50.0, 50.0],  N=100)
            #print init_particles, mu_x, mu_y
            object_candidate = {'name':name,
                                'score': 1.0,
                                'centerX':mu_x,
                                'centerY':mu_y, 
                                'height':bounding_boxes[name][2], 
                                'width':bounding_boxes[name][3], 
                                'particles':init_particles}
            initial_trackers.append(object_candidate)
    
    #print "#=================output information==========================#"
    #print objects, feature_map, rpn_scores.shape
    #print scales
    
    return initial_trackers

def draw_points(img, particles):
    N = len(particles)
    for i in range(N):
        Cx = particles[i, 0]
        Cy = particles[i, 1]
        weight = particles[i, 2]
        img = cv2.rectangle(img, (int(Cx-1),int(Cy-1)), (int(Cx),int(Cy)), (0,0, 255*weight), 2)
    return img
    
def draw_boxes(img, img_size, boxes, gt_box=[]):
    for ind, box in enumerate(boxes):
        #print box
        if box.has_key('particles'):
            img = draw_points(img, box['particles'])
        box_X = box['centerX']
        box_Y = box['centerY']
        box_height = box['height']
        box_width = box['width']
        img_height = img_size[1]
        img_width = img_size[0]
        
        startX = np.ceil(box_X-box_width/2.0)-1
        startY = np.ceil(box_Y-box_height/2.0)-1
        endX = np.floor(box_X+box_width/2.0)-1
        endY = np.floor(box_Y+box_height/2.0)-1
        
        if startX<0:
            startX = 0
        if startY<0:
            startY = 0
        if endX>(img_width-1):
            endX = img_width-1
        if endY>(img_height-1):
            endY = img_height-1
        img = cv2.rectangle(img, (int(startX),int(startY)), (int(endX),int(endY)), (0,255,0), 2)
        
        result_box = str(int(startX)) +','+ str(int(startY)) +','+ str(int(endX)) +','+ str(int(endY))+'\n'
        
        gt_start_X = np.float32(gt_box[0])
        gt_start_Y = np.float32(gt_box[1])
        gt_width = np.float32(gt_box[2])
        gt_height = np.float(gt_box[3])
        
        gt_end_X = gt_start_X + gt_width
        gt_end_Y = gt_start_Y + gt_height
        
        img = cv2.rectangle(img, (int(gt_start_X),int(gt_start_Y)), (int(gt_end_X),int(gt_end_Y)), (0,0,255), 2)
        
    return img, result_box

def load_data(img_path, rec_path):
    frames = os.listdir(img_path)
    frames.sort(key=lambda x:int(x[:-4])) 
    imgs = []
    for frame_ind, frame_name in enumerate(frames):
        frame_path = img_path+frame_name
        frame = cv2.imread(frame_path)
        imgs.append(frame)
    ground_truth = open(rec_path,'r') 
    lines = ground_truth.readlines()
    ground_boxes = []
    print len(lines)
    for line in lines:
        if line.find('\t')>=0:
           new_line = line.strip('\r\n').split('\t')
        if line.find(',')>=0:
           new_line = line.strip('\r\n').split(',')
        """
        box = ['', '', '', '']
        box_ind = 0
        for l in range(len(line)):
            if line[l] !='\t' and line[l] !=',':
                box[box_ind] = box[box_ind]+str(line[l])
            else:
                box_ind = box_ind+1
        """
        ground_boxes.append(new_line)
    ground_truth.close()
    
    return imgs, ground_boxes
    

if __name__ == "__main__":
    cls_name = 'Skater2'
    tracking_type = "20_times"
    sequences_path = cwd+'/'+'sequences'+'/'+cls_name+'/'
    results_path = cwd+'/'+'results'+'/'+ tracking_type + '/' + cls_name
    
    if os.path.exists(sequences_path+'img'):
        img_path = sequences_path+'img'+'/'
        rec_path = sequences_path+'groundtruth_rect.txt'
    else:
        img_path = sequences_path + cls_name + '/'+'img'+'/'
        rec_path = sequences_path + cls_name + '/'+'groundtruth_rect.txt'
        
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    #============load models===============
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    #sess = tf.Session()
    
    ############====models===#####################
    faster_full = Faster_full(init_score_thresh=0.7, 
                              init_nms_thresh=0.1, 
                              Meta_path = None, #cwd+'/'+'faster_full/model/VGGnet_fast_rcnn_iter_80000.ckpt.meta', 
                              Model_path= cwd+'/'+'faster_full/model/'+cls_name+'/full_model'+'/'+'VGGnet_fast_rcnn_iter_20000.ckpt')
    
    imgs, gt_boxes = load_data(img_path,  rec_path)
    result_path = results_path + '/result.txt'
    result = file(result_path, 'a+')
    print "As for file in :", img_path
    CLE_ave = 0.0
    ROI_ave = 0.0
    positions_tracker = []
    positions_trackers = []
    speed_sum =0.0
    for frame_ind, frame_name in enumerate(imgs):
        #img_path = file_path+frame_name
        #get the img of rgb and depth data
        frame = frame_name
        height = frame.shape[0]
        width = frame.shape[1]
        size = np.array((width, height), dtype=np.float32)
        
        if frame_ind<=INITIAL_FRAME_NUM:
            positions_tracker= get_initial_localization(img=frame, img_size=size, Faster_RCNN_ins=faster_full)
            positions_trackers.append(positions_tracker)
            if (frame_ind==INITIAL_FRAME_NUM):
                positions_tracker = initial_tracker(first_trackers=positions_trackers, threshold=INITIAL_TRACKER_NUM)
                #print '=============================tracker==========================='
                #print positions_tracker,frame_ind
        else:
            start = time.clock()
            positions_tracker = particle_filter_VGG_3(img=frame, img_size=size, last_positions=positions_tracker, Faster_RCNN_ins=faster_full)
            speed = (time.clock()-start)
            speed_sum = speed_sum+speed
            positions_trackers.append(positions_tracker)
        
        #print frame_ind, positions_tracker
        if EVAL:
            this_CLE = 0.0
            this_ROI = 0.0
        if len(positions_trackers[frame_ind]) == 0:
            del positions_trackers[-1]
            positions_trackers.append(positions_trackers[frame_ind-1])
            frame, result_box = draw_boxes(img=frame, img_size=size, boxes=positions_trackers[frame_ind-1], gt_box=gt_boxes[frame_ind])
            result.write(result_box)
        else:
            #print positions_tracker
            frame, result_box = draw_boxes(img=frame, img_size=size, boxes=positions_tracker, gt_box=gt_boxes[frame_ind])
            result.write(result_box)
            
        if EVAL:
            CLE_ave += this_CLE
            ROI_ave += this_ROI
        cv2.imshow('vodeo', frame)
        k = cv2.waitKey(1)
        if(k & 0xff ==ord('q')):# or(frame_ind==):
            break
    result.close()
    print "time for all frames:", speed_sum, "s"