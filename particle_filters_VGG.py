
import numpy as np 
import cv2
import tensorflow as tf
from scipy.spatial.distance import pdist
from numpy.random import uniform, randn, randint

#from Faster_demo import Faster_RCNN
DEBUG = 0

def roi(box_1, box_2):
    X_min_1 = box_1[0]
    Y_min_1 = box_1[1]
    X_max_1 = box_1[2]
    Y_max_1 = box_1[3]
    width_1 = X_max_1-X_min_1
    height_1 = Y_max_1-Y_min_1
    
    X_min_2 = box_2[0]
    Y_min_2 = box_2[1]
    X_max_2 = box_2[2]
    Y_max_2 = box_2[3]
    width_2 = X_max_2-X_min_2
    height_2 = Y_max_2-Y_min_2
    
    overlapW = np.min([X_max_2, X_max_1])-np.max([X_min_2, X_min_1])
    overlapH = np.min([Y_max_2, Y_max_1])-np.max([Y_min_2, Y_min_1])
    if overlapW<0 or overlapH<0:
        return 0
    else:
        return overlapW*overlapH*1.0/(width_1*height_1+width_2*height_2-overlapW*overlapH)
    
##===========particles with Gaussian Distribution=================
def particle_filter_VGG_1(img, img_size, last_positions, Faster_RCNN_ins):  
    """
    img: original image
    last positions:list of several positions of several objects
    descriptors:the feature vector for each object
    """
    score_threshold = 0.7
    
    objects, feature_map, rpn_deltas, rpn_scores, im_scales = Faster_RCNN_ins.Faster_run(image = img, is_init=False)
    dx = rpn_deltas[:, :, :, 0::4]
    dy = rpn_deltas[:, :, :, 1::4]
    dw = rpn_deltas[:, :, :, 2::4]
    dh = rpn_deltas[:, :, :, 3::4]
    
    #print rpn_deltas.shape, dx.shape
    #print rpn_scores.shape
    probs = rpn_scores[:,:,:,1::2]
    print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    print rpn_scores[:,1,1,:]
    feature_H = rpn_scores.shape[1]
    feature_W = rpn_scores.shape[2]    
    img_height = img_size[1]
    img_width = img_size[0]
    #ratioX = 16.0/im_scales
    #ratioY = 16.0/im_scales
    ratioX = img_width*1.0/feature_W
    ratioY = img_height*1.0/feature_H
    #print "===================================================================================================="
    #print ratioY, im_scales
    anchors = Faster_RCNN_ins.anchors
    """
    for i in range(2):
        for j in range(2):
            print np.sum(scores[0,i,j,:,15])
    """
    #print scores
    this_positions = []
    #print descriptors
    #print "####################################################"
    #============particle filter for each object===============
    for object_ind, last_position in enumerate(last_positions):
        object_name = last_position['name']
        last_centerX = last_position['centerX']
        last_centerY = last_position['centerY']
        last_height = last_position['height']
        last_width = last_position['width']
        last_box = np.array([(0-last_width/2),
                             (0-last_height/2),
                             (last_width/2),
                             (last_height/2),
                            ])
        
        #print "#=================anchor_ind=======================#"
        #print anchor_ind
        #================descriptor of this object=================
        #feature = descriptors[object_ind]['descriptor']
        #===========predicted centers by 2D Gaussian, with shape being [last_height, last_width, 2]==========
        mu = [last_centerX, last_centerY]
        sigma = [[10000,10000], [10000,10000]]
        particle_shape = [int(5), int(5)]
        #particle_num = particle_shape[0]*particle_shape[0]
        pre_center = np.random.multivariate_normal(mu, sigma, particle_shape)
        #===========get the descriptor for each particle===============
        particles = []
        distance_sum = 0.0#np.zeros((1,128), dtype = tf.float32)
        for h in range(particle_shape[0]):
            for w in range(particle_shape[1]):
                #===================get the center in subspace======================
                pre_centerX = pre_center[h, w, 0]
                pre_centerY = pre_center[h, w, 1]
                startX = np.ceil(pre_centerX-last_width/2.0)-1
                startY = np.ceil(pre_centerY-last_height/2.0)-1
                endX = np.floor(pre_centerX+last_width/2.0)-1
                endY = np.floor(pre_centerY+last_height/2.0)-1
                
                feature_X = int(np.round(pre_centerX/ratioX))
                feature_Y = int(np.round(pre_centerY/ratioY))  
                if feature_X<0:
                    feature_X = 0
                if feature_Y<0:
                    feature_Y = 0
                if feature_X>feature_W-1:
                    feature_X = feature_W-1
                if feature_Y>feature_H-1:
                    feature_Y = feature_H-1
                if (startX >= endX) or (startY >= endY):
                    continue
                #===================get the index of which anchor in subspace======================
                this_dw = np.array([dw[0, feature_Y, feature_X, :],
                                    dh[0, feature_Y, feature_X, :],
                                    dw[0, feature_Y, feature_X, :],
                                    dh[0, feature_Y, feature_X, :]])
                
                pre_box = (np.exp(this_dw.T))*anchors
                rois = np.zeros((9), dtype = np.float32)
                for i in range(9):
                    rois[i] = roi(last_box, pre_box[i,:])
                    #print "#=================anchors=======================#"
                    #print last_box, rois, anchors[i,:]
                anchor_ind = np.argmax(rois)
                pre_width = pre_box[anchor_ind, 2]-pre_box[anchor_ind, 0]
                pre_height = pre_box[anchor_ind, 3]-pre_box[anchor_ind, 1]
                #===============================get the weights=============================
                print "mini center"
                print feature_X, feature_Y
                particle_weights = probs[0, feature_Y, feature_X, :]
                
                num_votes = len(np.where(particle_weights>score_threshold))
                max_ind = np.argmax(particle_weights)
                particle_weight = particle_weights[max_ind]
                if particle_weight>score_threshold:
                    particle_weight=particle_weight#particle_weights[:, anchor_ind]
                else:
                    particle_weight=0.01
                    
                particle = {'center_x':pre_centerX, 'center_y':pre_centerY, 'height':pre_height, 'width':pre_width, 'weight':particle_weight}#, 'descriptor':particle_descriptor}
                particles.append(particle)
                distance_sum = distance_sum + particle_weight
        new_X = 0
        new_Y = 0
        new_height = 0
        new_width = 0
        for prediction in particles:
            predictionX = prediction['center_x']
            predictionY = prediction['center_y']
            predictionHeight = prediction['height']
            predictionWidth = prediction['width']
            predictionW = prediction['weight']
            
            new_X = new_X + predictionX*predictionW/distance_sum
            new_Y = new_Y + predictionY*predictionW/distance_sum
            new_height = new_height + predictionHeight*predictionW/distance_sum
            
        #print new_X, predictionX, predictionW, distance_sum
        object_position = {'name':object_name, 'centerX':np.round(new_X), 'centerY':np.round(new_Y), 'height':new_height, 'width':new_width}
        this_positions.append(object_position)
    return this_positions
##===========particles with Resampling=================
def simple_resampling(particles,  threshold=0.8):
    N = len(particles)
    weights = particles[:, 2]
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] =1.0
    indexes = np.searchsorted(cumulative_sum, np.random.random(N))
    #print '===============================resampling indexes==================='
    #print indexes
    particles = particles[indexes, :]
    particles[:,2].fill(1.0/N)
    return particles
#==============Gaussian in this position=======================
def Gaussian_resampling(particles, this_X, this_Y, threshold=0.9):
    N = len(particles)
    #print particles, N
    for i in range(N):
        if particles[i, 2]<threshold:
            particles[i, 0]=this_X+randn(1)[0]*50.0
            particles[i, 1]=this_Y+randn(1)[0]*50.0
    particles[:,2].fill(1.0/N)
    return particles

def resampling_within_box(particles, imH, imW, this_X, this_Y, this_W, this_H, ori_weights, threshold=0.9):
    N = len(particles)
    if np.max(ori_weights)<=threshold:
        particles = np.empty((N, 3))
        particles[:, 0] = uniform(0, imW, N)
        particles[:, 1] = uniform(0, imH, N)
        particles[:, 2] = np.ones((N), dtype = np.float32)/N
        #print len(np.where(ori_weights>0.8))
        return particles
    else:
        #print particles, N
        resampleH = this_H*1.5/2.0
        if resampleH<10:
            resampleH = 10
        resampleW = this_W*1.5/2.0
        if resampleW<10:
            resampleW = 10
        for i in range(N):
            if particles[i, 2]<threshold:
                particles[i, 0] = np.float32(randint(this_X-resampleW, this_X+resampleW, (1,1))[0])
                particles[i, 1] = np.float32(randint(this_Y-resampleH, this_Y+resampleH, (1,1))[0])
                particles[i, 2] = 1.0/N
        return particles
    
def particle_filter_VGG_3(img, img_size, last_positions, Faster_RCNN_ins):  
    """
    img: original image
    last positions:list of several positions of several objects
    descriptors:the feature vector for each object
    """
    High_score_threshold = 0.7
    Low_score_threshold = 0.3
    score_threshold = 0.5
    
    objects, feature_map, rpn_deltas, rpn_scores, im_scales = Faster_RCNN_ins.Faster_run(image = img, is_init=False)
    dx = rpn_deltas[:, :, :, 0::4]
    dy = rpn_deltas[:, :, :, 1::4]
    dw = rpn_deltas[:, :, :, 2::4]
    dh = rpn_deltas[:, :, :, 3::4]
    
    #print rpn_deltas.shape, dx.shape
    #print rpn_scores.shape
    probs = rpn_scores[:,:,:,1::2]
    #print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    
    is_filtering = len(probs[np.where(probs>High_score_threshold)])
    if DEBUG:
        print probs[np.where(probs>High_score_threshold)], is_filtering
    feature_H = rpn_scores.shape[1]
    feature_W = rpn_scores.shape[2]    
   
    img_height = img_size[1]
    img_width = img_size[0]
    #ratioX = 16.0/im_scales
    #ratioY = 16.0/im_scales
    ratioX = img_width*1.0/feature_W
    ratioY = img_height*1.0/feature_H
    ratioW = im_scales
    ratioH = im_scales
    #print "===================================================================================================="
    #print ratioY, im_scales
    anchors = Faster_RCNN_ins.anchors
    """
    for i in range(2):
        for j in range(2):
            print np.sum(scores[0,i,j,:,15])
    """
    #print scores
    this_positions = []
    #print descriptors
    #print "####################################################"
    #============particle filter for each object===============
    for object_ind, last_position in enumerate(last_positions):
        object_name = last_position['name']
        last_centerX = last_position['centerX']
        last_centerY = last_position['centerY']
        last_height = last_position['height']
        last_width = last_position['width']
        last_particles = last_position['particles']
        last_box = np.array([(0-last_width/2),
                             (0-last_height/2),
                             (last_width/2),
                             (last_height/2),
                            ])
        #===========get the descriptor for each particle===============
        particles = []
        distance_sum = 0.0#np.zeros((1,128), dtype = tf.float32)
        particle_num = len(last_particles)
        num_sparse_conf = 0
        for particle_ind in range(particle_num):
            #===================get the center in subspace======================
            pre_centerX = last_particles[particle_ind, 0]
            pre_centerY = last_particles[particle_ind, 1]
            startX = np.ceil(pre_centerX-last_width/2.0)-1
            startY = np.ceil(pre_centerY-last_height/2.0)-1
            endX = np.floor(pre_centerX+last_width/2.0)-1
            endY = np.floor(pre_centerY+last_height/2.0)-1
            
            feature_X = int(np.round(pre_centerX/ratioX))
            feature_Y = int(np.round(pre_centerY/ratioY))  
            if feature_X<0:
                feature_X = 0
            if feature_Y<0:
                feature_Y = 0
            if feature_X>feature_W-1:
                feature_X = feature_W-1
            if feature_Y>feature_H-1:
                feature_Y = feature_H-1
            if (startX >= endX) or (startY >= endY):
                continue
            #===================get the index of which anchor in subspace======================
            this_dx = np.array(dx[0, feature_Y, feature_X, :])
            this_dy = np.array(dy[0, feature_Y, feature_X, :])
            this_dw = np.array([dw[0, feature_Y, feature_X, :],
                                dh[0, feature_Y, feature_X, :],
                                dw[0, feature_Y, feature_X, :],
                                dh[0, feature_Y, feature_X, :]])
            
            pre_box = np.multiply(np.exp(this_dw.T), anchors)#np.array([anchors[:, 2]-anchors[:, 0], anchors[:, 3]-anchors[:, 1]])
            rois = np.zeros((9), dtype = np.float32)
            particle_weights = probs[0, feature_Y, feature_X, :]
            
            num_votes = len(np.where(particle_weights>score_threshold))
            max_ind = np.argmax(particle_weights)
            pre_box[:, 0::2] = pre_box[:, 0::2]/ratioW
            pre_box[:, 1::2] = pre_box[:, 1::2]/ratioH
            for i in range(9):
                if particle_weights[i]>score_threshold:
                    rois[i] = roi(last_box, pre_box[i,:])
                else:
                    rois[i] = 0.1
                #print "#=================anchors=======================#"
                #print last_box, rois, anchors[i,:]
            #===============================get the weights=============================
            #print "mini center"
            if DEBUG:
               print feature_X, feature_Y, max_ind, particle_weights[max_ind]
            
            anchor_ind = np.argmax(rois)
            
            pre_width = (pre_box[anchor_ind, 2]-pre_box[anchor_ind, 0])#/ratioW
            pre_height = (pre_box[anchor_ind, 3]-pre_box[anchor_ind, 1])#/ratioH
            
            pre_width = (anchors[max_ind, 2] - anchors[max_ind, 0]) * np.exp(dw[0, feature_Y, feature_X, max_ind])/ratioW
            pre_height = (anchors[max_ind, 3] - anchors[max_ind, 1]) * np.exp(dh[0, feature_Y, feature_X, max_ind])/ratioH
            particle_weight = particle_weights[max_ind]#*last_particles[particle_ind, 2]
            if particle_weight<Low_score_threshold:#(particle_weight<High_score_threshold)&(particle_weight>Low_score_threshold):
                particle_weight = 0.01
            if particle_weight>High_score_threshold:
                num_sparse_conf += 1
            pre_centerX = this_dx[max_ind] * 16.0 + pre_centerX
            pre_centerY = this_dy[max_ind] * 16.0 + pre_centerY
            
            particle = {'center_x':pre_centerX, 'center_y':pre_centerY, 'height':pre_height, 'width':pre_width, 'weight':particle_weight}#, 'descriptor':particle_descriptor}
            particles.append(particle)
            distance_sum = distance_sum + particle_weight
        
        new_X = 0
        new_Y = 0
        new_height = 0
        new_width = 0
        original_weight = []
        for pre_ind, prediction in enumerate(particles):
            predictionX = prediction['center_x']
            predictionY = prediction['center_y']
            predictionHeight = prediction['height']
            predictionWidth = prediction['width']
            predictionW = prediction['weight']
            original_weight.append(predictionW/distance_sum)
            last_particles[pre_ind, 2] = predictionW/distance_sum
            if DEBUG:
                print predictionW, distance_sum, last_particles[pre_ind, 2] 
            new_X = new_X + predictionX*predictionW/distance_sum
            new_Y = new_Y + predictionY*predictionW/distance_sum
            new_height = new_height + predictionHeight*predictionW/distance_sum
            new_width = new_width + predictionWidth*predictionW/distance_sum
        is_constrain = (new_width>img_width*0.02)&(new_height>img_height*0.02)#&(particle_weight>3)
        if is_constrain:
            if new_width<last_width*0.8:
                new_width = last_width
            if new_height<last_height*0.8:
                new_height = last_height
            if new_width>last_width*1.2:
                new_width=last_width
            if new_height>last_height*1.2:
                new_height=last_height
            if np.abs(new_X-last_centerX)>new_width/2.0:
                new_X=last_centerX
            if np.abs(new_Y-last_centerY)>new_height/2.0:
                new_Y=last_centerY
        if is_filtering==0:
            print "pass filtering"
            new_width = last_width
            new_height = last_height
            new_X=last_centerX
            new_Y=last_centerY
        object_position = {'name':object_name,
                           'centerX':np.round(new_X), 
                           'centerY':np.round(new_Y), 
                           'height':new_height, 
                           'width':new_width, 
                           'particles':resampling_within_box(particles=last_particles,
                                                             imH=img_height, 
                                                             imW=img_width,
                                                             this_X=new_X, 
                                                             this_Y=new_Y, 
                                                             this_W=new_width, 
                                                             this_H = new_height, 
                                                             ori_weights = original_weight,
                                                             threshold=1.0/particle_num*1.1)}#last_particles}#
        if DEBUG:
            print object_position
        this_positions.append(object_position)
    return this_positions
##===========particles with searching=================
def particle_filter_VGG_4(img, img_size, last_positions, Faster_RCNN_ins):  
    """
    img: original image
    last positions:list of several positions of several objects
    descriptors:the feature vector for each object
    """
    score_threshold = 0.7
    
    objects, feature_map, rpn_deltas, rpn_scores, im_scales = Faster_RCNN_ins.Faster_run(image = img)
    dx = rpn_deltas[:, :, :, 0::4]
    dy = rpn_deltas[:, :, :, 1::4]
    dw = rpn_deltas[:, :, :, 2::4]
    dh = rpn_deltas[:, :, :, 3::4]
    
    anchors = Faster_RCNN_ins.anchors
    widths = anchors[:, 2] - anchors[:, 0] + 1.0
    heights = anchors[:, 3] - anchors[:, 1] + 1.0
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights
    print widths, heights, ctr_x, ctr_y
    #print rpn_deltas.shape, dx.shape
    #print rpn_scores.shape
    probs = rpn_scores[:,:,:,1::2]
    print '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    print rpn_scores[:,1,1,:]
    feature_H = rpn_scores.shape[1]
    feature_W = rpn_scores.shape[2]    
    img_height = img_size[1]
    img_width = img_size[0]
    #ratioX = 16.0/im_scales
    #ratioY = 16.0/im_scales
    ratioX = img_width*1.0/feature_W
    ratioY = img_height*1.0/feature_H
    #print "===================================================================================================="
    #print ratioY, im_scales

    #print scores
    this_positions = []
    #print descriptors
    #print "####################################################"
    #============particle filter for each object===============
    for object_ind, last_position in enumerate(last_positions):
        object_name = last_position['name']
        last_centerX = last_position['centerX']
        last_centerY = last_position['centerY']
        last_height = last_position['height']
        last_width = last_position['width']
        last_particles = last_position['particles']
        last_box = np.array([(0-last_width/2),
                             (0-last_height/2),
                             (last_width/2),
                             (last_height/2),
                            ])
        
        #===========get the descriptor for each particle===============
        particles = []
        distance_sum = 0.0#np.zeros((1,128), dtype = tf.float32)
        particle_num = len(last_particles)
        for particle_ind in range(particle_num):
            #===================get the center in subspace======================
            pre_centerX = last_particles[particle_ind, 0]
            pre_centerY = last_particles[particle_ind, 1]
            startX = np.ceil(pre_centerX-last_width/2.0)-1
            startY = np.ceil(pre_centerY-last_height/2.0)-1
            endX = np.floor(pre_centerX+last_width/2.0)-1
            endY = np.floor(pre_centerY+last_height/2.0)-1
            
            feature_X = int(np.round(pre_centerX/ratioX))
            feature_Y = int(np.round(pre_centerY/ratioY))  
            if feature_X<0:
                feature_X = 0
            if feature_Y<0:
                feature_Y = 0
            if feature_X>feature_W-1:
                feature_X = feature_W-1
            if feature_Y>feature_H-1:
                feature_Y = feature_H-1
            if (startX >= endX) or (startY >= endY):
                continue
            #===================get the index of which anchor in subspace======================
            this_delta = np.array([dw[0, feature_Y, feature_X, :],
                                dh[0, feature_Y, feature_X, :],
                                dw[0, feature_Y, feature_X, :],
                                dh[0, feature_Y, feature_X, :]])
            this_dx = dx[0, feature_Y, feature_X, :]
            this_dy = dy[0, feature_Y, feature_X, :]
            this_dw = dw[0, feature_Y, feature_X, :]
            this_dh = dh[0, feature_Y, feature_X, :]
            print this_delta
            pred_ctr_x = this_dx * widths + pre_centerX
            pred_ctr_y = this_dy * heights + pre_centerY
            pred_w = np.exp(this_dw) * widths
            pred_h = np.exp(this_dh) * heights
            
            pre_box = (np.exp(this_delta.T))*anchors
            rois = np.zeros((9), dtype = np.float32)
            for i in range(9):
                rois[i] = roi(last_box, pre_box[i,:])
                #print "#=================anchors=======================#"
                #print last_box, rois, anchors[i,:]
            anchor_ind = np.argmax(rois)
            pre_width = pre_box[anchor_ind, 2]-pre_box[anchor_ind, 0]
            pre_height = pre_box[anchor_ind, 3]-pre_box[anchor_ind, 1]
            #===============================get the weights=============================
            print "mini center"
            print feature_X, feature_Y
            particle_weights = probs[0, feature_Y, feature_X, :]
            
            num_votes = len(np.where(particle_weights>score_threshold))
            max_ind = np.argmax(particle_weights)
            particle_weight = particle_weights[anchor_ind]
            particle = {'center_x':pred_ctr_x[anchor_ind], 'center_y':pred_ctr_y[anchor_ind], 'height':pre_height, 'width':pre_width, 'weight':particle_weight}#, 'descriptor':particle_descriptor}
            particles.append(particle)
            distance_sum = distance_sum + particle_weight
        new_X = 0
        new_Y = 0
        new_height = 0
        new_width = 0
        for prediction in particles:
            predictionX = prediction['center_x']
            predictionY = prediction['center_y']
            predictionHeight = prediction['height']
            predictionWidth = prediction['width']
            predictionW = prediction['weight']
            
            new_X = new_X + predictionX*predictionW/distance_sum
            new_Y = new_Y + predictionY*predictionW/distance_sum
            new_height = new_height + predictionHeight*predictionW/distance_sum
            new_width = new_width + predictionWidth*predictionW/distance_sum
        #pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        #pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        print "=============================bug===================="
        print ratioX, new_X
        
        #print new_X, predictionX, predictionW, distance_sum
        object_position = {'name':object_name,
                           'centerX':np.round(new_X), 
                           'centerY':np.round(new_Y), 
                           'height':new_height, 
                           'width':new_width, 
                           'particles':simple_resampling(last_particles)}
        this_positions.append(object_position)
    return this_positions  
    
    
    
    

##===========particles with Gaussian Distribution=================
def particle_filter_VGG_2(img, img_size, last_positions, descriptors, SSD_ins, sess):  
    """
    img: original image
    last positions:list of several positions of several objects
    descriptors:the feature vector for each object
    """
    img_height = img_size[1]
    img_width = img_size[0]
    ratioX = img_width*1.0/SSD_ins.net_shape[0]*8.0
    ratioY = img_height*1.0/SSD_ins.net_shape[1]*8.0
    print "ratio in X and Y"
    print ratioX, ratioY
    SSD_img = cv2.resize(img, SSD_ins.net_shape, interpolation=cv2.INTER_LINEAR)
    classes, scores, bboxes, VGG_feature_map, out_predictions = SSD_ins.run_SSD(SSD_img)# only get the feature_map in practice
    #print out_predictions[0].shape#(1,64,64,4,21)
    scores = out_predictions[0]
    this_positions = []
    #print descriptors
    #print "####################################################"
    #============particle filter for each object===============
    
    for object_ind, last_position in enumerate(last_positions):
        print "=========================== new start ====================================================="
        object_name = last_position['name']
        last_centerX = last_position['centerX']
        last_centerY = last_position['centerY']
        last_height = last_position['height']
        last_width = last_position['width']
        img_height = img_size[1]
        img_width = img_size[0]
        #================descriptor of this object=================
        feature = descriptors[object_ind]['descriptor']
        #============particle for center==========================
        center_particle_shape = [int(2), int(2)]
        center_particles = []
        center_weight_sum = 0.0#np.zeros((1,128), dtype = tf.float32)
        new_X = 0
        new_Y = 0
        for h in range(center_particle_shape[0]):
            for w in range(center_particle_shape[1]):
                last_startX = last_centerX+last_width/2.0*(w-1)
                if last_startX<0:
                    last_startX=0
                last_startY = last_centerY+last_height/2.0*(h-1)
                if last_startY<0:
                    last_startY=0
                last_endX = last_startX+last_width/2.0
                if last_endX>img_width:
                    last_endX=img_width
                last_endY = last_startY+last_height/2.0
                if last_endY>img_height:
                    last_endY=img_height
                pre_centerX = np.random.randint(last_startX, last_endX, (1,1))[0]
                pre_centerY = np.random.randint(last_startY, last_endY, (1,1))[0]
                print "####==============pre_center============"
                print last_startX, last_endX, last_startY, last_endY, pre_centerX, pre_centerY
                startX = np.ceil((pre_centerX-last_width/2.0)/ratioX)
                startY = np.ceil((pre_centerY-last_height/2.0)/ratioY)
                endX = np.floor((pre_centerX+last_width/2.0)/ratioX)
                endY = np.floor((pre_centerY+last_height/2.0)/ratioY)
                if startX<0:
                    startX = 0
                if startY<0:
                    startY = 0
                if endX>64:
                    endX = 64
                if endY>64:
                    endY = 64
                if (startX >= endX) or (startY >= endY):
                    continue
                feature_X = int(np.round(pre_centerX/ratioX))
                feature_Y = int(np.round(pre_centerY/ratioY))
                particle_weight = np.sum(scores[0, feature_Y, feature_X, :, 14])/4.0
                print "all scores print"
                print scores[0, feature_Y, feature_X, :, :]
                #print particle_weight
                #X = np.vstack([particle_descriptor.reshape((1,-1)), feature.reshape((1,-1))])
                print "##############################################"
                #particle_distance = pdist(X, 'seuclidean')
                print particle_weight 
                print "##############################################"
                center_particle = {'center_x':pre_centerX, 'center_y':pre_centerY, 'weight':particle_weight}#, 'descriptor':particle_descriptor}
                center_particles.append(center_particle)
                center_weight_sum = center_weight_sum + particle_weight
        print "weight_sum"
        print center_weight_sum
        for prediction in center_particles:
            predictionX = prediction['center_x']
            predictionY = prediction['center_y']
            predictionW = prediction['weight']
            new_X = new_X + predictionX*predictionW/center_weight_sum
            new_Y = new_Y + predictionY*predictionW/center_weight_sum
        if new_X>img_width:
                    new_X=img_width
        if new_Y>img_height:
                    new_Y=img_height
        
        object_position = {'name':object_name, 'centerX':np.round(new_X), 'centerY':np.round(new_Y), 'height':last_height, 'width':last_width}
        this_positions.append(object_position)
    return this_positions