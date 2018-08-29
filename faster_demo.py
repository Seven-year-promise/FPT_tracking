import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
import os.path as osp
import glob

this_dir = osp.dirname(__file__)
print(this_dir)

from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.test import im_detect
from lib.fast_rcnn.nms_wrapper import nms
from lib.utils.timer import Timer

CLASSES = ('__background__',
           '1')


# CLASSES = ('__background__','person','bike','motorbike','car','bus')

class Faster_RCNN():
    def __init__(self, init_score_thresh, init_nms_thresh, Meta_path=None, Model_path=None):
        cfg.TEST.HAS_RPN = True  # Use RPN for proposals
        self.score_thresh = init_score_thresh
        self.nms_thresh = init_nms_thresh
        #args = parse_args()
    
        if Model_path == ' ' or not os.path.exists(Model_path):
            print ('current path is ' + os.path.abspath(__file__))
            raise IOError(('Error: Model not found.\n'))
        self.anchors = np.array([[ -83.,  -39.,  100.,   56.],
                                 [-175.,  -87.,  192.,  104.],
                                 [-359., -183.,  376.,  200.],
                                 [ -55.,  -55.,   72.,   72.],
                                 [-119., -119.,  136.,  136.],
                                 [-247., -247.,  264.,  264.],
                                 [ -35.,  -79.,   52.,   96.],
                                 [ -79., -167.,   96.,  184.],
                                 [-167., -343.,  184.,  360.]])
        # init session
        # load network
        #print "loading kekekekekkeeeeeeeeeeeeeeeeeeeeeeeeeeee"
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.net = get_network('VGGnet_test')
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()#write_version = tf.train.SaverDef.V2)
            
            self.saver.restore(self.sess, Model_path)
        '''
            self.saver=tf.train.import_meta_graph(Meta_path)
        
        
        with self.sess.as_default():
            with self.graph.as_default():
                print ('Loading network {:s}... '.format('VGGnet_test'))
                #tf.get_variable_scope().reuse_variables()
                self.saver.restore(self.sess, Model_path)
        '''
        
        # load model
        
        
    def vis_detections(im, class_name, dets, ax, thresh=0.5):
        """Draw detected bounding boxes."""
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            return
    
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
    
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')
    
        ax.set_title(('{} detections with '
                      'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                      thresh),
                     fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.draw()
    
    
    def demo(self, image_name, is_init=True):
        """Detect object classes in an image using pre-computed object proposals."""
    
        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        if is_init:
            raw_scores, raw_boxes, self.feature_map, self.rpn_boxes, self.rpn_scores, self.im_scales = im_detect(self.sess, self.net, image_name,  is_part=False)
            CONF_THRESH = self.score_thresh
            NMS_THRESH = self.nms_thresh
            self.objects = []
            for cls_ind, cls in enumerate(CLASSES[1:]):
                cls_ind += 1  # because we skipped background
                cls_boxes = raw_boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                cls_scores = raw_scores[:, cls_ind]
                dets = np.hstack((cls_boxes,
                                  cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(dets, NMS_THRESH)
                dets = dets[keep, :]
                
                inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
                if len(inds) > 0:
                    for i in inds:
                        bbox = dets[i, :4]
                        score = dets[i, -1]
                        box_height = bbox[3] - bbox[1]
                        box_width = bbox[2] - bbox[0]
                        c_x = np.round(bbox[0]+box_width/2.0)
                        c_y = np.round(bbox[1]+box_height/2.0)
                        if cls=='stawberry':
                            cls='strawberry'
                        object_coordinates = {'name':cls, 'score': score, 'boxes':list([c_x, c_y, box_width, box_height])}
                        self.objects.append(object_coordinates)
        else:
            _, _, self.feature_map, self.rpn_boxes, self.rpn_scores, self.im_scales = im_detect(self.sess, self.net, image_name,  is_part=True)
        timer.toc()
    
    def parse_args():
        """Parse input arguments."""
        parser = argparse.ArgumentParser(description='Faster R-CNN demo')
        parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                            default=0, type=int)
        parser.add_argument('--cpu', dest='cpu_mode',
                            help='Use CPU mode (overrides --gpu)',
                            action='store_true')
        parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                            default='VGGnet_test')
        parser.add_argument('--model', dest='model', help='Model path',
                            default=' ')
    
        args = parser.parse_args()
    
        return args
    
    
    def Faster_run(self, image, is_init=True):
        
        #print (' done.')
    
        # Warmup on a dummy image
        im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
        for i in xrange(2):
            im_detect(self.sess, self.net, im)
        self.demo(image_name = image, is_init=is_init)
        return self.objects, self.feature_map, self.rpn_boxes, self.rpn_scores, self.im_scales

