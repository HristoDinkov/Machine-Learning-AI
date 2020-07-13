import argparse
import logging
import time
from tf_pose import common
import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


import keras

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    
    
    try_list = []
    for i in range(39):
        i=[]
        try_list.append(i)
        
# =============================================================================
#     model = keras.Sequential([
#         keras.layers.Dense(38, activation=tf.nn.relu),
#         keras.layers.Dense(40, activation=tf.nn.relu),
#         keras.layers.Dense(5, activation=tf.nn.softmax)
#     ])
#     
#     model.compile(optimizer='adam', 
#           loss='sparse_categorical_crossentropy',
#           metrics=['accuracy'])
# =============================================================================
    
    model = keras.models.load_model('model.h5')

    while True:
        comp_list=[]
        ret_val, image = cam.read()

        logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        logger.debug('postprocess+')
        
        image_image = common.read_imgfile("maxresdefault.jpg", None, None)
        for i in range(20):
            try:
                try_list[i*2].append(humans[0].body_parts[i].x)
                try_list[i*2+1].append(humans[0].body_parts[i].y)
            except:
                if i==19:
                    try_list[i*2].append(6)
                else:
                    try_list[i*2].append(0)
                    try_list[i*2+1].append(0)
        
        for i in range(19):
            try:
                comp_list.append(humans[0].body_parts[i].x)
                comp_list.append(humans[0].body_parts[i].y)
            except:
                comp_list.append(0)
                comp_list.append(0)
        
        comp_list = np.asmatrix(comp_list)
        print (comp_list)
        
        predictions = model.predict(comp_list)
        
        prediction = np.argmax(predictions[0])
        
        accuracy = predictions[0][prediction]*100
        
        if accuracy < 90:
            prediction = 0
        
        print(prediction)
        print(accuracy)
        
        
        
        image = TfPoseEstimator.draw_humans(image_image, humans, imgcopy=False)
    
        
        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')
    try_list = np.asarray(try_list)
    np.savetxt("data_temp.csv", try_list.transpose(), delimiter=",")
    cv2.destroyAllWindows()
