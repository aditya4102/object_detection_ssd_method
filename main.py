import cv2


vid = cv2.VideoCapture(1)
vid.set(3,640)
vid.set(4,480)


classNames = []
classFile = 'Resources/coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    flag, img = vid.read()
    classIds,  confs, bbox = net.detect(img,confThreshold=0.5)
    print(classIds, bbox, confs)

    if len(classIds) !=0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classId - 1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),thickness=1)
            cv2.putText(img, str(round(confidence*100,2)), (box[0] + 17p0, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=1)


    cv2.imshow("Output",img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break



