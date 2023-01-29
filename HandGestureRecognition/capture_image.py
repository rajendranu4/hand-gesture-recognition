import cv2
import torch
import torchvision
import imutils
import argparse
import matplotlib.pyplot as plt

def capture(model):
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                  OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
    parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
    args = parser.parse_args()
    # create Background Subtractor objects
    if args.algo == 'MOG2':
        print("I am in MOG2")
        backSub = cv2.createBackgroundSubtractorMOG2(varThreshold = 30)
    else:
        print("I am in KNN")
        backSub = cv2.createBackgroundSubtractorKNN()

    # Initialize the webcam for Hand Gesture Recognition Python project
    cap = cv2.VideoCapture(0)
    counter = 0
    class_label = ''
    score = 0
    queue_frames = [0, 0]
    while True:
        # Read each frame from the webcam
        _, frame = cap.read()

        frame_gray = backSub.apply(frame)
        cv2.imshow('Bgremoved', frame_gray)
        x, y, c = frame.shape
        # Flip the frame vertically
        frame = cv2.flip(frame, 1)

        #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('gray', frame_gray)

        '''frame_gray = cv2.GaussianBlur(frame_gray, (5, 5), 0)
        thresh = cv2.threshold(frame_gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        cv2.imshow('Thresh', thresh)'''

        '''(thresh, im_bw) = cv2.threshold(frame_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresh = 127
        frame_gray = cv2.threshold(frame_gray, thresh, 255, cv2.THRESH_BINARY)[1]'''

        # send this image to classifier and get the class name

        resize_transforms = torchvision.transforms.Compose([
            #torchvision.transforms.CenterCrop(10),
            torchvision.transforms.Resize((128, 128))
        ])

        tensor_img = torch.from_numpy(frame_gray)

        tensor_img = resize_transforms(tensor_img[None,... ])
        tensor_img = tensor_img / 255

        '''if counter % 30 == 0:
            print("I am frame....{}".format(counter))
            plt.imshow(tensor_img.permute(1, 2, 0))'''

        counter = counter + 1

        if counter == 30:
            counter = 0
            class_label, score = model.test_single_unknown(tensor_img)
            queue_frames.pop(0)
            queue_frames.append(score)
            print("Class label: {} & Score: {}".format(class_label, score))
            #print("Class Label....{}".format(class_label))
            #print("Confidence....{}".format(score))

            '''counter = counter + 1
    
            if counter == 10000:
                counter = 0'''

        if all(frame_score > 45 for frame_score in queue_frames):
            class_label = class_label
        else:
            class_label = ''
        print(queue_frames)

        cv2.putText(frame, class_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Output", frame)

        if cv2.waitKey(1) == ord('q'):
            break
    # release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()