import cv2
import torch
import torchvision
import imutils
import argparse
import matplotlib.pyplot as plt
import PIL
from torchvision import transforms

def capture_2(model):
    parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                                      OpenCV. You can process both videos and images.')
    parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
    parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
    args = parser.parse_args()
    # create Background Subtractor objects
    if args.algo == 'MOG2':
        print("I am in MOG2")
        backSub = cv2.createBackgroundSubtractorMOG2()
    else:
        print("I am in KNN")
        backSub = cv2.createBackgroundSubtractorKNN()

    print(backSub)
    cap = cv2.VideoCapture(0)
    #cap.set(3, 1280)
    #cap.set(4, 720)
    fps = 0
    show_score = 0
    show_res = 'Nothing'
    sequence = 0
    data_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]
    )

    while True:
        ret, frame = cap.read()  # Capture each frame
        if fps == 30:
            image = frame[100:450, 150:570]
            image = backSub.apply(image)
            cv2.imshow("converted", image)
            image = PIL.Image.fromarray(image)  # Webcam frames are numpy array format
            # Therefore transform back to PIL image
            #print(image)
            image_data = data_transforms(image)
            #print(image_data)
            result, score = model.test_single_unknown(image_data)
            fps = 0
            print("Class Label....{}".format(result))
            print("Confidence....{}".format(score))
            if score >= 0.5:
                show_res = result
                show_score = score
            else:
                show_res = "Nothing"
                show_score = score

        fps += 1
        cv2.putText(frame, '%s' % (show_res), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(frame, '(score = %.5f)' % (show_score), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
        cv2.rectangle(frame, (64, 64), (128, 128), (250, 0, 0), 2)
        cv2.imshow("ASL SIGN DETECTER", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow("ASL SIGN DETECTER")