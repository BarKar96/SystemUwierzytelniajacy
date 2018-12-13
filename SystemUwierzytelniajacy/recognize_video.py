from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
from PIL import Image
from PIL import ImageTk
import tkinter as tk
import threading
import Timer
import firebase_module
import sys
import list_of_emails


global lala
lala = None

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
    help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
    help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
    help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
    help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")


class SampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self._frame = None
        self.switch_frame(StartPage)
        self.recognizedPerson = None
        self.recognizedConfidence = None
        self.maxConfidence = 0

    def switch_frame(self, frame_class):
        """Destroys current frame and replaces it with a new one."""
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()


class StartPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Label(self, text="Welcome in the authentication system!").pack(side="top", fill="x", pady=10)
        tk.Button(self, text="Authenticate yourself",
                  command=lambda: master.switch_frame(RecognizerPage)).pack(side="top", fill="x", pady=10)
        tk.Button(self, text="Exit",
                  command=lambda: self.exitFromApp()).pack(side="top", fill="x", pady=10)

    def exitFromApp(self):
        sys.exit()

class EndPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Label(self, text="You were not recognized.").pack(side="top", fill="x", pady=10)
        tk.Button(self, text="Back to start",
                  command=lambda: master.switch_frame(StartPage)).pack()
        tk.Button(self, text="Exit",
                  command=lambda: self.exitFromApp()).pack(side="top", fill="x", pady=10)

    def exitFromApp(self):
        sys.exit()




class AuthPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Label(self, text="System recognized you as: " + str(master.recognizedPerson) + " with confidence of " + str(master.recognizedConfidence*100)[:2] + "%").pack(side="top", fill="x", pady=10)
        email = tk.StringVar()
        password = tk.StringVar()
        tk.Label(self, text="Enter your email address:").pack(side="top", fill="x", pady=10)
        self.entry_box1 = tk.Entry(self, textvariable=email, width=25, bg="lightblue")
        self.entry_box1.pack()
        tk.Label(self, text="Enter your password:").pack(side="top", fill="x", pady=10)
        self.entry_box2 = tk.Entry(self, textvariable=password, width=25, bg="lightblue")
        self.entry_box2.pack()
        self.label = tk.Label(self, text="")
        self.label.pack(side="top", fill="x", pady=10)
        tk.Button(self, text="       OK       ",
                  command=lambda: self.connectToSystem(email.get(), password.get())).pack(pady=10)
        tk.Button(self, text="      Logout    ",
                  command=lambda: master.switch_frame(StartPage)).pack(pady=10)


    def getEmailFromFile(self):
        emailFromFile = None
        for i in range(len(list_of_emails.listOfTuples)):
                if str(self.master.recognizedPerson) == str(list_of_emails.listOfTuples[i][0]):
                    emailFromFile = list_of_emails.listOfTuples[i][1]
                    return emailFromFile

    def connectToSystem(self, email, password):
        email_from_file = self.getEmailFromFile()
        if email_from_file == email:
            print("byl taki email i osoba w pliku")
            isAuthOk = firebase_module.signInClient(email, password)
            if isAuthOk == 1:
                self.master.switch_frame(WelcomeInTheSystemPage)
            else:
                self.label.config(text='Incorrect credentials! Try again!')
                self.entry_box1.delete(0, 'end')
                self.entry_box2.delete(0, 'end')
        else:
            self.label.config(text='Email does not match to recognized person! Try again!')
            self.entry_box1.delete(0, 'end')
            self.entry_box2.delete(0, 'end')



class WelcomeInTheSystemPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Label(self, text="Welcome in the system " + str(master.recognizedPerson) + "!").pack(side="top", fill="x", pady=10)
        tk.Button(self, text="Logout",
                  command=lambda: master.switch_frame(StartPage)).pack(side="top", fill="x", pady=10)
        tk.Button(self, text="Exit",
                  command=lambda: self.exitFromApp()).pack(side="top", fill="x", pady=10)

    def exitFromApp(self):
        sys.exit()

class RecognizerPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Label(self, text="System is scanning your face for 5 seconds. Please wait...").pack(side="top", fill="x", pady=10)
        self.vs = cv2.VideoCapture(0)
        time.sleep(2.0)
        self.thread = None
        self.stopEvent = None
        self.panel = None
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        self.master.wm_title("PyImageSearch PhotoBooth")
        self.master.wm_protocol("WM_DELETE_WINDOW", self.onClose)

        tk.Button(self, text="Back to Start Page",
                             command= lambda: self.switchBetweenScenes(StartPage)).pack()



    def switchBetweenScenes(self, page_name):
        self.stopEvent.set()
        self.vs.release()
        if self.panel != None:
            self.panel.pack_forget()
        self.master.switch_frame(page_name)
        print("scene switched")


    def videoLoop(self):
        self.master.maxConfidence=0
        self.master.recognizedPerson = None
        self.master.recognizedConfidence = None
        Timer.startTimer(5)
        # I'm not a GUI developer, nor do I even pretend to be. This
        # try/except statement is a pretty ugly hack to get around
        # a RunTime error that Tkinter throws due to threading
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set() and Timer.q.empty():
                #print("robie")
                # grab the frame from the video stream and resize it to
                # have a maximum width of 300 pixels
                s, self.frame = self.vs.read()
                self.frame = imutils.resize(self.frame, width=600)
                (h, w) = self.frame.shape[:2]
                # construct a blob from the image
                imageBlob = cv2.dnn.blobFromImage(
                    cv2.resize(self.frame, (300, 300)), 1.0, (300, 300),
                    (104.0, 177.0, 123.0), swapRB=False, crop=False)

                # apply OpenCV's deep learning-based face detector to localize
                # faces in the input image
                detector.setInput(imageBlob)
                detections = detector.forward()

                # loop over the detections
                for i in range(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with
                    # the prediction
                    confidence = detections[0, 0, i, 2]
                    # filter out weak detections
                    if confidence > args["confidence"]:
                        # compute the (x, y)-coordinates of the bounding box for
                        # the face
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # extract the face ROI
                        face = self.frame[startY:endY, startX:endX]
                        (fH, fW) = face.shape[:2]

                        # ensure the face width and height are sufficiently large
                        if fW < 20 or fH < 20:
                            continue

                        # construct a blob for the face ROI, then pass the blob
                        # through our face embedding model to obtain the 128-d
                        # quantification of the face
                        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                         (96, 96), (0, 0, 0), swapRB=True, crop=False)
                        embedder.setInput(faceBlob)
                        vec = embedder.forward()

                        # perform classification to recognize the face
                        preds = recognizer.predict_proba(vec)[0]
                        j = np.argmax(preds)
                        proba = preds[j]
                        name = le.classes_[j]

                        # draw the bounding box of the face along with the
                        # associated probability
                        text = "{}: {:.2f}%".format(name, proba * 100)
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.rectangle(self.frame, (startX, startY), (endX, endY),
                                      (0, 0, 255), 2)
                        cv2.putText(self.frame, text, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)



                        if name != "unknown":
                            if proba*100 > 100:
                                self.master.maxConfidence = proba
                                self.master.recognizedPerson = name
                                self.master.recognizedConfidence = proba
                                self.switchBetweenScenes(WelcomeInTheSystemPage)

                            if self.master.maxConfidence < proba:
                                self.master.maxConfidence = proba
                                self.master.recognizedPerson = name
                                self.master.recognizedConfidence = proba
                                print(str(proba) + " " + name)

                self.frame = imutils.resize(self.frame, width=500)

                # OpenCV represents images in BGR order; however PIL
                # represents images in RGB order, so we need to swap
                # the channels, then convert to PIL and ImageTk format
                image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)

                # if the panel is not None, we need to initialize it
                if self.panel is None:
                    self.panel = tk.Label(image=image)
                    self.panel.image = image
                    self.panel.pack(side="left", padx=10, pady=10)

                # otherwise, simply update the panel
                else:
                    self.panel.configure(image=image)
                    self.panel.image = image



        except RuntimeError as e:
            print("[INFO] caught a RuntimeError")

        #print("rozpoznano na " + str(self.master.maxConfidence) + "% " + self.master.recognizedPerson)

        if (self.master.maxConfidence*100) > 40 and (self.master.maxConfidence*100) < 80:
            print("rozpoznano na " + str(self.master.maxConfidence) +"%")
            self.switchBetweenScenes(AuthPage)
        else:
            print("nie rozpoznano")
            self.switchBetweenScenes(EndPage)





    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        self.vs.release()
        self.quit()


cv2.destroyAllWindows()
