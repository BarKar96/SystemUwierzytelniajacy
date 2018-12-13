import datetime

from PIL import Image
from PIL import ImageTk
import tkinter as tk
import threading
import sys
import time

import cv2
import imutils

import firebase_module
import os


class CreateAccountSampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self._frame = None
        self.switch_frame(StartPage)
        self.nick = None

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
        tk.Label(self, text="Welcome in the registration system!").pack(side="top", fill="x", pady=10)
        tk.Button(self, text="Register",
                 command=lambda: master.switch_frame(SignInPage)).pack(side="top", fill="x", pady=10)
        tk.Button(self, text="Exit",
                  command=lambda: self.exitFromApp()).pack(side="top", fill="x", pady=10)

    def exitFromApp(self):
        sys.exit()

class SignInPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Label(self, text="Welcome in the registration system!").pack(side="top", fill="x", pady=10)


        nick = tk.StringVar()
        email = tk.StringVar()
        password = tk.StringVar()
        tk.Label(self, text="Enter your nickname:").pack(side="top", fill="x", pady=10)
        self.entry_box0 = tk.Entry(self, textvariable=nick, width=25, bg="lightblue")
        self.entry_box0.pack()

        tk.Label(self, text="Enter your email address:").pack(side="top", fill="x", pady=10)
        self.entry_box1 = tk.Entry(self, textvariable=email, width=25, bg="lightblue")
        self.entry_box1.pack()
        tk.Label(self, text="Enter your password:").pack(side="top", fill="x", pady=10)
        self.entry_box2 = tk.Entry(self, textvariable=password, width=25, bg="lightblue")
        self.entry_box2.pack()

        tk.Button(self, text="Register",
                 command=lambda: self.signIn(email.get(),password.get(), nick.get())).pack(side="top", fill="x", pady=10)
        tk.Button(self, text="Exit",
                  command=lambda: self.exitFromApp()).pack(side="top", fill="x", pady=10)

    def signIn(self, email, password, nick):
        f = open("emails.txt", "a+")
        f.write(nick + " " + email)
        f.write("\n")
        self.master.nick = nick
        firebase_module.createClient(email,password)
        self.master.switch_frame(VideoPage)



    def exitFromApp(self):
        sys.exit()

class VideoPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Label(self, text="Welcome in the registration system!").pack(side="top", fill="x", pady=5)
        tk.Label(self, text="Take 10 photos of yourself to successful register").pack(side="top", fill="x", pady=5)
        self.label = tk.Label(self, text="You already took: 0/10 photos")
        self.label.pack(side="top", fill="x", pady=5)

        btn = tk.Button(self, text="Take picture",
                         command=self.takeSnapshot)
        btn.pack(side="bottom", fill="both", expand="yes", padx=10,
                 pady=10)

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

        self.picture_counter = 0;

    def exitFromApp(self):
        self.onClose()

    def videoLoop(self):
        self.master.maxConfidence=0
        self.master.recognizedPerson = None
        self.master.recognizedConfidence = None

        # I'm not a GUI developer, nor do I even pretend to be. This
        # try/except statement is a pretty ugly hack to get around
        # a RunTime error that Tkinter throws due to threading
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():
                s, self.frame = self.vs.read()

                self.frame = imutils.resize(self.frame, width=700)

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

    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        self.vs.release()
        self.quit()

    def takeSnapshot(self):

        if not os.path.exists("dataset/" + self.master.nick):
            os.mkdir("dataset/" + self.master.nick)
            print("Directory ", "dataset/" + self.master.nick, " Created ")
        else:
            print("Directory ", "dataset/" + self.master.nick, " already exists")

        filename = "{}.jpg".format("0000" + str(self.picture_counter))
        self.picture_counter = self.picture_counter + 1
        outputPath="dataset/" + self.master.nick + "/"
        p = cv2.os.path.sep.join((outputPath, filename))
        # save the file
        cv2.imwrite(p, self.frame.copy())
        print("[INFO] saved {}".format(filename))

        self.label.config(text="You already took " + str(self.picture_counter) + "/10 photos")

        if self.picture_counter == 10:
            self.switchBetweenScenes(EndPage)


    def switchBetweenScenes(self, page_name):
        self.stopEvent.set()
        self.vs.release()
        if self.panel != None:
            self.panel.pack_forget()
        self.master.switch_frame(page_name)
        print("scene switched")


class EndPage(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        tk.Label(self, text="User has been successfully registered!").pack(side="top", fill="x", pady=10)

        tk.Button(self, text="Exit",
                  command=lambda: self.exitFromApp()).pack(side="top", fill="x", pady=10)

    def exitFromApp(self):
        sys.exit()