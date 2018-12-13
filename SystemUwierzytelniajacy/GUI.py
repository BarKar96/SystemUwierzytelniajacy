# import the necessary packages
from __future__ import print_function
from recognize_video import SampleApp
from CreateAccountApp import CreateAccountSampleApp
from imutils.video import VideoStream
import time




print("[INFO] warming up camera...")






# start the app
pba = SampleApp()
#pba = CreateAccountSampleApp()
pba.mainloop()