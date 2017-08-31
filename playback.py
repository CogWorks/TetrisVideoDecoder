'''
    File name: TVD.py
    Copyright: Copyright 2017, The CogWorks Laboratory
    Author: Hassan Alshehri
    Email: alsheh@rpi.edu or eng.shehriusa@gmail.com
    Data created: May 22, 2017
    Date last modified: August 26, 2017
    Description: Video Manual Playback
    Status: Research
    Requirements/Dependencies:
        1. Python 2.7,
        2. OpenCV 3 (may not work with OpenCV 2),
        3. NumPy
'''
import numpy as np
import cv2
from Tkinter import *
import tkMessageBox
from TVD import frameAnalysis
from PIL import Image
from PIL import ImageTk

class VideoWindow:
    '''
    This class creates a window and adds various
    functionalties to it depending on the specified mode.
    Functionalties in the test mode include settings for
    targeting the removal of the background color and
    jumping to different frames. In the non-test mode,
    a manual video playback is implemented and buttons to
    determine the start and end frames are included.
    '''
    def __init__(self, master, game, obj=None, start=0, end=0, windowName='Video', testMode=False):
        self.game = game
        self.didSliderChangePosition = True
        self.sliderPosition = start
        self.window = Label(master)
        self.window.grid(row=0, column=0)
        self.hasZoomChanged = False
        self.isFrameReady = True
        self.isThereNewUpdate = False
        self.zoomLevel = 5
        width = 639

        # Add frame slider
        Label(master, text="Rewind/Fastforward" ).grid(row=1, column=0)
        self.frameSlider = Scale(master, from_=start, to=end, length=width,\
                       orient=HORIZONTAL, command=self.jumpToFrameX)
        self.frameSlider.grid(row=2, column=0)


        if testMode:
            master.wm_title("Settings")

            # Add 'Next Frame' and 'Previous Frame' buttons
            f2 = Frame(master)
            f2.grid(row=3, column=0)
            b1 = Button(f2, text="Previous Frame", width = 11, command=obj.previousFrame)
            b1.grid(row=0,column=0, sticky='nsew')
            b2 = Button(f2, text="Next Frame", width = 11, command=obj.nextFrame)
            b2.grid(row=0,column=1, sticky='nsew')

            # Add sliders for targetting the color of the backgournd (Screen)
            Label(master, text="Screen Background Color Range").grid(row=4, column=0, sticky='nsew')
            f = Frame(master)
            f.grid(row=5, column=0, sticky='nsew')

            # Add HSV SCREEN hue lower slider
            Label(f, text="Screen Hue Lower").grid(row=0, column=0)
            self.hueSlider = Scale(f, from_=0, to=179, length=(width-2)/2,\
                                     orient=HORIZONTAL, command=self.hsvScreenHueLower)
            self.hueSlider.grid(row=1, column=0)
            self.hueSlider.set(game.SCREEN_LOW[0])

            # Add HSV SCREEN Saturation lower slider
            Label(f, text="Screen Saturation Lower").grid(row=2, column=0)
            self.saturationSlider = Scale(f, from_=0, to=255, length=(width-2)/2,\
                                     orient=HORIZONTAL, command=self.hsvScreenSaturationLower)
            self.saturationSlider.grid(row=3, column=0)
            self.saturationSlider.set(game.SCREEN_LOW[1])

            # Add HSV SCREEN Brightness lower slider
            Label(f, text="Screen Brighness Lower").grid(row=4, column=0)
            self.brightnessSlider = Scale(f, from_=0, to=255, length=(width-2)/2,\
                                          orient=HORIZONTAL, command=self.hsvScreenBrightnessLower)
            self.brightnessSlider.grid(row=5, column=0)
            self.brightnessSlider.set(game.SCREEN_LOW[2])


            # Add HSV SCREEN hue upper slider
            Label(f, text="Screen Hue Upper").grid(row=0, column=1)
            self.hueSlider = Scale(f, from_=0, to=179, length=(width-2)/2,\
                                     orient=HORIZONTAL, command=self.hsvScreenHueUpper)
            self.hueSlider.grid(row=1, column=1)
            self.hueSlider.set(game.SCREEN_HIGH[0])

            # Add HSV SCREEN Saturation upper slider
            Label(f, text="Screen Saturation Upper").grid(row=2, column=1)
            self.saturationSlider = Scale(f, from_=0, to=255, length=(width-2)/2,\
                                     orient=HORIZONTAL, command=self.hsvScreenSaturationUpper)
            self.saturationSlider.grid(row=3, column=1)
            self.saturationSlider.set(game.SCREEN_HIGH[1])

            # Add HSV SCREEN Brightness upper slider
            Label(f, text="Screen Brighness Upper").grid(row=4, column=1)
            self.brightnessSlider = Scale(f, from_=0, to=255, length=(width-2)/2,\
                                          orient=HORIZONTAL, command=self.hsvScreenBrightnessUpper)
            self.brightnessSlider.grid(row=5, column=1)
            self.brightnessSlider.set(game.SCREEN_HIGH[2])


            # Add sliders for targetting the color of the backgournd (Digit)
            Label(master, text="Digit Background Color Range").grid(row=6, column=0, sticky='nsew')

            f = Frame(master)
            f.grid(row=7, column=0, sticky='nsew')

            # Add HSV DIGIT hue lower slider
            Label(f, text="Digit Hue Lower").grid(row=0, column=0)
            self.hueSlider = Scale(f, from_=0, to=179, length=(width-2)/2,\
                                     orient=HORIZONTAL, command=self.hsvDigitHueLower)
            self.hueSlider.grid(row=1, column=0)
            self.hueSlider.set(game.DIGIT_LOW[0])

            # Add HSV DIGIT Saturation lower slider
            Label(f, text="Digit Saturation Lower").grid(row=2, column=0)
            self.saturationSlider = Scale(f, from_=0, to=255, length=(width-2)/2,\
                                     orient=HORIZONTAL, command=self.hsvDigitSaturationLower)
            self.saturationSlider.grid(row=3, column=0)
            self.saturationSlider.set(game.DIGIT_LOW[1])

            # Add HSV DIGIT Brightness lower slider
            Label(f, text="Digit Brighness Lower").grid(row=4, column=0)
            self.brightnessSlider = Scale(f, from_=0, to=255, length=(width-2)/2,\
                                          orient=HORIZONTAL, command=self.hsvDigitBrightnessLower)
            self.brightnessSlider.grid(row=5, column=0)
            self.brightnessSlider.set(game.DIGIT_LOW[2])

            # Add HSV DIGIT hue upper slider
            Label(f, text="Digit Hue Upper").grid(row=0, column=1)
            self.hueSlider = Scale(f, from_=0, to=179, length=(width-2)/2,\
                                     orient=HORIZONTAL, command=self.hsvDigitHueUpper)
            self.hueSlider.grid(row=1, column=1)
            self.hueSlider.set(game.DIGIT_HIGH[0])

            # Add HSV DIGIT Saturation upper slider
            Label(f, text="Digit Saturation Upper").grid(row=2, column=1)
            self.saturationSlider = Scale(f, from_=0, to=255, length=(width-2)/2,\
                                     orient=HORIZONTAL, command=self.hsvDigitSaturationUpper)
            self.saturationSlider.grid(row=3, column=1)
            self.saturationSlider.set(game.DIGIT_HIGH[1])

            # Add HSV DIGIT Brightness upper slider
            Label(f, text="Digit Brighness Upper").grid(row=4, column=1)
            self.brightnessSlider = Scale(f, from_=0, to=255, length=(width-2)/2,\
                                          orient=HORIZONTAL, command=self.hsvDigitBrightnessUpper)
            self.brightnessSlider.grid(row=5, column=1)
            self.brightnessSlider.set(game.DIGIT_HIGH[2])

        else:
            # Add 'Previous Frame', 'Next Frame', 'Zoom In', and 'Zoom Out' buttons.
            master.wm_title("Video")
            f = Frame(master)
            f.grid(row=4, column=0, sticky='nsew')
            Button(f, text="Previous Frame", width=11, command=obj.previousFrame).pack(side='left')
            Button(f, text="Next Frame", width=11, command=obj.nextFrame).pack(side='left')
            self.StartEndButton = Button(f, text='Match Begins', width=11, command=obj.selectFrame)
            self.StartEndButton.pack(side='left')
            Button(f, text="Zoom Out", width=11, command=self.zoomOut).pack(side='left')
            Button(f, text="Zoom In", width=11, command=self.zoomIn).pack(side='left')

    def hsvScreenHueUpper(self, x):
        self.game.SCREEN_HIGH[0] = int(x)
        self.isThereNewUpdate = True

    def hsvScreenSaturationUpper(self, x):
        self.game.SCREEN_HIGH[1] = int(x)
        self.isThereNewUpdate = True

    def hsvScreenBrightnessUpper(self, x):
        self.game.SCREEN_HIGH[2] = int(x)
        self.isThereNewUpdate = True

    def hsvScreenHueLower(self, x):
        self.game.SCREEN_LOW[0] = int(x)
        self.isThereNewUpdate = True

    def hsvScreenSaturationLower(self, x):
        self.game.SCREEN_LOW[1] = int(x)
        self.isThereNewUpdate = True

    def hsvScreenBrightnessLower(self, x):
        self.game.SCREEN_LOW[2] = int(x)
        self.isThereNewUpdate = True

    def hsvDigitHueUpper(self, x):
        self.game.DIGIT_HIGH[0] = int(x)
        self.isThereNewUpdate = True

    def hsvDigitSaturationUpper(self, x):
        self.game.DIGIT_HIGH[1] = int(x)
        self.isThereNewUpdate = True

    def hsvDigitBrightnessUpper(self, x):
        self.game.DIGIT_HIGH[2] = int(x)
        self.isThereNewUpdate = True

    def hsvDigitHueLower(self, x):
        self.game.DIGIT_LOW[0] = int(x)
        self.isThereNewUpdate = True

    def hsvDigitSaturationLower(self, x):
        self.game.DIGIT_LOW[1] = int(x)
        self.isThereNewUpdate = True

    def hsvDigitBrightnessLower(self, x):
        self.game.DIGIT_LOW[2] = int(x)
        self.isThereNewUpdate = True

    def changeAlgo(self):
        choice = int(self.v.get())
        if choice == 1:
            self.game.SHINY_SPOT_ALGO = False
        elif choice == 2:
            self.game.SHINY_SPOT_ALGO = True

    def adjustThreshold(self, x):
        self.game.THRESHOLD = int(x)
        self.isThereNewUpdate = True

    def speed(self, x):
        self.delay = 30 + 50 * (int(x)-1)

    def zoomOut(self):
        self.hasZoomChanged = True
        if self.zoomLevel > 1:
            self.zoomLevel -= 1
        self.isThereNewUpdate = True

    def zoomIn(self):
        self.hasZoomChanged = True
        if self.zoomLevel < 20:
            #print self.zoomLevel
            self.zoomLevel += 1
        self.isThereNewUpdate = True

    def zoom(self, x):
        self.hasZoomChanged = True
        self.zoomLevel = int(x)
        self.isThereNewUpdate = True

    def jumpToFrameX(self, x):
        self.didSliderChangePosition = True
        self.isFrameReady = True
        self.sliderPosition = int(x)

    def updateFrame(self, video, frameNumber):
        if self.didSliderChangePosition:
            self.didSliderChangePosition = False
            video.set(cv2.CAP_PROP_POS_FRAMES, self.sliderPosition)
            frameNumber = self.sliderPosition
        self.frameSlider.set(frameNumber)
        return frameNumber

    def showFrame(self, frame):
        # resize frame
        w, h = frame.shape[1], frame.shape[0]
        w, h = int(w * self.zoomLevel/10.0), int(h * self.zoomLevel/10.0)
        frame = cv2.resize(frame, (w, h))
        # convert frame from BGR to RGB
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.window.imgtk = imgtk
        self.window.configure(image=imgtk)


class Playback:
    '''
    This class implements the manual playback behavior.
    '''
    def __init__(self, video, grid, game, v, nxtZoid=None, testMode=False):
        self.master = Tk()
        self.game = game
        self.grid = grid
        self.v = v
        self.video = video
        self.currentFrame = None
        self.nxtZoid = nxtZoid
        self.totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.start = game.START_FRAME
        self.end = min(game.END_FRAME, self.totalFrames)
        self.frameNumber = self.start

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Set up video window
        self.mainWindow = VideoWindow(self.master, game, obj=self, start=self.start, end=self.end, testMode=testMode)

        # Start reading frames from vidoes.
        if testMode:
            self.read()
        else:
            self.readFrame()

        self.master.mainloop()

    def nextFrame(self):
        if self.frameNumber < self.end:
            self.frameNumber += 1
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.frameNumber)
            self.mainWindow.frameSlider.set(self.frameNumber)
            self.mainWindow.isFrameReady = True

    def previousFrame(self):
        if self.frameNumber > self.start:
            self.frameNumber -= 1
            self.video.set(cv2.CAP_PROP_POS_FRAMES, self.frameNumber)
            self.mainWindow.frameSlider.set(self.frameNumber)
            self.mainWindow.isFrameReady = True

    def selectFrame(self):
        status = self.mainWindow.StartEndButton['text']
        if status == 'Match Begins':
            self.start = self.frameNumber
            self.mainWindow.StartEndButton['text'] = 'Match Ends'
        elif status == 'Match Ends':
            self.end = self.frameNumber
            if self.end <= self.start:
                tkMessageBox.showerror("Error","The match must end after the match begins.")
                self.start = 0
                self.end = float('inf')
                self.mainWindow.StartEndButton['text'] = 'Match Begins'
            else:
                self.master.destroy()

    def readFrame(self):
        '''
        Read and display frames.
        '''
        if self.mainWindow.isFrameReady:
            self.mainWindow.isFrameReady = False
            self.frameNumber = self.mainWindow.updateFrame(self.video, self.frameNumber)
            ok, frame = self.video.read()
            if ok:
                self.currentFrame = frame
            else:
                frame = self.currentFrame
            self.mainWindow.showFrame(self.currentFrame)
        elif self.mainWindow.hasZoomChanged:
            self.mainWindow.hasZoomChanged = False
            self.mainWindow.showFrame(self.currentFrame)
        self.master.after(100, self.readFrame)

    def read(self):
        '''
        Read frames and apply the frameAnalysis function on each frame.
        '''
        if self.mainWindow.isFrameReady:
            self.mainWindow.isFrameReady = False
            self.frameNumber = self.mainWindow.updateFrame(self.video, self.frameNumber)
            ok, frame = self.video.read()
            if ok:
                self.currentFrame = frame
            else:
                frame = self.currentFrame
            frameAnalysis(self.currentFrame.copy(), self.grid, self.game, self.v, self.frameNumber, self.nxtZoid, None)
        elif self.mainWindow.isThereNewUpdate:
            self.mainWindow.isThereNewUpdate = False
            frameAnalysis(self.currentFrame.copy(), self.grid, self.game, self.v, self.frameNumber, self.nxtZoid, None)
        self.master.after(100, self.read)

    def play(self):
        self.frameNumber += 1
        self.frameNumber = self.mainWindow.updateFrame(self.video, self.frameNumber)
        ok, frame = self.video.read()
        if not ok:
            exit(0)
        self.mainWindow.showFrame(frame)
        frameAnalysis(frame, self.grid, self.game, self.v, self.frameNumber, self.nxtZoid)
        self.master.after(self.mainWindow.delay, self.play)

    def on_closing(self):
        if tkMessageBox.askokcancel("Quit", "Do you want to quit?"):
            self.master.destroy()
            sys.exit()


if __name__ == '__main__':
    '''
    The code below is for testing purposes.
    '''
    # Read video
    video = cv2.VideoCapture('./videos/Finals2016.mp4')

    # Exit if video not opened.
    if not video.isOpened():
        print "ERROR: Could not open video"
        exit(1)

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print 'ERROR: Cannot read video file'
        exit(1)

    p = Playback(video, None, None, None, manual=True)
