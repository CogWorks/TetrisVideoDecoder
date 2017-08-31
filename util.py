'''
    File name: util.py
    Copyright: Copyright 2017, The CogWorks Laboratory
    Author: Hassan Alshehri
    Email: alsheh@rpi.edu or eng.shehriusa@gmail.com
    Data created: May 22, 2017
    Date last modified: August 26, 2017
    Description: Utilities Module for TVD.py
    Status: Research
    Requirements/Dependencies:
        1. Python 2.7,
        2. OpenCV 3 (may not work with OpenCV 2),
        3. NumPy
'''


import matplotlib
import math
import cv2
import numpy as np
from Tkinter import *
import tkMessageBox

class GameInfo:
    '''
    Stores various meta data about the current game.
    '''
    def __init__(self):
        self.AVERAGE_INTENSITY = 7
        self.IN_VIDEO_FILE = None
        self.OUT_DATA_FILE = None
        self.WHITE_PIXLES_PERCENTAGE = None
        self.DETECTION_WINDOW = None
        self.RECREATE_GAME = None
        self.SHOW_VIDEO = None
        self.FPS = None
        self.TOTAL_FRAMES = 0
        self.INFINITY = float('inf')
        self.WIDTH = 10
        self.HEIGHT = 20
        self.MAX_SQUARES = self.WIDTH * self.HEIGHT
        self.BOARD = [ [2] * self.WIDTH for i in range(self.HEIGHT)]
        self.PLAYER_SCREEN_DIMENSIONSN = (None, None, None, None)
        self.SCALES = [-0.8, -0.6, -0.4, -0.2, 0 ,0.2, 0.4, 0.6, 0.8, 1, 1.2]
        self.SCREEN_LOW = [0, 0, 0]
        self.SCREEN_HIGH = [179, 255, 60]
        self.DIGIT_LOW = [0, 0, 0]
        self.DIGIT_HIGH = [179, 255, 132]
        self.digitData = []
        self.TEST_MODE = False
        self.previousFrameRowSum = 0
        self.DIGIT_RECOGNITION = []
        self.episode = 0
        self.START_FRAME = 0
        self.END_FRAME = float('inf')
        self.WRITING_MODE = None
        self.COLOR_DATA = None
        self.WRITE_COLOR_DATA = False

def MSG_Box(question=False, error=False, info=False, title=None, msg=None):
    '''
    Message boxes to display to the user
    '''
    root = Tk()
    root.withdraw()
    answer = None

    if info:
        tkMessageBox.showinfo(title, msg, parent=root)
    elif error:
        tkMessageBox.showerror(title, msg, parent=root)
    elif question:
        answer = tkMessageBox.askquestion(title, msg, parent=root)

    root.destroy()
    return answer

class NumberParser:
    '''
    This class works as an API for the OCR class which
    is implemented in the CharacterRcognition module.
    It also stores the location of data in the frame.
    '''
    def __init__(self, name, frame, ocr, game):
        self.ocr = ocr
        self.game = game
        self.name = name

        # Let the user crop the data from the frame and
        # it store its location.
        c = Crop()
        roi, x, y, w, h = c.crop(frame, taskName=name)
        self.location = x, y, w, h

        roi = frame[y:y+h, x:x+w]

        # Let the user crop the first digit in the data and
        # store its location.
        c = Crop(defaultZoom=0, maxZoom=20, isZoomAllowed=False)
        w, h = roi.shape[1::-1]
        scale = float(135)/h
        roi = cv2.resize(roi, (int(w*scale), int(h*scale)))
        char, x, y, w, h = c.crop(roi, taskName="first digit")
        roi = roi[y:y+h, x:x+w]
        cv2.imshow('First Digit', roi)
        answer = MSG_Box(question=True, title='First Digit', msg="Do you want to continue?")
        cv2.destroyWindow('First Digit')
        cv2.waitKey(1)
        if answer == 'no':
            exit(1)

        self.firstChar = x, y, w, h
        self.data = None

    def readNumber(self, frame):
        # Crop data from frame
        x, y , w, h = self.location
        scoreBox = frame[y:y+h, x:x+w]
        # Read digits from the cropped data
        resultString, resultImg, imgThresh = self.ocr.readCharacters(scoreBox, game=self.game, firstChar=self.firstChar)

        if resultString == '':
            resultString = 'X'

        if self.game.TEST_MODE:
            imgThresh = cv2.cvtColor(imgThresh, cv2.COLOR_GRAY2BGR)
            cv2.imshow(self.name, np.hstack([resultImg, imgThresh]))

        # Store numerical data
        self.data = resultString

    def putDataOnImage(self, image):
        x, y , w, h = self.location
        if self.data.strip() == 'X':
            cv2.rectangle(image, (x,y), (x+w, y+h), (0, 0, 255), 4)
            cv2.putText(image, self.data , (x, y), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0),10)
            cv2.putText(image, self.data, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 255),4)
        else:
            cv2.rectangle(image, (x,y), (x+w, y+h), (0, 255, 255), 4)
            cv2.putText(image, self.data , (x, y), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 0, 0),10)
            cv2.putText(image, self.data, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 255),4)


class Visualization:
    '''
    This class is used to visualize data extracted from frames.
    '''
    def __init__(self, game, grid):
        self.game = game
        self.sliderPosition = game.START_FRAME
        self.didSliderChangePosition = True
        self.grid = grid

        self.showVideoWindow = game.SHOW_VIDEO
        self.showDetectionWindow = game.DETECTION_WINDOW
        self.showGameRecreation = game.RECREATE_GAME

        # Make a real time video window
        if self.showVideoWindow:
            windowName = 'Video'
            cv2.namedWindow(windowName)
            self.videoWindow = Zoom(windowName, defaultZoom=-1)

        # Make a window for the real time game board
        if self.showDetectionWindow:
            windowName = 'Detection Window'
            self.detectionWindow = Window(windowName, game=game, grid=grid, mouse=True)

        # Make a window to show real time game recreation.
        if self.showGameRecreation:
            windowName = 'Game Recreation'
            self.gameRecreationWindow = Window(windowName, game=game, grid=grid)

    def visualize(self, frame, screen,  colorImg, blackWhiteImg, board, f, nxtZoid, recreatedBoard):
        # Show video window
        if self.showVideoWindow:
            # Draw the detected next zoid letter
            nxtZoid.drawBoxFrame(frame)

            x, y = self.grid.point1
            window = self.createVideoWindow(frame, x, y)

            for npr in self.game.digitData:
                npr.putDataOnImage(window)

            # Put current next zoid letter on frame
            nxtZoid.putZoidLetterOnImage(window)

            # Put current episode number on frame if test mode is not used.
            if not self.game.TEST_MODE:
                x, y, w, h = self.game.PLAYER_SCREEN_DIMENSIONS
                text = "Ep.#" + str(self.game.episode)
                cv2.putText(window, text, (x,y-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 9)
                cv2.putText(window, text, (x,y-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 4)

            # Show modified frame
            self.videoWindow.showZoomedImage(window, self.game.DELAY)

        # Show detection window
        if self.showDetectionWindow:
            gridOnScreen = self.detectionWindow.drawGrid(screen, board)
            self.detectionWindow.createScreenWindow(gridOnScreen, f, self.game.FPS)
            self.detectionWindow.showImage(self.game.DELAY)

        # Show game recreation window
        if self.showGameRecreation and recreatedBoard is not None:
            # create a window to show the color of the detected squares in the screen
            self.gameRecreationWindow.createScreenWindow(recreatedBoard, f, self.game.FPS)
            self.gameRecreationWindow.showImage(self.game.DELAY)

        # Show the black and white image of the screen
        if self.game.TEST_MODE:
            blackWhiteImg = cv2.cvtColor(blackWhiteImg, cv2.COLOR_GRAY2BGR)
            cv2.imshow("Scrren", np.hstack([colorImg, blackWhiteImg]))

    def createVideoWindow(self, frame, x, y):
        width = frame.shape[1]
        height = frame.shape[0]
        frame = frame.copy()
        x2, y2, w2, h2 = self.game.PLAYER_SCREEN_DIMENSIONS
        cv2.rectangle(frame, (x2,y2), (x2+w2, y2+h2), (0, 255, 255), 4)
        return frame

    def destroyAllWindows(self):
        cv2.destroyAllWindows()
        cv2.waitKey(1)


class Grid:
    '''
    This class stores information about the prebuilt grid.
    It also draws a grid on the frame and marks cells where squares are detected.
    '''
    def __init__(self, gridBuild):
        self.zoomLevel = 0
        self.numOfCellsHorizontally = gridBuild.numOfCellsHorizontally
        self.numOfCellsVertically = gridBuild.numOfCellsVertically
        self.point1 = gridBuild.point1
        self.point2 = gridBuild.point2
        self.cellWidth = gridBuild.cellWidth
        self.cellHeight = gridBuild.cellHeight
        self.gridColor = gridBuild.gridColor
        self.zoomLevel = gridBuild.image.zoomLevel/float(10)

    def getGridDimensions(self):
        x1, y1 = self.point1
        x2, y2 = self.point2
        x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        x, y, w, h = x1, y1, abs(x1-x2), abs(y1-y2)
        return 0, 0, w, h

    def drawGrid(self, image, bd):
        '''
        Draws a grid on the image and mark cells where squares are detected.
        '''
        # Make a copy of the image because this funtion
        # modifes the image by drawing the grid on image.
        img = image.copy()

        # Get the size of the board and cells.
        x, y, w, h = self.getGridDimensions()
        cWidth, cHeight = self.cellWidth, self.cellHeight
        verticalSteps = self.numOfCellsVertically
        horizontalSteps = self.numOfCellsHorizontally

        # Make the boundary of the screen colorful
        cv2.rectangle(img, (x,y), (w, h), (0, 255, 255), 2)

        # Iterate over all the cells in the grid/board.
        for j in range(verticalSteps):
            jPix = y +  j * cHeight # upper left corner j pixle coordinate
            for i in range(horizontalSteps):
                iPix = x + i * cWidth # upper left corner i pixle coordinate
                square = img[jPix:jPix+cHeight, iPix:iPix+cWidth]
                cv2.rectangle(img,(iPix,jPix),(iPix+cWidth,jPix+cHeight), self.gridColor, 1)

                if bd[j][i] == 1:
                    # Mark the current cell to indicate that a square is detected by drawing
                    # little circel inside the cell
                    tmpX, tmpY = iPix+cWidth/2, jPix+cHeight/2
                    r = min(cWidth/3, cHeight)
                    assert r > 2, "Image/grid is too small; visulaization window cannot be used."
                    tmpImg = square[r:r*2, r:r*2] # this will fail when we have rectangular shape peices
                    mean = cv2.mean(tmpImg)[:3]
                    cv2.circle(img,(tmpX,tmpY), r, (0,0,0), -1)
                    cv2.circle(img,(tmpX,tmpY), r-1, (255,130,255), -1)

        return img


class GridBuild:
    '''
    Interactive grid builder.
    '''
    def __init__(self, image, windowName='Grid', w=10, h=20):
        self.numOfCellsHorizontally = w
        self.numOfCellsVertically = h
        self.mousePreviousLocation = (0, 0)
        self.point1 = (0, 0)
        self.point2 = (0, 0)
        self.cellWidth = 0
        self.cellHeight = 0
        self.isCellSquareShape=True
        self.isDrawing = False
        self.isMovingGrid = False
        self.isResizingGrid = False
        self.gridColor = (0,255,0)
        self.windowName = windowName
        cv2.namedWindow(windowName)
        cv2.setMouseCallback(windowName, self.mouseEvents)
        cv2.createTrackbar('Grid color', windowName, 20, 15*3, self.changeGridColor)
        self.image = Zoom(windowName, image=image)
        self.r = 5
        self.resizePoints = [(w/2,0), (0,h/2), (w/2,h), (w, h/2), (w,h)]
        self.imgSize = image.size
        self.pinnedPoint = 1

    def printInstructions(self):
        msg1 = "Please draw a grid such that all squares fit in their corresponding cells."
        msg2 = " Follow the instructions below:"
        msg3 = "\n1. Press Shift+Mouse-Left-Button and drag to draw a new grid."
        msg4 = "\n2. Press Mouse-Left-Button on the newly built grid and drag to move the grid."
        msg5 = "\n3. To redraw the grid, repeat step 1."
        msg6 = "\n4. To adjust the size of the grid proportionally, drag the green handle."
        msg7 = "\n5. To adjust the size of the grid disproportionally, drag one of the blue handles."
        msg8 = "\n6. Prees 'c' at any time to finish building the grid."

        m = '\n\n' + msg1 + msg2 + msg3 + msg4 + msg5 + msg6 + msg7 + msg8
        print m
        m += '\n\nNote: this instructions is pritned in the terminal window for reference.'
        MSG_Box(info=True, title='GridBuild', msg=m)

    def createGrid(self, origX, origY):
        key = self.showImage(1)
        self.printInstructions()
        while True:
            key = self.showImage(1)
            if key == ord('c'):
                if self.point1 == self.point2:
                    msg = "ERROR: grid was not drawn: try again."
                    MSG_Box(error=True, title= 'ERROR', msg=msg)
                elif self.isGridOutOfBound():
                    msg = "ERROR: drawn grid seems to be out of bound: try again."
                    MSG_Box(error=True, title= 'ERROR', msg=msg)
                else:
                    x, y, w, h = self.finish(origX, origY)
                    return x, y, w, h

    def finish(self, origX, origY):
        '''
        Finalize building the grid.
        '''
        x, y, w, h = self.getGridDimensions()
        self.point1 = (x, y)
        self.point2 = x+w, y+h
        self.updateGridInfo()
        point1 = (x, y)
        point2 = (x+w, y+h)
        wSteps = self.numOfCellsHorizontally
        hSteps = self.numOfCellsVertically
        d = min(wSteps, hSteps)
        z = self.image.zoomLevel
        w = wSteps*self.cellWidth
        h = hSteps*self.cellHeight
        x, y, w, h = self.getAndSetOriginalGridDimensions()
        self.image.zoomLevel = z
        x, y, w, h = origX+x, origY+y, w, h
        self.point1 = point1
        self.point2 = point2
        return x, y, w, h

    def showImage(self, n):
        return self.image.showImage(n)

    def changeGridColor(self, x):
        '''
        Trackbar for choosing different colors for the grid.
        '''
        channel = x/15
        if channel >= 3:
            BGR = [255, 255, 255]
        else:
            BGR = [0, 0, 0]
            BGR[channel] = 100
            for i in range(x%15):
                if BGR[channel] < 250:
                    BGR[channel] += 50
                else:
                    channel = (channel+1)%3
        self.gridColor = tuple(BGR)
        self.drawGrid()

    def isGridOutOfBound(self):
        '''
        Check if any part of the grid if outside the image
        Retrun:
          True if gird is not completely inside the image and False otherwise.
        '''
        x, y, w, h = self.getGridDimensions()
        img = self.image.getImage()
        maxHeight = img.shape[0]
        maxWidth = img.shape[1]
        if x < 0 or y < 0:
            return True
        elif x+w >= maxWidth or y+h >= maxHeight:
            return True
        else:
            return False

    def hasZoomChanged(self):
        '''
        Check if the user has changed the zoom level of the image.

        Return:
           True if user changed zoom and False otherwise.
        '''
        currentImg = self.image.getImage()
        if self.imgSize != currentImg.size:
            self.imgSize = currentImg.size
            return True
        return False

    def getAndSetOriginalGridDimensions(self):
        '''
        Get original grid dimensions (i.e. dimensions before zooming)
        '''
        p1, p2 = self.image.mapResizedImagePixlesToOriginalImage(self.point1, self.point2)
        self.point1, self.point2 = p1, p2
        x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
        x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        x, y, w, h = x1, y1, abs(x1-x2), abs(y1-y2)
        return x, y, w, h

    def getGridDimensions(self):
        '''
        Get grid Dimensions
        '''
        x1, y1, x2, y2 = self.point1[0], self.point1[1], self.point2[0], self.point2[1]
        x, y, w, h = x1, y1, (x2-x1), (y2-y1)
        return x, y, w, h

    def updateGridInfo(self):
        # Update cell size
        x, y, w, h = self.getGridDimensions()
        numCells = self.numOfCellsVertically * self.numOfCellsHorizontally
        if self.isCellSquareShape:
            boardArea = abs(w * h)
            cellArea = boardArea/float(numCells)
            cellWidth = int(round(math.sqrt(cellArea)))
            cellHeight = int(round(math.sqrt(cellArea)))
            ##################
            if w < 0:
                cellWidth = -cellWidth
            if h < 0:
                cellHeight = -cellHeight
            ##################
        else:
            cellWidth = int(round(w/self.numOfCellsHorizontally))
            cellHeight = int(round(h/self.numOfCellsVertically))
        self.cellWidth, self.cellHeight = cellWidth, cellHeight

        # Update grid start and end points
        w = self.numOfCellsHorizontally * cellWidth
        h = self.numOfCellsVertically * cellHeight
        if self.pinnedPoint != 2:
            self.point2 = (x+w, y+h)

    def isMouseOnResizingPoints(self, i, j):
        '''
        Check if mouse is on the blue or green handles.
        Return:
            True if mouse on handles and False otherwise.
        '''
        x, y, w, h = self.getGridDimensions()
        cWidth, cHeight = self.cellWidth, self.cellHeight
        r = self.r
        epsilon = int(r * 1.5)
        self.mouseLocation = i, j

        for p in self.resizePoints:
            tmpX, tmpY = (x + p[0] * cWidth), (y + p[1] * cHeight)
            if (tmpX - r - epsilon) < i  < (tmpX + r + epsilon) and\
               (tmpY - r - epsilon) < j < (tmpY + r + epsilon):
                self.resizeControlPoint = p
                return True
        return False

    def resizeGrid(self, i, j):
        '''
        Resize grid in reponse to moving any of the blue or green handle.
        '''
        x, y, w, h = self.getGridDimensions()
        cWidth, cHeight = self.cellWidth, self.cellHeight
        verticalCells = self.numOfCellsVertically
        horizontalCells = self.numOfCellsHorizontally
        p = self.resizeControlPoint
        r = self.r

        self.mouseLocation = i, j
        if p[0] <= horizontalCells/2 and p[1] <= verticalCells/2: # these are the resize controls for p1
            self.pinnedPoint = 2
            # if this is the green resize control
            if p == (0, verticalCells/2):
                self.point1 = (i, self.point1[1])
                self.isCellSquareShape=False
            elif p == (horizontalCells/2, 0):
                self.point1 = (self.point1[0], j)
                self.isCellSquareShape=False
        else:
            self.pinnedPoint = 1
            if p == (horizontalCells, verticalCells):
                self.point2 = (i, j)
                self.isCellSquareShape=True
            elif p == (horizontalCells, verticalCells/2):
                self.point2 = (i, self.point2[1])
                self.isCellSquareShape=False
            elif p == (horizontalCells/2, verticalCells):
                self.point2 = (self.point2[0], j)
                self.isCellSquareShape=False

    def moveGrid(self, x, y):
        '''
        Move grid when user presses mouse left button on grid and drag.
        '''
        x2, y2 = self.mousePreviousLocation
        self.mousePreviousLocation = x, y
        xDelta, yDelta = x-x2, y-y2
        p1 = self.point1[0]+xDelta, self.point1[1]+yDelta
        p2 = self.point2[0]+xDelta, self.point2[1]+yDelta
        self.point1 = p1
        self.point2 = p2

    def getStartPoint(self):
        '''
        Return coordinates of the location where the user started drawing.
        '''
        x, y, w, h = self.getGridDimensions()
        if self.pinnedPoint == 1:
            return x, y
        elif self.pinnedPoint == 2:
            x = x+w - self.numOfCellsHorizontally * self.cellWidth
            y = y+h - self.numOfCellsVertically * self.cellHeight
            return x, y

    def drawGrid(self):
        '''
        Draw grid on image.
        '''
        self.updateGridInfo()
        img = self.image.getCleanImage()

        # Get grid dimensions
        x, y = self.getStartPoint()
        cWidth, cHeight = self.cellWidth, self.cellHeight
        verticalSteps = self.numOfCellsVertically
        horizontalSteps = self.numOfCellsHorizontally

        # Draw the cells of the grid
        for j in range(verticalSteps):
            jPix = y +  j * cHeight # upper left corner j pixle coordinate
            for i in range(horizontalSteps):
                iPix = x + i * cWidth # upper left corner i pixle coordinate
                tmpX, tmpY = iPix, jPix
                cv2.rectangle(img,(tmpX,tmpY),(tmpX+cWidth,tmpY+cHeight), self.gridColor, 1)

        # Draw the handles on the grid
        r = self.r
        for p in self.resizePoints:
            tmpX, tmpY = (x + p[0] * cWidth), (y + p[1] * cHeight)
            if p[0]%horizontalSteps == 0 and p[1]%verticalSteps == 0:
                cv2.circle(img,(tmpX,tmpY), r,(200,200,200),-1)
                cv2.circle(img,(tmpX,tmpY), r-1, (0,204,0),-1)
            else:
                cv2.circle(img,(tmpX,tmpY), r,(200,200,200),-1)
                cv2.circle(img,(tmpX,tmpY), r-1, (255,144,30),-1)
        return img


    def isMouseOnGrid(self, x, y):
        '''
        Return True if mouse is on grid and False otherwise.
        '''
        x1, y1, x2, y2 = self.point1[0], self.point1[1], self.point2[0], self.point2[1]
        x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
        else:
            return False

    def mouseEvents(self, event, x, y, flags, param):
        # Reset grid dimensions if zoom has changed.
        if self.hasZoomChanged():
            self.point1 = (0, 0)
            self.point2 = (0, 0)

        # If Shift+Mouse_Left_Button is pressend, store start point
        if event == cv2.EVENT_LBUTTONDOWN and flags == cv2.EVENT_FLAG_SHIFTKEY+cv2.EVENT_LBUTTONDOWN:
            # Start drawing a new grid
            self.isDrawing = True
            self.isCellSquareShape = True
            self.point1 = (x,y)

        # If the left mouse button was clicked
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.isMouseOnResizingPoints(x, y):
                # Start resizing grid
                self.resizeGrid(x,y)
                self.drawGrid()
                self.isResizingGrid = True
            # While the mouse is moving, keep drawing the updated selected region.
            elif self.isMouseOnGrid(x, y):
                # Start moving grid
                self.isMovingGrid = True
                self.mousePreviousLocation = x, y

        # If mouse is moving
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.isMovingGrid:
                # Keep moving grid
                self.moveGrid(x, y)
                self.drawGrid()
            elif self.isDrawing:
                # Kepp drawing grid
                self.point2 = (x,y)
                self.drawGrid()
            elif self.isResizingGrid:
                # Keep resizing grid
                self.resizeGrid(x,y)
                self.drawGrid()

        # If the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            if self.isResizingGrid:
                # Stop resizing the grid
                self.resizeGrid(x,y)
                self.drawGrid()
                self.isResizingGrid = False
                self.pinnedPoint = 1
            elif self.isDrawing:
                # Stop drawing the grid
                self.point2 = (x,y)
                self.drawGrid()
                self.isDrawing = False
            elif self.isMovingGrid:
                # Stop moving the grid
                self.moveGrid(x,y)
                self.isMovingGrid = False


class Zoom:
    '''
    This class makes an image zoomable (i.e. zoom in/out).
    '''
    def __init__(self, windowName, image=None, defaultZoom=0, maxZoom=10, minZoom=5):
        self.windowName = windowName
        self.maxZoom = maxZoom
        self.minZoom = minZoom
        self.zoomLevel = defaultZoom * 2
        self.isZoomActive = True
        if image is not None:
            self.originalImage = image
            self.currentZoomedImage = image
            self.selectedImg = image.copy()
        else:
            self.originalImage = None
            self.currentZoomedImage = None
            self.selectedImg = None

        self.scale = 10
        defaultZoom += minZoom
        cv2.createTrackbar('Zoom In/Out', windowName, defaultZoom, (maxZoom + minZoom + 1)-1, self.zoomInAndOut)

    def setImage(self, img):
        self.originalImage = img
        img = self.resizeImg()
        self.currentZoomedImage = img
        self.selectedImg = img.copy()

    def showImage(self, n):
        cv2.imshow(self.windowName, self.selectedImg)
        return cv2.waitKey(n)

    def showZoomedImage(self, img, n):
        img = self.resizeImg(origImg=img)
        cv2.imshow(self.windowName, img)
        return cv2.waitKey(n)

    def getCleanImage(self):
        self.reset()
        return self.getImage()

    def getImage(self):
        return self.selectedImg

    def getOriginalImage(self):
        return self.originalImage.copy()

    def setOriginal(self):
        self.zoomLevel = 0
        self.selectedImg = self.originalImage.copy()

    def reset(self):
        self.selectedImg = self.currentZoomedImage.copy()

    def resizeImg(self, origImg=None):
        '''
        Resize the image according to the current zoom level.
        '''
        if origImg is None:
            origImg = self.originalImage
        x = self.zoomLevel
        width, height = origImg.shape[1], origImg.shape[0]
        w, h = width+x/float(10)*width, height+x/float(10)*height
        assert w > 1 and h > 1
        img = cv2.resize(origImg, (int(round(w)), int(round(h))))
        return img

    def zoomInAndOut(self, x):
        '''
        When the slider in a trackbar changes position, change the size of the image.
        '''
        if self.isZoomActive:
            x -= self.minZoom
            x *= 2
            self.zoomLevel = x
            if self.selectedImg is not None:
                img = self.resizeImg()
                self.currentZoomedImage = img
                self.selectedImg = self.currentZoomedImage.copy()

    def enableZoom(self):
        self.isZoomActive = True

    def disableZoom(self):
        self.isZoomActive = False

    def mapResizedImagePixlesToOriginalImage(self, p1, p2):
        '''
        Map pixle coordinates for zoomed image to the original image.
        '''

        zoomFactor = self.zoomLevel/float(10)
        origImg =  self.originalImage
        deltaWidth = zoomFactor * origImg.shape[1]
        deltaHeight = zoomFactor * origImg.shape[0]

        x1 = min(p1[0], p2[0])
        x2 = max(p1[0], p2[0])
        y1 = min(p1[1], p2[1])
        y2 = max(p1[1], p2[1])

        x1 -= float(x1)/self.currentZoomedImage.shape[1] * deltaWidth
        y1 -= float(y1)/self.currentZoomedImage.shape[0] * deltaHeight
        x2 -= float(x2)/self.currentZoomedImage.shape[1] * deltaWidth
        y2 -= float(y2)/self.currentZoomedImage.shape[0] * deltaHeight

        self.setOriginal()
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

        return (x1, y1), (x2, y2)


class Window:
    '''
    This class is used to create a customized window that contains the screen/board.
    The class allows the window to respond to mouse envents when user interaction is allowed.
    '''
    def __init__(self, windowName, game=None, grid=None, windowDefaultZoom=0, mouse=False):
        self.windowName = windowName
        self.game = game
        self.grid = grid
        self.image = None
        self.gridWidth = None
        self.gridHeight = None
        self.point1 = None
        self.point2 = None
        self.width = None
        self.height = None
        self.hasChanged = True
        cv2.namedWindow(windowName)
        self.zoom = Zoom(windowName, defaultZoom=windowDefaultZoom)
        if mouse:
            cv2.setMouseCallback(windowName, self.mouseEvents)

    def drawGrid(self, img, bd):
        return self.grid.drawGrid(img, bd)

    def createScreenWindow(self, screen, f, fps):
        infoHeight = int(screen.shape[0] * 0.2)
        gridWidth = screen.shape[1]
        gridHeight = screen.shape[0]
        pt1 = (0, infoHeight)
        pt2 = (pt1[0]+gridWidth, pt1[1]+gridHeight)
        width = gridWidth
        height = gridHeight + pt1[1]
        window = np.zeros((height,width,3), np.uint8)
        totalFrames = self.game.TOTAL_FRAMES
        whiteColor = (255,255,255)
        f = f - self.game.START_FRAME + 1

        # Decide the appropriate font size
        # for the current image based on its size.
        million = float(1000000)
        if window.size > million:
            fontSize = 1.25
            thickness = 2
        elif window.size > int(million * 0.75):
            fontSize = 1
            thickness = 2
        elif window.size > int(million * 0.5):
            fontSize = 0.75
            thickness = 1
        else:
            fontSize = 0.5
            thickness = 1

        window[pt1[1]:pt2[1], pt1[0]:pt2[0]] = screen[0:gridHeight, 0:gridWidth]

        # Display progress percentage on the upper part of the window
        x, y = int(width * 0.01), int(height * 0.05)
        text = "Progress:%%%.3f" %(f/float(totalFrames) * 100)
        cv2.putText(window, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, fontSize, whiteColor, thickness)

        # Display number of frames on the upper part of the window
        x, y = int(width * 0.01), int(height * 0.1)
        text = "Frame#" + str(f)
        cv2.putText(window, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, fontSize, whiteColor, thickness)

        # Display frame rate on the window
        x, y = int(width * 0.01), int(height * 0.15)
        text = "FPS: "+str(fps)
        cv2.putText(window, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, fontSize, whiteColor,thickness)

        self.point1 = pt1
        self.point2 = pt2
        self.image = window

    def showImage(self, n):
        return self.zoom.showZoomedImage(self.image, n)

    def getGridPoints(self):
        # Update grid start and end points
        x1, y1 = self.point1
        x2, y2 = self.point2
        zf = self.zoom.zoomFactor
        x1, y1 = int(x1+x1*zf), int(y1+y1*zf)
        x2, y2 = int(x2+x2*zf), int(y2+y2*zf)
        return x1, y1, x2, y2

    def getGridDimensions(self):
        x1, y1, x2, y2 = self.getGridPoints()
        x, y, w, h = x1, y1, abs(x1-x2), abs(y1-y2)
        return x, y, w, h

    def isMouseOnGrid(self, x, y):
        x1, y1, x2, y2 = self.getGridPoints()
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
        else:
            return False

    def mapCoordinatesToCell(self, i, j):
        '''
        Activate/Deactivate the cell that contains the (i, j) coordinates.
        '''
        if self.isMouseOnGrid(i,j):
            self.hasChanged = True
            # Update cell dimensions
            zf = self.zoom.zoomFactor
            cellWidth = self.grid.cellWidth + int(self.grid.cellWidth * zf)
            cellHeight = self.grid.cellHeight+ int(self.grid.cellHeight * zf)
            x, y, w, h = self.getGridDimensions()
            bd = self.game.BOARD
            i, j = i-x, j-y
            i, j = i, j
            i, j = i/cellWidth, j/cellHeight
            if bd[j][i]:
                bd[j][i] = 0
            else:
                bd[j][i] = 1

    def mouseEvents(self, event, x, y, flags, param):
        # If the left mouse button was clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mapCoordinatesToCell(x, y)

class Crop:
    '''
    This class is a tool for cropping images.
    '''
    def __init__(self, windowName='image', defaultZoom=0, maxZoom=10, isZoomAllowed=True):
        self.image = None
        self.cropping = False
        self.isCroppingDone = False
        self.point1 = None
        self.point2 = None
        self.isReselect = None
        self.windowName = windowName
        cv2.namedWindow(windowName)
        cv2.setMouseCallback(windowName, self.mouseEvents)
        self.image = Zoom(windowName=windowName, image=None, defaultZoom=defaultZoom, maxZoom=maxZoom)
        self.isZoomAllowed = isZoomAllowed

    def mouseEvents(self, event, x, y, flags, param):
        # If the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being performed.
        if event == cv2.EVENT_LBUTTONDOWN:
            self.isCroppingDone = False
            self.image.reset()
            self.isReselect = True
            self.cropping = True
            self.point1 = (x,y)

        # While the mouse is moving, keep drawing the updated selected region.
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.cropping:
                cv2.rectangle(self.image.getCleanImage(), self.point1, (x,y), (0,255,0), 2)

        # If the left mouse button was released,
        # draw the most recently selected region
        elif event == cv2.EVENT_LBUTTONUP:
            if self.cropping:
                cv2.rectangle(self.image.getCleanImage(), self.point1, (x,y), (0,255,0), 2)
                self.point2 = (x,y)
            self.cropping = False
            self.isCroppingDone = True

    def crop(self, image, taskName=None):
        # Create a zoomable image
        self.image.setImage(image)

        self.image.showImage(1)

        # Give user instructions
        msg = "Press mouse left button and drag to select \"" + taskName + "\" region"
        MSG_Box(info=True, title='Crop ' + taskName.capitalize(), msg=msg)

        while not self.isCroppingDone:
            if self.isZoomAllowed:
                self.image.enableZoom()
            else:
                self.image.disableZoom()
            # Start tracking mouse selection
            while not self.isCroppingDone:
                # Display the image
                self.image.showImage(1)

            self.image.disableZoom()
            self.isReselect = False

            # At this point the user has selected a region.
            # Give the user the option to reselect a different region or confirm the selection.
            msg = "Click 'Yes' to crop.\nClick 'No' to reset the selected region."
            answer = MSG_Box(question=True, title='Crop ' + taskName.capitalize(), msg=msg)

            key = self.image.showImage(1)
            # If the 'no' button is clicked, reset the cropping region.
            if answer == "no":
                self.isCroppingDone = False
                self.image.reset()
            # If the 'yes' button is clicked, crop the image.
            elif answer == "yes":
                self.isCroppingDone = True

        cv2.destroyWindow(self.windowName)
        cv2.waitKey(1)

        # If there are two reference points, then crop
        # the region of interest from the image and display it.
        if self.point1 is not None and self.point2 is not None:
            # Map pixle coordinates for zoomed image to the original image
            (x1, y1), (x2, y2) = self.image.mapResizedImagePixlesToOriginalImage(self.point1, self.point2)
            # Pixle coordianes of the original image
            x, y, w, h = int(x1), int(y1), abs(int(x2 - x1)), abs(int(y2 - y1))
            # Get the region of interest from the original image (ROI) (i.e user selected region)
            roi = self.image.getOriginalImage()[y:y+h+1, x:x+w+1]
        else:
            x,y,w,h = None, None, None, None
            roi = None

        # Return the cropped region as well as its location.
        return roi,x,y,w,h
