'''
    File name: TVD.py
    Copyright: Copyright 2017, The CogWorks Laboratory
    Author: Hassan Alshehri
    Email: alsheh@rpi.edu or eng.shehriusa@gmail.com
    Data created: May 22, 2017
    Date last modified: August 26, 2017
    Description: Next Zoid Detection
    Status: Research
    Requirements/Dependencies:
        1. Python 2.7,
        2. OpenCV 3 (may not work with OpenCV 2),
        3. NumPy
'''

import numpy as np
import cv2
from TVD import *

class NextZoid:
    '''
    This class uses "Template Matching" to detect the next zoid.
    Upon creating an instance of this class, a template is created
    for each Tetris letter. For each frame, the letter inside the
    next zoid box is being compared to the precreated templates.
    If the letter matches with any of the tamplates, the next zoid
    letter is known otherwise 'X' is used to indicate unknown letter.
    Unkowen letters could be when the next zoid box is empty or
    the letter is unidentifiable.
    '''
    def __init__(self, loc, game, grid):
        self.nextZoidLoc = loc
        self.game = game
        self.sTemplate = None
        self.zTemplate = None
        self.oTemplate = None
        self.iTemplate = None
        self.tTemplate = None
        self.jTemplate = None
        self.lTemplate = None

        # Create zoid templates
        self.createTemplates(grid)

        # Check that zoids templates were created
        self.nextZoidLoc is not None
        self.sTemplate is not None
        self.zTemplate is not None
        self.oTemplate is not None
        self.iTemplate is not None
        self.tTemplate is not None
        self.jTemplate is not None
        self.lTemplate is not None

        # Current next zoid letter
        self.nextZoidLetter = None

    def drawBoxFrame(self, frame):
        '''
        Draw a yellow rectangle around the next zoid box when
        a letter is detected successfully and red otherwise.
        '''
        x, y, w, h = self.nextZoidLoc
        nextZoidLetter = self.nextZoidLetter
        if nextZoidLetter != 'X':
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 4)
        else:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 4)

    def putZoidLetterOnImage(self, frame):
        '''
        Put the detected next zoid letter above its box.
        '''
        x, y, w, h = self.nextZoidLoc
        nextZoidLetter = self.nextZoidLetter
        if nextZoidLetter != 'X':
            cv2.putText(frame, nextZoidLetter, (x+w/2,y+10), cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 0, 0),9)
            cv2.putText(frame, nextZoidLetter, (x+w/2,y+10), cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 255, 255),4)
        else:
            cv2.putText(frame, 'X', (x+w/2,y+10), cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 0, 0),9)
            cv2.putText(frame, 'X', (x+w/2,y+10), cv2.FONT_HERSHEY_COMPLEX, 1.7, (0, 0, 255),4)


    def findNextZoid(self, box):
        '''
        Compare the letter in the box with the templates and
        store the letter if a match is found or 'X' otherwise.
        '''
        if self.matchTetrisShape(box, self.sTemplate):
            self.nextZoidLetter = 'S'
            return 'S'
        elif self.matchTetrisShape(box, self.zTemplate):
            self.nextZoidLetter = 'Z'
            return 'Z'
        elif self.matchTetrisShape(box, self.tTemplate):
            self.nextZoidLetter = 'T'
            return 'T'
        elif self.matchTetrisShape(box, self.lTemplate):
            self.nextZoidLetter = 'L'
            return 'L'
        elif self.matchTetrisShape(box, self.jTemplate):
            self.nextZoidLetter = 'J'
            return 'J'
        elif self.matchTetrisShape(box, self.oTemplate):
            self.nextZoidLetter = 'O'
            return 'O'
        elif self.matchTetrisShape(box, self.iTemplate):
            self.nextZoidLetter = 'I'
            return 'I'
        else:
            self.nextZoidLetter = 'X'
            return 'X'

    def matchTetrisShape(self, box, template):
        '''
        Compare the letter in box with the given tamplate.
        If they match, return True and False otherwise.
        '''
        colorImg, blackWhiteImg = removeNoise(box, self.game)

        # Make the edges in the image thicker and reverse that effect by applying
        # dilation and erosion respectively to the edged image.
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(blackWhiteImg,kernel,iterations = 1)
        kernel = np.ones((4,4),np.uint8)
        dilation = cv2.dilate(erosion,kernel,iterations = 2)
        blackWhiteImg = dilation

        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(blackWhiteImg, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7
        loc = np.where(res >= threshold)
        if len(loc[0]) != 0 and len(loc[1]) != 0:
            return True
        else:
            return False

    def createTemplates(self, grid):
        '''
        Create a template for each Tetris letter.
        '''
        # Get tetris square size
        cWidth = grid.cellWidth - 2
        cHeight = grid.cellHeight - 2
        separation = 2
        width = cWidth * 4 + cWidth/2 + 3 * separation
        height = cHeight * 2 + cHeight/2 + separation

        # Creating S Tetris template
        w, h = width-cWidth/2, height
        S = np.zeros((h,w,3), np.uint8)
        x, y = cWidth/2, cHeight + cHeight/4 + separation
        cv2.rectangle(S, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        x, y = x + cWidth + separation, y
        cv2.rectangle(S, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        x, y = x, y - cHeight - separation
        cv2.rectangle(S, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        x, y = x + cWidth + separation, y
        cv2.rectangle(S, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        self.sTemplate = S

        # Creating Z Tetris template
        w, h = width-cWidth/2, height
        Z = np.zeros((h,w,3), np.uint8)
        x, y = cWidth/2, cHeight/4
        cv2.rectangle(Z, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        x, y = x + cWidth + separation, y
        cv2.rectangle(Z, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        x, y = x, y + cHeight + separation
        cv2.rectangle(Z, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        x, y = x + cWidth + separation, y
        cv2.rectangle(Z, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        self.zTemplate = Z

        # Creating O Tetris template
        w, h = width-cWidth, height
        O = np.zeros((h,w,3), np.uint8)
        x, y = cWidth/2, cHeight/4
        cv2.rectangle(O, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        x, y = x + cWidth + separation, y
        cv2.rectangle(O, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        x, y = x, y + cHeight + separation
        cv2.rectangle(O, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        x, y = x - cWidth - separation, y
        cv2.rectangle(O, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        self.oTemplate = O

        # Creating I Tetris template
        #w, h = width, height-cHeight/2
        w, h = width-cWidth/2, height-cHeight/2
        I = np.zeros((h,w,3), np.uint8)
        #x, y = cWidth/4, cHeight/2
        x, y = 0, cHeight/2
        cv2.rectangle(I, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        x, y = x + cWidth + separation, y
        cv2.rectangle(I, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        x, y = x + cWidth + separation, y
        cv2.rectangle(I, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        x, y = x + cWidth + separation, y
        cv2.rectangle(I, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        self.iTemplate = I

        # Creating T Tetris template
        w, h = width-cWidth/2, height
        T = np.zeros((h,w,3), np.uint8)
        x, y = cWidth/2, cHeight/4
        cv2.rectangle(T, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        x, y = x + cWidth + separation, y
        cv2.rectangle(T, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        x, y = x + cWidth + separation, y
        cv2.rectangle(T, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        x, y = x - cWidth - separation, y + cHeight + separation
        cv2.rectangle(T, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        self.tTemplate = T

        # Creating L Tetris template
        w, h = width-cWidth/2, height
        L = np.zeros((h,w,3), np.uint8)
        x, y = cWidth/2, cHeight/4
        cv2.rectangle(L, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        x, y = x + cWidth + separation, y
        cv2.rectangle(L, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        x, y = x + cWidth + separation, y
        cv2.rectangle(L, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        x, y = x - 2*cWidth - 2*separation, y + cHeight + separation
        cv2.rectangle(L, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        self.lTemplate = L

        # Creating J Jetris template
        w, h = width-cWidth/2, height
        J = np.zeros((h,w,3), np.uint8)
        x, y = cWidth/2, cHeight/4
        cv2.rectangle(J, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        x, y = x + cWidth + separation, y
        cv2.rectangle(J, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        x, y = x + cWidth + separation, y
        cv2.rectangle(J, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        x, y = x, y + cHeight + separation
        cv2.rectangle(J, (x,y), (x+cWidth, y+cHeight), (255, 255, 255), -1)
        self.jTemplate = J
