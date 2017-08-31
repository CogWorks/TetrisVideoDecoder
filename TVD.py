'''
    File name: TVD.py
    Copyright: Copyright 2017, The CogWorks Laboratory
    Author: Hassan Alshehri
    Email: alsheh@rpi.edu or eng.shehriusa@gmail.com
    Data created: May 22, 2017
    Date last modified: August 26, 2017
    Description: TETRIS VIDEO DECODER (TVD)
    Status: Research
    Requirements/Dependencies:
        1. Python 2.7,
        2. OpenCV 3 (may not work with OpenCV 2),
        3. NumPy,
        4. tqdm
'''
import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import numpy as np
import math
import os
import argparse
import util
import nextZoid
from CharacterRecognition import ocr
from tqdm import *

'''
Ouptput data file format: (Tab Separated Values)
<Frame_rate_per_seond>
<total_frames>
<episode_number><tab><next_zoid_letter><tab>[<numberical_data>...]<tab><board_rep><tab><color_data>'\n'
'''

INFINITY = float('inf')
flag = True
empty = np.zeros((200,200,3), np.uint8)



def videoReader(video, grid, game, v, nxtZoid):
    '''
    Read video frame by frame and call the frameAnalysis function on each frame.
    '''
    frameNumber = game.START_FRAME
    out_file = None

    # If writing mode is specified
    if game.WRITING_MODE is not None:
        out_file = getOutputDataFile(game)

    # Iterate over the frames; trange() is used for displaying progress bar in terminal.
    for f in trange(game.TOTAL_FRAMES, ncols=90):
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break

        # Apply image processing algorithm on frame
        frameAnalysis(frame, grid, game, v, frameNumber, nxtZoid, out_file)

        # If this last frmae, stop reading frames from the video.
        if game.END_FRAME == frameNumber:
            break

        frameNumber += 1

    # When everything is done, release the video and close all windows
    video.release()
    v.destroyAllWindows()

    # Close output data file
    if game.WRITING_MODE is not None:
        out_file.close()


def frameAnalysis(frame, grid, game, v, frameNumber, nxtZoid, out_file):
    '''
    Extract data from frame. Store and/or visualize data if specified by user.
    '''
    # Crop the screen from the frame
    screen = getScreen(frame, game, grid)

    # Remove background and noise from frame
    colorImg, blackWhiteImg = removeNoise(screen, game)

    # Detect whether there is a square inside each cell.
    isScreenFlashing = findSquares(blackWhiteImg, game, grid)
    # Flashing effect occurs when four lines are being cleared.
    # If the screen has a flashing effect, skip this frame.
    if (isScreenFlashing):
        return

    # Check if frame has a new episode.
    if isNewEpisode(game):
        game.episode += 1

    # Crop the next zoid box from frame
    box =  getNextZoidBox(frame, nxtZoid, grid)

    # Determine the letter of the Tetris piece displayed in the next zoid box
    nextZoidNLetter = nxtZoid.findNextZoid(box)

    # Get color data if the user specifies to write it to a files or visualize it.
    recreatedBoard = None
    if game.WRITE_COLOR_DATA or game.RECREATE_GAME:
        recreatedBoard = detectColor(screen, grid, game)

    # Read numerical data from frame (i.e. score, lelve, etc.)
    digitData = ''
    for npr in game.digitData:
        npr.readNumber(frame)
        digitData += npr.data + '\t'

    # Detect errors such as falsely detected squares or
    # failure to detect squares in the screen.
    #errorDetection(screen, gameWindow, gamePlayWindow, game, f, fps, nextZoidNLetter)

    # If writing/appending is specified, write data to file.
    if game.WRITING_MODE is not None:
        line  = str(game.episode) + '\t'
        line += nextZoidNLetter + '\t'
        line += digitData
        line += str(game.BOARD)
        if game.WRITE_COLOR_DATA:
            line += '\t' + str(game.COLOR_DATA)
        out_file.write(line+'\n')

    # Visualize data according to the specified optional flags.
    v.visualize(frame, screen, colorImg, blackWhiteImg, game.BOARD, frameNumber, nxtZoid, recreatedBoard)

def Preprocessing(game):
    '''
    In the preprocessing step, open video for reading, find frame rate, total frames and
    allow user to determine the following:
      1. The start and end frames of the match.
      2. Location of screen (i.e. board).
      3. Location of next zoid box.
      4. Location of all numerical data to be parsed (optional).
    '''
    # Read video
    video = cv2.VideoCapture(game.IN_VIDEO_FILE)

    # Exit if video not opened.
    if not video.isOpened():
        print "ERROR: Could not open video"
        exit(1)

    # Get frame rate
    game.FPS = round(video.get(cv2.CAP_PROP_FPS))

    # Count the total number of frames; don't panic, this is O(1) operation :).
    game.TOTAL_FRAMES = total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print 'ERROR: Cannot read video file'
        exit(1)

    # Determine the begginning and end of the match (i.e. start and end frames)
    from playback import Playback
    p = Playback(video, None, game, None)
    game.START_FRAME = p.start
    game.END_FRAME = p.end

    # Rewind video to the selected start frame
    video.set(cv2.CAP_PROP_POS_FRAMES, p.start)
    frame = video.read()
    ok, frame = video.read()
    if not ok:
        print 'ERROR: Cannot read video file'
        exit(1)

    # Update total number of frames
    game.TOTAL_FRAMES = total = game.END_FRAME - game.START_FRAME

    # Find the screen manually with the help of the user.
    grid, x, y, w, h = manualCrop(frame, game)

    # Find location of next zoid box
    c = util.Crop()
    roi, x, y, w, h = c.crop(frame, taskName = 'next zoid')
    nxtZoid = nextZoid.NextZoid((x, y, w, h), game, grid)
    cv2.imshow('Next Zoid', roi)
    answer = util.MSG_Box(question=True, title='Next Zoid', msg="Do you want to continue?")
    cv2.destroyWindow('Next Zoid')
    cv2.waitKey(1)
    if answer == 'no':
        exit(1)

    # Find the locations of each of the numerical data
    ocrObj = ocr.OCR()
    for name in game.DIGIT_RECOGNITION:
        game.digitData.append(util.NumberParser(name, frame, ocrObj, game))

    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return video, grid, game, nxtZoid

def getOutputDataFile(game):
    '''
    Open a file in writing or appending mode. Add headers to file if in writing mode.

    Return:
        out_file: file object
    '''
    filename = game.OUT_DATA_FILE.split('.')[0]
    mode = game.WRITING_MODE
    out_file = open(filename+'.tab', mode)

    # If writing mode is specified (i.e. not append), add the headers to the file.
    if mode == 'w':
        columnNames = "episode_number\tnext_zoid\t"
        for d in game.digitData:
            columnNames += d.name + '\t'
        columnNames+= 'board_representation'
        if game.WRITE_COLOR_DATA:
            columnNames += "\tcolor_data"
        out_file.write(str(game.FPS)+'\n')
        out_file.write(str(game.TOTAL_FRAMES)+'\n')
        out_file.write(columnNames+'\n')

    return out_file


def isNewEpisode(game):
    '''
    To check for a new episdoe, the current frame must have four more squares
    in first two rows than the previous frame.

    Return:
      True if new episode is detected and False otherwise.
    '''
    bd = game.BOARD
    row1 = sum(bd[0])
    row2 = sum(bd[1])
    if row1+row2-4 == game.previousFrameRowSum:
        game.previousFrameRowSum = row1+row2
        return True
    else:
        game.previousFrameRowSum = row1+row2
        return False

def manualCrop(frame, game):
    '''
    Allow the user to manuualy crop the player's screen and
    build a grid on top of it and store pixle coordinates.
    '''
    c = util.Crop()
    roi, x, y, w, h = c.crop(frame, taskName='screen')
    gb = util.GridBuild(roi, windowName='Grid')
    x1, y1, w, h = gb.createGrid(x, y)
    grid = util.Grid(gb)
    img = frame[y1:y1+h, x1:x1+w]

    # Show cropped image
    cv2.imshow('Grid', img)

    answer = util.MSG_Box(question=True, title='Screen', msg="Do you want to continue?")

    cv2.destroyWindow('Grid')
    cv2.waitKey(1)

    if answer == 'no':
        exit(1)

    game.PLAYER_SCREEN_DIMENSIONS = x1, y1, w, h
    return grid, x1, y1, w, h

def getScreen(image, game, grid):
    '''
    Crop screen from frame and return the screen.
    '''
    x,y,w,h = game.PLAYER_SCREEN_DIMENSIONS
    screen = image[y:y+h+2, x:x+w+2]
    w, h = screen.shape[1], screen.shape[0]
    w, h  = int(w+w*grid.zoomLevel), int(h+h*grid.zoomLevel)
    screen = cv2.resize(screen, (w, h))
    return screen

def getNextZoidBox(image, nxtZoid, grid):
    '''
    Crop the next zoid box and return the box.
    '''
    x, y, w, h = nxtZoid.nextZoidLoc
    box = image[y:y+h, x:x+w]
    w, h = box.shape[1], box.shape[0]
    w, h  = int(w+w*grid.zoomLevel), int(h+h*grid.zoomLevel)
    box = cv2.resize(box, (w, h))
    return box

def detectColor(img, grid, game):
    '''
    Compute the average color of each cell in grid (i.e. board).
    '''
    # Make an empty image to store the new colors.
    t = img.shape[0], img.shape[1], 3
    res = np.zeros(t ,dtype=np.uint8)

    # Get the size of the board and cells.
    x, y, w, h = grid.getGridDimensions()
    cWidth, cHeight = grid.cellWidth, grid.cellHeight
    verticalSteps = grid.numOfCellsVertically
    horizontalSteps = grid.numOfCellsHorizontally

    # Get the current board rep.
    board = game.BOARD

    # Make the boundary of the screen colorful
    cv2.rectangle(res, (0,0), (w, h), (0, 255, 255), 2)

    # Iterate over all the cells in the grid/board.
    for j in range(verticalSteps):
        jPix = 1 +  j * cHeight # upper left corner j pixle coordinate
        for i in range(horizontalSteps):
            iPix = 1 + i * cWidth # upper left corner i pixle coordinate
            square = img[jPix:jPix+cHeight, iPix:iPix+cWidth]
            x, y = iPix, jPix
            w, h = cWidth, cHeight

            r = cWidth/3
            tmpImg = square[r:r*2, r:r*2] # this will fail when we have rectangular shape peices
            tmpImg = cv2.erode(tmpImg, None, iterations=1)
            tmpImg = cv2.dilate(tmpImg, None, iterations=1)

            # Compute the average color of the cell
            mean = cv2.mean(tmpImg)[:3]

            # Store average color in RGB format.
            game.COLOR_DATA[j][i] = int(mean[2]), int(mean[1]), int(mean[0])
            x, y, w, h = x+1, y+1, w-2, h-2
            s = 0.7
            m = int(mean[0]*s) , int(mean[1]*s) , int(mean[2]*s)

            # Paint the cell in the new image with average color.
            cv2.rectangle(res,(x,y),(x+w,y+h), m,-1)
            x, y, w, h = x+3, y+3 ,w-6, h-6
            cv2.rectangle(res,(x,y),(x+w,y+h),mean,-1)
    return res

def findSquares(blackWhite, game, grid):
    '''
    findSquares iterates over all the cells in the blackWhite image.
    It assigns a binary value to each cell.'1' if there are sufficient
    white pixles in the cell and '0' otherwise.

    Args:
        blackWhite: the image to be analyzed.
        game: it stores meta data about the current game.

    Retrun:
        True if flashing effect occurs. False otherwise.
    '''
    # If blackWhite is not a grayscale image, fail.
    if len(blackWhite.shape) > 2:
        print "ERROR: expected a black and white image but received a color image"
        exit(1)

    # Get info about the size of the board and cell
    x, y, w, h = grid.getGridDimensions()
    cWidth, cHeight = grid.cellWidth, grid.cellHeight
    verticalSteps = grid.numOfCellsVertically
    horizontalSteps = grid.numOfCellsHorizontally

    # Initialize a temprary boolean list
    tmpBOARD = [ [0] * game.WIDTH for i in range(game.HEIGHT)]

    # Iterate over all cells in the board
    for j in range(verticalSteps):
        jPix = y +  j * cHeight # upper left corner j pixle coordinate
        for i in range(horizontalSteps):
            iPix = x + i * cWidth # upper left corner i pixle coordinate
            # Crop the the the cell from the blackWhite image for analysis
            square = blackWhite[jPix:jPix+cHeight, iPix:iPix+cWidth]
            numOfWhitePixels = cv2.countNonZero(square)
            totalPixels = square.size

            minPercentage = game.WHITE_PIXLES_PERCENTAGE/100.0

            # If the number of white pixles is >= %50 of total pixles in the cell
            if numOfWhitePixels/float(totalPixels) >= minPercentage:
                tmpBOARD[j][i] = 1
            else:
                tmpBOARD[j][i] = 0

    # Detect flashing effect that occurs when a player clears 4 lines
    previousEpisode = 0 # number of squares in the previous episode
    currentEpisode = 0 # number of squares in the current episode
    for j in range(verticalSteps):
        previousEpisode += sum(game.BOARD[j])
        currentEpisode += sum(tmpBOARD[j])
    # primary case: if more than 4 squares have been detected in the current episode than
    # the previous one, then we have false positive detections due to flasing effect.
    primaryCase = previousEpisode != (game.HEIGHT * game.WIDTH)-1
    primaryCase = primaryCase and currentEpisode == (game.HEIGHT * game.WIDTH)
    # edge case: 4 lines are being cleared when board is completely full.
    edgeCase = previousEpisode == (game.HEIGHT * game.WIDTH)
    if primaryCase or edgeCase:
        return True # return True if flashing effect occured

    game.BOARD = tmpBOARD

    # Return False indicating no flashing effect occured
    return False

def errorDetection(screen, gameWindow, gamePlayWindow, game, f, fps, nextZoidNLetter):
    '''
    When a false detection occurs, freeze the program and
    allow the user to manually fix any false detections on the current frame.
    '''

    '''
    board_rep = game.BOARD
    colorData = game.COLOR_DATA
    nexZoidLetter = nextZoidNLetter
    frameNumbeer = f
    frameRate = fps
    score =
    '''

    global flag
    totalPieces = 0
    for row in game.BOARD:
        totalPieces += sum(row)

    if totalPieces == previousTotalPieces+4:
        previousTotalpieces = totalPieces
    elif totalPieces == previousTotalPieces-2:
        previousTotalpieces = totalPieces
    elif totalPieces != previousTotalPieces and flag:
        flag = False
        img = screen.copy()
        grid = drawGrid(img)

        print "==> Please use the mouse and click on the falsely detected or undetected peices"
        print "    (Press the 'c' key to continue) "

        while True:
            if gameWindow.hasChanged:
                # ...
                gameWindow.hasChanged = False
                grid2 = gameWindow.drawGrid(screen)
                gameWindow.createScreenWindow(grid2,f, fps)
                key1 = gameWindow.showImage(game.DELAY)
                # ...
                board = gamePlay(screen)
                gamePlayWindow.createScreenWindow(board,f, fps)
                key2 = gamePlayWindow.showImage(game.DELAY)
            else:
                key1 = gameWindow.showImage(game.DELAY)
                key2 = gamePlayWindow.showImage(game.DELAY)

            if key1 == ord('c') or key2 == ord('c'):
                break


def removeNoise(img, game):
    '''
    Remove background color and noise img

    Return:
       colorImg: the modified color image
       maskIn: black and white image where
               backgroudn is black and
               everything else is white
    '''
    # Convert BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range of background color in HSV
    lower = np.array(game.SCREEN_LOW)
    upper = np.array(game.SCREEN_HIGH)

    # Create a mask of the background.
    # This makes the background white and everything else black.
    mask = cv2.inRange(hsv, lower, upper)

    # Invert the mask of the background to make
    # the backgorund black and everything else white.
    maskInv = cv2.bitwise_not(mask)

    # Remove the background from the color image
    colorImg = cv2.bitwise_and(img, img, mask=maskInv)

    return colorImg, maskInv

def commandLineArgsParsing(game):
    '''
    Parse command line arguments and make error checking.
    '''
    usageMsg  ="[OPTIONS] in-video-file"
    formatter_class = lambda prog: argparse.HelpFormatter(prog, width=100, max_help_position=100)
    parser = argparse.ArgumentParser(description='Tetris Video Decoder', usage='%(prog)s '+ usageMsg, formatter_class=formatter_class)

    # Positional Arguments (required arguments)
    parser.add_argument('in_video_file',
                        metavar='in-video-file',
                        type=argparse.FileType('r'),
                        help='path to input video file.')

    # Optional Arguments
    parser.add_argument('-w', '--write-mode',
                        metavar='PATH',
                        type=str,
                        help='path to output data file; truncate file if it exists otherwise create a new one.')

    parser.add_argument('-ap', '--append-mode',
                        metavar='PATH',
                        type=str,
                        help='path to output data file; append data to file if it exists otherwise create a new one.')

    parser.add_argument('-wpp', '--white-pixles-percentage',
                        metavar='P',
                        type=float,
                        default=50,
                        help='the minimum percentage of white pixles that must exist in a cell to determine there is a square in that cell.(default: %(default)s).')

    parser.add_argument('-e', '--episode',
                        metavar='E',
                        type=int,
                        default=0,
                        help='set the first episode number. (default: %(default)s).')

    parser.add_argument('-tm', '--test-mode',
                        action='store_true',
                        help='run parser in test mode and no data will be stored (default: %(default)s).')

    parser.add_argument('-dw', '--detection-window',
                        action='store_false',
                        help='close the window that shows detected squares for each frame (default: %(default)s).')

    parser.add_argument('-sv', '--show-video',
                        action='store_true',
                        help='display a window showing the video being parsed (default: %(default)s).')

    parser.add_argument('-cd', '--color-data',
                        action='store_true',
                        help='include color data in the outptut data file (default: %(default)s).')

    parser.add_argument('-rc', '--recreate-game',
                        action='store_true',
                        help='display a window showing the average color of the detected squares (default: %(default)s).')

    parser.add_argument('-dr', '--digit-recognition',
                        metavar='N',
                        type=str,
                        nargs='+',
                        help='variable name(s) for reading the numerical data from the video')

    parser.add_argument('-scu', '--screen-upper',
                        metavar=('H', 'S', 'B'),
                        type=int,
                        nargs=3,
                        default=game.SCREEN_HIGH,
                        help='screen backgorund color upper bound (default: %(default)s).')

    parser.add_argument('-scl', '--screen-lower',
                        metavar=('H', 'S', 'B'),
                        type=int,
                        nargs=3,
                        default=game.SCREEN_LOW,
                        help='screen backgorund color lower bound (default: %(default)s).')

    parser.add_argument('-du', '--digit-upper',
                        metavar=('H', 'S', 'B'),
                        type=int,
                        nargs=3,
                        default=game.DIGIT_HIGH,
                        help='digit backgorund color upper bound (default: %(default)s).')

    parser.add_argument('-dl', '--digit-lower',
                        metavar=('H', 'S', 'B'),
                        type=int,
                        nargs=3,
                        default=game.DIGIT_LOW,
                        help='digit backgorund color lower bound (default: %(default)s).')

    parser.add_argument('-d', '--delay',
                        metavar='D',
                        type=int,
                        default=1,
                        help='add delay in ms when displaying the detection window (default: %(default)s).')

    parser.add_argument('-bd', '--board-size',
                        metavar=('W', 'H'),
                        type=int,
                        nargs=2,
                        default=[10,20],
                        help='specify Tetris board size (default: %(default)s).')

    parser.add_argument('-v', '--version',
                        action='version',
                        version='%(prog)s 2.0')

    # Start parsing command line arguments
    args = parser.parse_args()

    # Do more error checking on some of the arguments
    if args.delay < 0:
        msg = "negative vlaue for -d/--delay is not allowed"
        parser.error(msg)

    if args.board_size[0] <= 0 or args.board_size[1] <= 0:
        msg = "non-positive vlaues for -bd/--board-size are not allowed"
        parser.error(msg)

    if 0 > args.white_pixles_percentage or args.white_pixles_percentage > 100:
        msg = "cannot select a vlaue for -wpp/--white-pixles-percentage that is not in [0 - 100]"
        parser.error(msg)

    if args.write_mode is not None and args.append_mode is not None:
        msg = "optional arguments -w/--writemode and -ap/--append-mode: cannot use both flags at the same time."
        parser.error(msg)
    elif args.write_mode is not None:
        game.WRITING_MODE = 'w'
        game.OUT_DATA_FILE = args.write_mode
    elif args.append_mode is not None:
        game.WRITING_MODE = 'a'
        game.OUT_DATA_FILE = args.append_mode

    if args.color_data is not None:
        game.WRITE_COLOR_DATA = args.color_data
        WIDTH = args.board_size[0]
        HEIGHT = args.board_size[1]
        game.COLOR_DATA = [ [2] * WIDTH for i in range(HEIGHT) ]

    if args.screen_upper is not None:
        upper = args.screen_upper
        isError = upper[0] < 0 or upper[0] > 179
        isError = isError or upper[1] < 0 or upper[1] > 255
        isError = isError or upper[2] < 0 or upper[2] > 255
        if isError:
            msg = "optional argument -scu/--screen-upper: one or more numeric value is out of range"
            parser.error(msg)
        else:
            game.SCREEN_HIGH = upper

    if args.screen_lower is not None:
        lower = args.screen_lower
        isError = lower[0] < 0 or lower[0] > 179
        isError = isError or lower[1] < 0 or lower[1] > 255
        isError = isError or lower[2] < 0 or lower[2] > 255
        if isError:
            msg = "optional argument -scu/--screen-lower: one or more numeric value is out of range"
            parser.error(msg)
        else:
            game.SCREEN_LOW = lower

    if args.digit_upper is not None:
        upper = args.digit_upper
        isError = upper[0] < 0 or upper[0] > 179
        isError = isError or upper[1] < 0 or upper[1] > 255
        isError = isError or upper[2] < 0 or upper[2] > 255
        if isError:
            msg = "optional argument -du/--digit-upper: one or more numeric value is out of range"
            parser.error(msg)
        else:
            game.DIGIT_HIGH = upper

    if args.digit_lower is not None:
        lower = args.digit_lower
        isError = lower[0] < 0 or lower[0] > 179
        isError = isError or lower[1] < 0 or lower[1] > 255
        isError = isError or lower[2] < 0 or lower[2] > 255
        if isError:
            msg = "optional argument -du/--digit-lower: one or more numeric value is out of range"
            parser.error(msg)
        else:
            game.DIGIT_LOW = lower

    if args.digit_recognition is not None:
        game.DIGIT_RECOGNITION = args.digit_recognition

    # Variable assignments
    game.IN_VIDEO_FILE = args.in_video_file.name
    game.WHITE_PIXLES_PERCENTAGE = args.white_pixles_percentage
    game.DETECTION_WINDOW = args.detection_window
    game.RECREATE_GAME = args.recreate_game
    game.SHOW_VIDEO = args.show_video
    game.DELAY = args.delay
    game.WIDTH = args.board_size[0]
    game.HEIGHT = args.board_size[1]
    game.TEST_MODE = args.test_mode

    if args.episode:
        game.episode = args.episdoe

    if game.TEST_MODE:
        game.WRITING_MODE = None
        game.SHOW_VIDEO = True
        game.DETECTION_WINDOW = True

if __name__  == '__main__':
    # Create a game object to store meta data about the current game
    game = util.GameInfo()

    # Parse command line arguments
    commandLineArgsParsing(game)

    # Do a preprocessing step before running the parser.
    video, grid, game, nxtZoid = Preprocessing(game)

    # Set up game visualization
    v = util.Visualization(game, grid)

    # Run in test mode if test mode is specified.
    if game.TEST_MODE:
        from playback import Playback
        Playback(video, grid, game, v, nxtZoid, testMode=True)
    else:
        # Rund the parser and start reading frames from video.
        videoReader(video, grid, game, v, nxtZoid)
