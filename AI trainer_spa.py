# Importing necessary Libraries
import threading
from tkinter import *
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk, ImageSequence
import time
from tkinter import filedialog
import cv2
import mediapipe as mp
import math
import numpy as np
import pygame

# Initializing pygame Module
pygame.mixer.init()

# Configuring the very First appearing Window
win = Tk()
win.geometry("1000x600")
win.title("AI Gym Trainer")


# Initiating Pose Solutions of Mediapipe
mp_pose = mp.solutions.pose


def detectPose(image, pose, display=True):

    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks need to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''

    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape

    # Check if any landmarks are detected.
    if results.pose_landmarks:

        landmarks = [(int(landmark.x * width), int(landmark.y * height), (landmark.z * width)) 
                    for landmark in results.pose_landmarks.landmark]
    
        # Return the output image and the found landmarks.
        return image, landmarks


def calculateAngle(landmark1, landmark2, landmark3):

    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''

    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

    # Check if the angle is less than zero.
    if angle < 0:

        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle

# Initial Value of Reps and flags for updating the data
reps = 0
flag_1 = 0
flag_2 = 0
bar_1 = 0
bar_2 = 0
right_shoulder_per_1 = 0
right_shoulder_per_2 = 0


def classifyPose(landmarks, output_image, display=False):

    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.
        reps: Number of correct repetitions.

    '''
    cv2.line(output_image, (landmarks[16][0], landmarks[16][1]), (landmarks[14][0], landmarks[14][1]), (255, 0, 0), 3)
    cv2.line(output_image, (landmarks[12][0], landmarks[12][1]), (landmarks[14][0], landmarks[14][1]), (255, 0, 0), 3)

    cv2.line(output_image, (landmarks[15][0], landmarks[15][1]), (landmarks[13][0], landmarks[13][1]), (0, 255, 0), 3)
    cv2.line(output_image, (landmarks[13][0], landmarks[13][1]), (landmarks[11][0], landmarks[11][1]), (0, 255, 0), 3)

    cv2.circle(output_image, (landmarks[12][0], landmarks[12][1]), 10, (255, 0, 0), cv2.FILLED)
    cv2.circle(output_image, (landmarks[12][0], landmarks[12][1]), 15, (0, 0, 255), 2)

    cv2.circle(output_image, (landmarks[14][0], landmarks[14][1]), 10, (255, 0, 0), cv2.FILLED)
    cv2.circle(output_image, (landmarks[14][0], landmarks[14][1]), 15, (0, 0, 255), 2)

    cv2.circle(output_image, (landmarks[16][0], landmarks[16][1]), 10, (255, 0, 0), cv2.FILLED)
    cv2.circle(output_image, (landmarks[16][0], landmarks[16][1]), 15, (0, 0, 255), 2)

    cv2.circle(output_image, (landmarks[11][0], landmarks[11][1]), 10, (0, 255, 0), cv2.FILLED)
    cv2.circle(output_image, (landmarks[11][0], landmarks[11][1]), 15, (0, 0, 255), 2)

    cv2.circle(output_image, (landmarks[13][0], landmarks[13][1]), 10, (0, 255, 0), cv2.FILLED)
    cv2.circle(output_image, (landmarks[13][0], landmarks[13][1]), 15, (0, 0, 255), 2)

    cv2.circle(output_image, (landmarks[15][0], landmarks[15][1]), 10, (0, 255, 0), cv2.FILLED)
    cv2.circle(output_image, (landmarks[15][0], landmarks[15][1]), 15, (0, 0, 255), 2)

    # Initialize the label of the pose. It is not known at this stage.
    global label
    global color
    label = 'WRONG'
    global reps
    global flag_1
    global flag_2
    color = "red"
    global bar_1
    global right_shoulder_per_1
    global bar_2
    global right_shoulder_per_2  

    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    # Get the angle between the right shoulder, elbow and wrist points. 
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])   
    
    # Get the angle between the left elbow, shoulder and hip points. 
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    # Get the angle between the right hip, shoulder and elbow points. 
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    
    #----------------------------------------------------------------------------------------------------------------

    if (80 < (right_shoulder_angle) < 195):

        bar_1 = np.interp((right_shoulder_angle), (105, 160), (0, 100))
        right_shoulder_per_1 = np.interp(right_shoulder_angle, (90, 160), (0, 100))

        bar_2 = np.interp((right_shoulder_angle), (105, 160), (0, 100))
        right_shoulder_per_2 = np.interp(right_shoulder_angle, (90, 160), (0, 100))

    elif(80 < (360-right_shoulder_angle) < 195):

        bar_1 = np.interp((360-right_shoulder_angle), (105, 160), (0, 100))
        right_shoulder_per_1 = np.interp((360-right_shoulder_angle), (90, 160), (0, 100))

        bar_2 = np.interp((right_shoulder_angle), (105, 160), (0, 100))
        right_shoulder_per_2 = np.interp(right_shoulder_angle, (90, 160), (0, 100))

    # Check if the both side arms are at 90 degrees.
    if (((80 < left_shoulder_angle < 110) and (80 < right_shoulder_angle < 110)) or
    ((80 < (360-left_shoulder_angle) < 110) and (80 < (360-right_shoulder_angle) < 110))):

        # Check if shoulders are at the required angle.
        if (((50 < left_elbow_angle < 110) and (50 < right_elbow_angle < 110)) or
        ((50 < (360-left_elbow_angle) < 110) and (50 < (360-right_elbow_angle) < 110))):

            label='CORRECT'
            flag_1 = 1

    elif (((110 < left_shoulder_angle < 195) and (110 < right_shoulder_angle < 195)) or
    ((110 < (360-left_shoulder_angle) < 195) and (110 < (360-right_shoulder_angle) < 195))):

        # Check if shoulders are at the required angle.
        if (((40 < left_elbow_angle < 195) and (40 < right_elbow_angle < 195)) or
        ((40 < (360-left_elbow_angle) < 195) and (40 < (360-right_elbow_angle) < 195))):

            label='CORRECT'
            flag_2 = 1

    if flag_1 == 1 and flag_2 == 1 and update_reps == 1:

        reps += 0.5
        flag_1 = 0
        flag_2 = 0

    #----------------------------------------------------------------------------------------------------------------

    # Check if the pose is classified successfully
    if label != 'WRONG':

        # Update the color (to green) with which the label will be written on the image.
        color = "green"

    # Return the output image and the classified label.
    return output_image, label


# Initiating the Pose Class
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)


def proceed():

    '''
    This function displays a Window with three Buttons, asking the user to select between Live Camera
    stream and Recorded Videos and About Us Button.

    '''

    win.destroy() # Destroying the previous window

    global show_result
    global burnt

    global reps 
    global hours 
    global minutes 
    global seconds
    global flag_3
    global good

    reps = 0
    hours = 0
    minutes = 0
    seconds = 0

    burnt = 0
    show_result = 0
    flag_3 = 0
    good = 0

    # Displaying the new window
    global root
    root = Tk()
    root.geometry("1000x600")
    root.title("Main Menu")

    im = PhotoImage(file="./asset/img/big-shoulders.png")

    can1 = Canvas(root, width=1000, height=600)
    can1.place(x=0, y=0)
    can1.create_image(0, 0, image=im, anchor="nw")

    # Opening in Threading Mode to allow parallel execution
    threading.Thread(target=gif).start() 

    # Live Button
    li_photo = PhotoImage(file="./asset/img/camera.png")

    Label(image=li_photo)

    li_btn = Button(root, image=li_photo, width=105, height=90, borderwidth = 0, relief="groove", command = live_bt)

    can1.create_window(30, 30, anchor="nw", window=li_btn)

    # Video Button 
    vid_photo = PhotoImage(file="./asset/img/vid.png")

    Label(image=vid_photo)

    vid_btn = Button(root, image=vid_photo, width=130, height=130, borderwidth = 0, relief="groove", command = videoBtn)
    can1.create_window(280, 15, anchor="nw", window=vid_btn)

    # Set Goal Button
    goal_photo = PhotoImage(file="./asset/img/set.png")

    Label(image=goal_photo)

    goal_btn = Button(root, image=goal_photo, width=135, height=135, borderwidth = 0, relief="groove", command = setGoal)
    can1.create_window(140, 140, anchor="nw", window=goal_btn)

    root.mainloop()

def setGoal():

    root.destroy() # Destroying previous window

    # Configuring new window
    global goal
    goal = Tk()
    goal.geometry("1000x600")
    goal.title("Set Your Goal")

    # Making global variables to access them in other functions as well

    global duration
    global name
    global weight
    global repitions
    global sets
    global show_result
    global sg

    # Widgets
    ima = PhotoImage(file="./asset/img/target.png") # Background Image

    can3 = Canvas(goal, width=1000, height=600) # Creating a Canvas for the window
    can3.place(x=0, y=0)
    can3.create_image(0, 0, image=ima, anchor="nw")

    # Taking Name of user
    name = Entry(goal, width=30, bg="white", fg="black", relief='groove')

    can3.create_text(80, 50, text="Name:", font=("Times", 20, "italic"), fill="white")

    can3.create_window(140, 45, anchor="nw", window=name)

    name.insert(0, "Enter_your_Name") # Initial Content

    name.bind("<Button-1>", clear_name) # Clearing the content on Single Click

    # Taking Duration of workout
    duration = Entry(goal, width=30, bg="white", fg="black", relief='groove')

    can3.create_text(80, 130, text="Duration:", font=("Times", 20, "italic"), fill="white")

    can3.create_window(150, 122, anchor="nw", window=duration)

    duration.insert(0,"Hours:Minutes:Seconds") # Initial Content

    duration.bind("<Button-1>", clear_time) # Clearing the content on Single Click

    # Taking Weight
    weight = Entry(goal, width=30, bg="white", fg="black", relief='groove')

    can3.create_text(80, 200, text="Weight:", font=("Times", 20, "italic"), fill="white")

    can3.create_window(150, 195, anchor="nw", window=weight)

    weight.insert(0,"Enter_Weight (in KGs)") # Initial Content

    weight.bind("<Button-1>", clear_weight) # Clearing the content on Single Click

    # Asking for reps
    repitions = Entry(goal, width=30, bg="white", fg="black", relief='groove')
    
    can3.create_text(80, 270, text="Reps:", font=("Times", 20, "italic"), fill="white")

    can3.create_window(150, 265, anchor="nw", window=repitions)

    repitions.insert(0,"Reps_per_Set") # Initial Content

    repitions.bind("<Button-1>", clear_reps) # Clearing the content on Single Click

    # Asking for Sets
    sets = Entry(goal, width=30, bg="white", fg="black", relief='groove')

    can3.create_text(80, 350, text="Sets:", font=("Times", 20, "italic"), fill="white")

    can3.create_window(150, 345, anchor="nw", window=sets)

    sets.insert(0,"Number_of_Sets") # Initial Content

    sets.bind("<Button-1>", clear_sets) # Clearing the content on Single Click

    # Back Button
    back_bt = PhotoImage(file="./asset/img/back.png")

    back_btn = Button(goal, image=back_bt, borderwidth = 0, width=110, height=50, relief="groove", command = goal_proceed)
    can3.create_window(30, 490, anchor="nw", window=back_btn)

    # Clear Button
    clr = Button(goal, text="CLEAR ALL", relief="groove", bg="black", fg="white", font = "Times 14 bold", borderwidth=2, command=clear)
    clr.place(x=50, y=400, width=130, height=60)

    # Done Button
    done_goal = Button(goal, text="DONE", relief="groove", bg="black", fg="white", font = "Times 14 bold", borderwidth=2, command=done)
    done_goal.place(x=250, y=400, width=90, height=60)

    show_result = 1
    sg = 1

    goal.mainloop()


def done():

    global temp_name
    global temp_weight
    global temp_repitions
    global temp_sets
    global duration
    global hours
    global minutes
    global seconds
    global count_repititions

    temp_name = str(name.get())
    temp_weight = int(weight.get())
    temp_repitions = int(repitions.get())
    temp_sets = int(sets.get())
    count_repititions = temp_repitions
    temp = duration.get()

    temp_2=temp.split(":")
  
    hours = int(temp_2[0])
    minutes = int(temp_2[1])
    seconds = int(temp_2[2])

    goal.destroy()

    global root
    root = Tk()
    root.geometry("1x1")

    live_bt()
                                                                                                 

def goal_proceed():

    goal.destroy() # Destroying previous window

    # Creating Temporary window
    global win
    win = Tk()
    win.geometry("1x1")

    # Calling the Proceed Function
    proceed()


def clear_name(e):

    if name.get()=='Enter_your_Name':

        name.delete(0,END)


def clear_time(e):

    if duration.get()=='Hours:Minutes:Seconds':

        duration.delete(0,END)


def clear_weight(e):

    if weight.get()=='Enter_Weight (in KGs)':

        weight.delete(0,END)


def clear_reps(e):

    if repitions.get()=='Reps_per_Set':

        repitions.delete(0,END)


def clear_sets(e):

    if sets.get()=='Number_of_Sets':

        sets.delete(0,END)


def clear():

    goal.destroy()

    global root
    root = Tk()
    root.geometry("1x1")

    setGoal()


# Video Button Window Function
def videoBtn():

    '''
    This function displays a Window on which Recorded videos will be played. User can switch between
    videos via the options given.
    
    '''

    root.destroy() # Destrying previous window

    # Configuring New Window
    global browse_btn
    global video
    global can2
    global bro

    video = Tk()
    video.geometry("1000x600")
    video.title("Choose Video")

    # Widgets
    ima = PhotoImage(file="./asset/img/browse_back.png")

    can2 = Canvas(video, width=1000, height=600)
    can2.place(x=0, y=0)
    can2.create_image(0, 0, image=ima, anchor="nw")

    browse_photo = PhotoImage(file="./asset/img/browse.png")

    Label(image=browse_photo)

    browse_btn = Button(video, image=browse_photo, width=120, height=110, borderwidth = 0, relief="groove", command = browse)
    can2.create_window(420, 450, anchor="nw", window=browse_btn)

    bro = Label(video, text = 'Browse Video', bg="white", fg = "black", font = "Times 20 bold")
    bro.place(x = 390, y = 545, width=180, height=60)

    back_btn = Button(video, text = 'Back', bg = "white", fg = "black", font = "Times 14 bold", relief = "groove", borderwidth = 2, command=back_video)
    back_btn.place(x = 50, y = 30, width=100, height=30)

    video.mainloop()

def browse():

    global vid
    global reps
    reps = 0

    bro.config(text="Change Video") 

    video.filename = filedialog.askopenfilename(initialdir="./asset/vid/",
    title="Select a Video", filetypes=(("mp4 files", "*.mp4"), ("all files", "*.*")))
    vid = video.filename
    if (vid != ''):
        play_video()


def play_video():

    global cap
    global w
    global h
    global label1

    frame_1 = Frame(video, width=400, height=400).place(x=300, y=30)

    cap = cv2.VideoCapture(vid)

    w = 400
    h = 400

    label1 = Label(frame_1, width=w, height=h)
    label1.place(x=300, y=30)

    select_img()

    back_btn = Button(video, text = 'Back', bg = "White", fg = "black", font = "Times 14 bold", relief = "groove", borderwidth = 2, command=back_browse)
    back_btn.place(x = 50, y = 30, width=100, height=30)


def select_img():

    _, frame = cap.read()

    frame = cv2.flip(frame, 1)

    frame, landmarks = detectPose(frame, pose_video, display=False)

    if landmarks:

        # Perform the Pose Classification.
        frame, _ = classifyPose(landmarks, frame, display=False)

    rep = Label(video, text = f'Reps:\n{int(reps):02}', bg="white", fg = "black", font = "Times 20 bold")
    rep.place(x = 100, y = 140, width=160, height=70)

    pos = Label(video, text = 'Posture:', bg="white", fg = "black", font = "Times 20 bold")
    pos.place(x = 100, y = 250, width=160, height=35)

    if label == 'CORRECT':
        lab = Label(video, text=f'{label}', bg='white', fg='green', font='Times 20 bold')
    else:
        lab = Label(video, text=f'{label}', bg='white', fg='red', font='Times 20 bold')

    lab.place(x = 100, y = 300, width=160, height=30)

    rect = Progressbar(video, orient=VERTICAL, length=200, mode="determinate")
    rect.place(x = 830, y = 80, width=50, height=450)
    rect['value'] = bar_1

    per = Label(video, text=f'{int(right_shoulder_per_1)}%', bg="white", fg = "black", font = "Times 20 bold")
    per.place(x = 820, y = 30, width=80, height=40) 

    frame = cv2.resize(frame, (w, h))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(rgb)
    finalImage = ImageTk.PhotoImage(image)
    label1.configure(image=finalImage)
    label1.image = finalImage

    video.after(2, select_img)


def live_stream():

    _, frame = cap.read()

    frame = cv2.flip(frame, 1)

    frame, landmarks = detectPose(frame, pose_video, display=False)

    if landmarks:

        # Perform the Pose Classification.
        frame, _ = classifyPose(landmarks, frame, display=False)

    pos = Label(live, text = 'Posture:', bg = "white", fg = "black", font = "Times 20 bold")
    pos.place(x = 800, y = 150, width=160, height=35) 

    if label == 'CORRECT':
        lab = Label(live, text=f'{label}', bg='white', fg='green', font='Times 20 bold')
    else:
        lab = Label(live, text=f'{label}', bg='white', fg='red', font='Times 20 bold')

    lab.place(x = 800, y = 200, width=160, height=30)  

    rect = Progressbar(live, orient=VERTICAL, length=200, mode="determinate")
    rect.place(x = 50, y = 140, width=50, height=350)
    rect['value'] = bar_2

    per = Label(live, text=f'{int(right_shoulder_per_2)}%',bg = "white", fg = "black", font = "Times 20 bold")
    per.place(x = 30, y = 120, width=80, height=40) 

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(rgb)
    finalImage = ImageTk.PhotoImage(image)
    label1.configure(image=finalImage)
    label1.image = finalImage

    live.after(2, live_stream)


def back_browse():

    global reps

    video.destroy()

    global root
    root = Tk()
    root.geometry("1x1")

    reps=0

    videoBtn()

se = 0

# Live Button Window Function
def live_bt():

    '''
    This function displays a Window on which Live Stream will be taken from Camera. User can select 
    any of the options given.
    
    '''
    root.destroy() # Destorying previous window

    # Configuring New Window
    global start_btn
    global live
    live = Tk()
    live.geometry("1000x600")
    live.title("Live Trainer")

    ima = PhotoImage(file="./asset/img/browse_back.png")

    can2 = Canvas(live, width=1000, height=600)
    can2.place(x=0, y=0)
    can2.create_image(0, 0, image=ima, anchor="nw")

    # Widgets

    global cap
    global w
    global h
    global label1
    global reps
    global se
    global hours
    global minutes
    global seconds
    
    

    frame_1 = Frame(live, width=600, height=400).place(x=170, y=90)

    cap = cv2.VideoCapture(0)

    w = 600
    h = 400

    label1 = Label(frame_1, width=w, height=h)
    label1.place(x=170, y=90)

    start_btn = Button(live, text = 'START', bg = "white", fg = "black", font = "Times 14 bold", relief = "groove", borderwidth = 5, command=start)
    start_btn.place(x = 400, y = 520, width=160, height=55) 

    back_btn = Button(live, text = 'BACK', bg = "white", fg = "black", font = "Times 14 bold", relief = "groove", borderwidth = 2, command=back)
    back_btn.place(x = 30, y = 20, width=100, height=30)

    music_btn_img = PhotoImage(file="./asset/img/music.png")
    music_btn = Button(live, image=music_btn_img, width=120, height=100, relief="groove", command=play_music)
    can2.create_window(840, 480, anchor="nw", window=music_btn)

    threading.Thread(target=live_stream).start()

    live.mainloop()


def gif():

    '''
    This function displays a Guide gif on the Window in order to guide the user how to do the exercise.
    
    '''

    # Widgets and Variables
    global img

    img = Image.open("./asset/img/Dumbbell_Shoulder_Press.gif")
    w8 = Label(root)
    w8.place(x=100, y=280, width=200, height=300)

    # Iterating over the gif images
    for img in ImageSequence.Iterator(img):

        img = img.resize((200,300))
        img = ImageTk.PhotoImage(img)
        w8.configure(image=img)
        root.update()
        time.sleep(0.3)

    # Calling trhe function again and again to show the gif continuously.
    root.after(0, gif)


def back():

    '''
    This function displays a Back Button which if pressed will get the user to the previous window.
    
    '''
    # Destroying current window and getting back to previous one.
    cap.release()
    global win
    live.destroy()
    win = Tk()
    win.geometry("1x1")
    pygame.mixer.music.stop()
    proceed()


def back_video():

    '''
    This function displays a Back Button which if pressed will get the user to the previous window.
    
    '''
    # Destroying current window and getting back to previous one.

    video.destroy()

    global win
    win = Tk()
    win.geometry("1x1")

    proceed()


# Initializing the value of Timer
hours = 0
minutes = 0
seconds = 0
flag_3 = 0
flag_4 = 0
first_time = 0
update_reps = 1
good = 0
sg = 0

def timer():

    global hours
    global minutes
    global seconds
    global hours_2
    global minutes_2
    global seconds_2
    global clock
    global flag_3
    global flag_4
    global first_time
    global clock
    global burnt
    global update_reps
    global se
    global temp_sets
    global reps
    global good
    global sg

    # Widgets

    if first_time == 0:

        seconds_2 = seconds
        hours_2 = hours
        minutes_2 = minutes


    w9 = Label(live, text = 'Time:', bg = "white", fg = "black", font = "Times 25 bold")
    w9.place(x = 790, y = 40, width=170, height=25)

    if flag_3 == 0:

        clock = Label(live, text='00:00:00', height=2, bg='white', fg='black', font='Times 20')

    if flag_3 == 1:

        clock = Label(live, text='00:00:00', height=2, bg='white', fg='black', font='Times 20')


    clock.place(x=790, y=80, width=170, height=25)        

    clock.config(text=f'{hours:02}:{minutes:02}:{seconds:02}')

    if flag_4==0:

        if flag_3 == 0:
            
            if seconds == 00:

                if minutes != 00:

                    minutes -= 1
                    seconds = 59


            if minutes == 00 and seconds == 00:

                if hours != 00:

                    hours -= 1
                    minutes = 59
                    seconds = 59
   

        if seconds == 00 and minutes == 00 and hours == 00:

            flag_3 = 1

        if flag_3 == 1:

            if seconds == 59: 

                minutes += 1
                seconds = 00

        if minutes == 59 and seconds == 59:

            hours += 1
            minutes = 00
            seconds = 00

        if flag_3 == 0:

            if ( seconds != 00 or minutes != 00 or hours != 00):

                seconds -= 1
                burnt = 0

        if flag_3 == 1:

            seconds+=1
            burnt = 1

        first_time+=1    

    if sg == 1:

        if reps == temp_sets:

            se  += 1
            reps = 0

        if se == temp_sets:

            good = 1
            results()

        shw_g = Label(live, text = f'Sets:  {se}/{temp_sets}', bg = "white", fg = "green", font = "Times 20 bold")
        shw_g.place(x = 200, y = 30, width=120, height=30)

    rep = Label(live, text = f'Reps:\n\n{int(reps):02}', bg = "white", fg = "black", font = "Times 20 bold")
    rep.place(x = 800, y = 300, width=160, height=100)

    live.after(1000, timer)


def start():

    '''
    This function displays a Timer on the window in order to note down the total time of workout.
    
    '''

    # Setting as Global variables to retain the value of timer
    global flag_3
    global reps
    global show_result
    global se
    global good
    
    timer()
    
    start_btn.configure(text='PAUSE', command=pause_resume)

    reset_btn = Button(live, text = 'RESET', bg = "white", fg = "black", font = "Times 14 bold", relief = "groove", borderwidth = 5, command=reset)
    reset_btn.place(x = 210, y = 520, width=160, height=55)


    if show_result == 1:

        results_btn = Button(live, text = 'SHOW\nRESULTS', bg = "black", fg = "green", font = "Times 14 bold", relief = "groove", borderwidth = 10, command=results)
        results_btn.place(x = 570, y = 510, width=180, height=70)



vari = 1

def pause_resume():

    '''
    This function displays a Stop Button which if pressed will stop the analysis.
    
    '''

    global vari
    global flag_4
    global root
    global update_reps

    vari+=1

    if vari % 2 == 0:

        flag_4=1

        start_btn.configure(text="Resume")

        update_reps = 0

    if vari % 2 != 0:

        flag_4=0

        start_btn.configure(text="Pause")

        update_reps = 1

    timer()


def reset():

    '''
    This function displays a Reset Button which if pressed will reset the Timer.
    
    '''

    # Setting global variables to initialize them.
    global root
    global hours
    global minutes
    global seconds
    global clock
    global reps

    # Giving the initial value of Timer and Reps
    hours=0
    minutes=0
    seconds=0
    reps = 0


burnt_calories = 0


def calories():

    global hours
    global hours_2
    global minutes
    global minutes_2
    global seconds
    global seconds_2
    global total_time
    global burnt_calories
    global burnt

    if burnt == 0:

        total_time = (((hours_2-hours)*60)+(minutes_2-minutes)+((seconds_2-seconds)/60))

    if burnt == 1:

        total_time = ((hours*60)+(hours_2*60)+(minutes)+(minutes_2)+(seconds/60)+(seconds_2/60))

    burnt_calories = (total_time*temp_weight*3.6)


def results():

    '''
    This function displays a Show Results Button which if pressed will display the summary of the exercise.
    
    '''

    # Destroying the current window and initializing the Timer again.
    live.destroy()
    pygame.mixer.music.stop()

    global hours
    global minutes
    global seconds
    global reps
    global good
    global burnt_calories
    global result
    global se
    global count_repititions

    global clock
    global result
    global stop_music

    calories()
    # Configuring the Results Window
    result = Tk()
    result.geometry("1000x600")
    result.title("Result")

    cap.release()

    # Widgets

    bg_img = PhotoImage(file="./asset/img/result.png")

    can4 = Canvas(result, width=1000, height=600)
    can4.place(x=0, y=0)
    can4.create_image(0, 0, image=bg_img, anchor="nw")


    if good == 1:

        achieved = Label(result, text = 'Target Achieved', bg = "white", fg = "black", font = "Times 14 bold italic underline")
        achieved.place(x = 340, y = 50, width=250, height=27)

        well_done = Label(result, text = f' Well Done Mr/Ms. {temp_name}', bg = "white", fg = "black", font = "Times 10 bold")
        well_done.place(x = 330, y = 150, width=300, height=30)

    elif good == 0:

        achieved = Label(result, text = 'Target not Achieved', bg = "white", fg = "black", font = "Times 15 bold italic underline")
        achieved.place(x = 340, y = 50, width=280, height=30)

        well_done = Label(result, text = f' Try Again Mr/Ms. {temp_name}\nYou can do it', bg = "white", fg = "black", font = "Times 14 bold")
        well_done.place(x = 330, y = 150, width=320, height=30)


    w10 = Label(result, text = 'Workout Duration:', bg = "white", fg = "black", font = "Times 15 bold italic underline")
    w10.place(x = 50, y = 250, width=260, height=35) 

    w11 = Label(result, text = f'{hours_2-hours:02}:{minutes_2-minutes:02}:{(seconds_2-seconds)-1:02}', bg = "black", fg = "red", font = "Times 10 bold")
    w11.place(x = 340, y = 260, width=130, height=25)

    w14 = Label(result, text = 'Total Reps:', bg = "white", fg = "black", font = "Times 15 bold italic underline")
    w14.place(x = 40, y = 320, width=230, height=35) 

    w15 = Label(result, text = f'{int(se*count_repititions+reps):02}', bg = "white", fg = "black", font = "Times 10 bold")
    w15.place(x = 300, y = 330, width=50, height=25)

    cal = Label(result, text = 'Calories Burnt:', bg = "white", fg = "black", font = "Times 15 bold italic underline")
    cal.place(x = 50, y = 420, width=230, height=35) 

    burn = Label(result, text = f'{burnt_calories:2} Kcal', bg = "white", fg = "black", font = "Times 10 bold")
    burn.place(x = 350, y = 430, width=140, height=25)

    back_btn = Button(result, text = 'Main Menu', bg = "white", fg = "black", font = "Times 14 bold", relief = "groove", borderwidth = 2, command=main_menu)
    back_btn.place(x = 20, y = 530, width=150, height=50)

    hours=0
    minutes=0
    seconds=0

    result.mainloop()


def main_menu():

    '''
    This function displays a Main Menu Button which if pressed will get the user to the Main Menu.
    
    '''

    # Destroying the current window and Displaying the Main Window
    result.destroy()

    global win
    win = Tk()
    win.geometry("1x1")

    proceed()


def play_music():

    global stop_music

    live.filename = filedialog.askopenfilename(initialdir="/Users",
    title="Select a Song", filetypes=(("wav files", "*.wav"), ("all files", "*.*")))
    beat = live.filename
    pygame.mixer.music.load(beat)
    pygame.mixer.music.play()


    stop_music = Button(live, text = 'STOP\nMUSIC', bg = "black", fg = "yellow", font = "Times 14 bold", relief = "groove", borderwidth = 7, command=stop_song)
    stop_music.place(x = 30, y = 520, width=130, height=70)


def stop_song():

    pygame.mixer.music.stop()
    stop_music.configure(text=" ", bg="black", fg="black", borderwidth=0, width=0, height=0)


# Widgets of the First Window

pre_im = PhotoImage(file="./asset/img/pic.png")

can = Canvas(win, width=1000, height=600)
can.place(x=0, y=0)

can.create_image(0, 0, image=pre_im, anchor="nw")

can.create_text(480, 30, text="WELCOME", font="Times 30 bold", fill="sky blue")

can.create_text(450, 90, text="AI GYM TRAINER", font="Times 22 bold", fill="sky blue")

can.create_text(110, 220, text="Developed By:", font="Times 20 underline", fill="sky blue")

can.create_text(130, 300, text="Albert 柳晟 \n 2023-CST ", font="Times 15 bold italic", fill="sky blue")

can.create_text(420, 220, text="Supervisor:", font="Times 21 underline", fill="sky blue")

can.create_text(410, 290, text="Ms. ZHONGYI XU ", font="Times 15 bold italic", fill="sky blue")

pro_bt = PhotoImage(file="./asset/img/im.png")

pro_bt_lab = Label(image=pro_bt)
proceed_btn = Button(win, image=pro_bt, borderwidth = 0, width=116, height=116, relief="raised", command = proceed)
can.create_window(420, 370, anchor="nw", window=proceed_btn)

from about import about
about_button = Button(win, text='About Me', bg='black', fg='white', font='Times 13 bold', borderwidth = 2, relief="raised", command = about)
about_button.place(x=20, y=540, width=100, height=50)

# Applying looping on main window
win.mainloop()