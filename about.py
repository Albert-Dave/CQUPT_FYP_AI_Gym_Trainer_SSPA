from tkinter import Tk, Label


def about():

    about = Tk()
    about.geometry('1000x330')
    about.title('About Me')

    about.configure(background='white')

    description = Label(about, fg='black', bg="white", font='arial 11', text='The prime objective and motivation for development of \"'
                                                                             'CV Based AI gym trainer or Exercise Analyst\" \n'
                                                                             'is a present to the community, a utility for easy, instant and user-friendly\n '
                                                                             'real-time exercise inspection, keeping in veiw the significance of workout for a healthy lifestyle in the hurry and scurry of life these days.'
                                                                             '\n There is a wide range of functions from live camera to recorded video analysis\n '
                                                                             'along with resoluting for a workout goal and calculation of calories burnt.\n\n '
                                                                             'Background music option and fascinating graphical interface empower the user towards accomplishing his target.\n'
                                                                             'Being a Computer Science and Technology major study, I strive to solve problems\n '
                                                                             'involving computer vision and AI, which is the one of the most promising sector and the future of Technology.\n\n'
                                                                             'I am Albert, studying  at Chongqing University of Posts and Telecommunication,Chongqing,China\n '
                                                                             'and have a profound and keen interest in Computer Vision and its applications and this very interest caused me\n '
                                                                             'to implement a computer vision approach as my graduation thesis project for deep learning based on camera pose estimation.\n '
                                                                             'Under the guidance my superviosr ZHONGYI XU, and a number of open-source python libraries encompassing OpenCV, Mediapipe, NumPy, Tkinter, Pillow etc.\n '
                                                                             'A long series of strivings and struggles are there in order to make the software efficient and methodical both in terms of pace and space.\n'
                                                                             'Future Horizons in this regard includes addition of some more exercises and stretches.')
    description.place(x=0, y=20)

    about.mainloop()