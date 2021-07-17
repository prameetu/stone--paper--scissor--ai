import cv2 as cv
from random import choice
import numpy as np

from tensorflow.keras.models import load_model

class_map  = { 0 : 'rock', 1 : 'paper' , 2 : 'scissor' , 3 : 'random'}

def mapper(x):
    return class_map[x]

def game_winner(mv1, mv2):
    if mv1 == mv2:
        return 'Tie'

    if mv1 == 'rock':

        if mv2 == 'scissor':
            return 'User'
        
        if mv2 == 'paper':
            return 'Computer'
    
    if mv1 == 'paper':

        if mv2 == 'scissor':
            return 'Computer'
        
        if mv2 == 'rock':
            return 'User'
    
    if mv1 == 'scissor':

        if mv2 == 'paper':
            return 'User'
        
        if mv2 == 'rock':
            return 'Computer'

model = load_model('rock-paper-scissors-model.h5')

cam = cv.VideoCapture(0)
cam.set(3,1250)
cam.set(4,720)

prev_move = None

while True:

    ret,frame = cam.read()

    if not ret:
        continue


    #rectangle for user
    cv.rectangle(frame,(25, 100), (325, 400), (0,255,0), 2)
    #rectangle for computer
    cv.rectangle(frame,(925, 100), (1225, 400), (0,255,0), 2)

    user = frame[100:400,25:325]
    user_img = cv.cvtColor(user, cv.COLOR_BGR2RGB)
    user_img = cv.resize(user, (227,227))

    pred = model.predict(np.array([user_img]))
    user_move = np.argmax(pred[0])
    user_move = mapper(user_move)

    if prev_move != user_move:
        if user_move != 'random':
            comp_move = choice(['rock','paper','scissor'])
            winner = game_winner(user_move,comp_move)
        else:
            comp_move = "none"
            winner = 'WAITING FOR USER'

    prev_move = user_move

    font = cv.FONT_HERSHEY_TRIPLEX

    cv.putText(frame, f'Your Move: {user_move}',(40,450),font,0.8,(0,0,0),2,cv.LINE_AA)
    cv.putText(frame, f"Computer's Move: {comp_move}",(915,450),font,0.8,(0,0,0),2,cv.LINE_AA)


    cv.putText(frame, f"WINNER : {winner}",(470,550),font,1.1,(224,224,224),2,cv.LINE_AA)


    if comp_move != 'none':
        img = cv.imread(f'images/{comp_move}.jpg')
        img = cv.resize(img, (300,300))
        frame[100:400 , 925:1225] = img

    cv.imshow('Rock--Paper--Scissor--AI',frame)

    k = cv.waitKey(10)
    if k == ord('q'):
        break

cam.release()
cv.destroyAllWindows()




        

