# question_generator:

#all these questions are counterfactual questions.

#question that requires collision with ego to happen
#[t0, ..... tx, ....tend]

#1. predict_collision
#Suggested now: tx-5
#Model observe: [t0,... , tx - 5], corresponding to 0.5 second before the collision happened
#Question: Is collision likely to happpen in the immediate future? If so, describe the object
#that we will collide with by specifying its type, color, and bounding box now.


#3. move around
#Suggested now: tend
#Model observe: tend
#Question: Suppose at tx we are at <dx,dy> instead of <x,y>. Will there be collision?
#Answer: [yes|no]. Describe the thing that will collide into at tend's perspective

#4. counterfactual trajectory
#Suggested now: tend
#Model observe: [t0, ....., tend]
#Question: Suppose the trajectory [t0, tend] is <x,y> * (tend-t0 + 1). Will we
#avoid the collision?
#

#don't need collision
#Suggested now: tend
#Model observe: [t0,....tend]
#Question: Suppose we stop all together at t_{i}. Will we run into collision?
#NOTE: i need to be fixed for all such question





