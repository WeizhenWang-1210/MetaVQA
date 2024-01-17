def follow(obj1, obj2)->bool:
    #obj1 is followed by obj2 if:
        #|obj1-obj2| < some value, for the entirety of the observation episode
        #obj1.heading dot (obj2-obj1) is negative, for the entirety of the observation episode
        #obj1 can shoot a ray to obj2, and obj2 can shoot a ray to obj1, for he entirety of the observation episode
                #|-> requrire further annotation: for each vehicle, record the lidar-observable vehicles.    
        #obj2 must be almost exactly behind obj1 
    pass

def pass_by(obj1, obj2):
    #obj2 pass_by obj1 if:
        #obj2-obj1 dot obj1.heading experience exactly one sign reversal for he entirety of the observation episode
    pass

def collide_with(obj1,obj2):
    #obj1 collided with obj2 if one collision record exists between obj1 and obj2, for the entirety of the observation
    #episode
    pass

def head_toward(obj1,obj2):
    #obj2 head toward obj1 if d2-d1 dot d2.heading is negative
    pass

def drive_alongside(obj1,obj2):
    #obj2 drive alongside obj1 if obj2 remains about directly to the right(left) of obj1 for the entirety of the 
    #observation episode.
    pass



#TODO: Episodic Reocrding
#TODO: Record Collision Event.  In scenario_generation.py