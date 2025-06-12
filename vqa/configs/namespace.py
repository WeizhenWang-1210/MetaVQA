NAMESPACE = dict(
    # type field is not accessed Anywhere Anymore, depcreated.
    type = [
        "SUV", "sedan", "truck", "sportscar","jeep","pickup","Compact Sedan",
        "Traffic Cone", "Warning Sign", "Planar Barrier", "Vehicle", "Pedestrian",
        "Traffic Participant","Traffic Obstacle"
    ],
    color = [
        "White","Black","Grey","Red","Blue","Orange","Yellow","Green"
    ],
    pos = [
        "l","f","r","b","lf","rf","lb","rb"
    ],

)

POSITION2CHOICE = dict(
    lf="left-front", rf="right-front", lb="left-back", rb="right-back", f="front", b="back", l="left", r="right", m="next-to"
)


MAX_DETECT_DISTANCE = 75
MIN_OBSERVABLE_PIXEL = 1200
OBS_WIDTH = 1920
OBS_HEIGHT = 1080




METADRIVE_TYPES = [
    "Hatchback", "Sedan", "Pickup", "Truck", "Barrier", "Warning", "Cone", "Bike", "Pedestrian", "TrafficLight"
]

METADRIVE_COLORS = [
    "Blue", "White", "Gray", "Red", "Orange", "Traffic Light"
]