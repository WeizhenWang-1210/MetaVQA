NAMED_MAPPING = dict(
    nil=dict(singular="traffic object", plural="traffic objects"),
    Bus=dict(singular="bus", plural="buses"),
    Caravan=dict(singular="caravan", plural="caravans"),
    Coupe=dict(singular="coupe", plural="coupes"),
    FireTruck=dict(singular="fire engine", plural="fire engines"),
    Jeep=dict(singular="jeep", plural="jeeps"),
    Pickup=dict(singular="pickup", plural="pickups"),
    Policecar=dict(singular="police car", plural="police cars"),
    SUV=dict(singular="SUV", plural="SUVs"),
    SchoolBus=dict(singular="school bus", plural="school buses"),
    Sedan=dict(singular="sedan", plural="sedans"),
    SportCar=dict(singular="sports car", plural="sports cars"),
    Truck=dict(singular="truck", plural="trucks"),
    Hatchback=dict(singular="hatchback", plural="hatchbacks"),
    Pedestrian=dict(singular="pedestrian", plural="pedestrians"),
    vehicle=dict(singular="vehicle", plural="vehicles"),
    Bike=dict(singular="bike", plural="bikes"),
    Barrier=dict(singular="traffic barrier", plural="traffic barriers"),
    Warning=dict(singular="warning sign", plural="warning signs"),
    Cone=dict(singular="traffic cone", plural="traffic cones"),
    # nusc additions
    Wheelchair=dict(singular="wheel chair", plural="wheel chairs"),
    Police_officer=dict(singular="police officer", plural="police officers"),
    Construction_worker=dict(singular="construction worker", plural="construction workers"),
    Animal=dict(singular="animal", plural="animals"),
    Car=dict(singular="car", plural="cars"),
    Motorcycle=dict(singular="motorcycle", plural="motorcycles"),
    Construction_vehicle=dict(singular="construction vehicle", plural="construction vehicles"),
    Ambulance=dict(singular="ambulance", plural="ambulances"),
    Trailer=dict(singular="trailer", plural="trailers"),
    Stroller=dict(singular="stroller", plural="strollers")
)
FONT_SCALE = 1
BACKGROUND = (0, 0, 0)
USEBOX = True
TYPES_WITHOUT_HEADINGS = ["Cone", "Barrier", "Warning", "TrafficLight", "Trailer"]
DIRECTION_MAPPING = {
    "l": "to the left of",
    "r": "to the right of",
    "f": "directly in front of",
    "b": "directly behind",
    "lf": "to the left and in front of",
    "rf": "to the right and in front of",
    "lb": "to the left and behind",
    "rb": "to the right and behind",
    "m": "in close proximity to"
}
CLOCK_TO_SECTOR = {1: "rf", 2: "rf", 3: "r", 4: "rb", 5: "rb", 6: "b", 7: "lb", 8: "lb", 9: "l", 10: "lf", 11: "lf",
                   12: "f"}
SECTORS = ["rf", "rb", "lb", "lf", "f", "r", "b", "l"]


#{
# l->2,3,4, lf->4,5, f->5,6,7, rf->7,8 -> r->8,9,10, rb->10 11 b->11 12 1 lb->1 2, m-determine if rays intersect with my line
# }
#
# 12->f, 1->rf, 2->rf, 3->r, 4->rb, 5->rb, 6 ->b, 7->lf, 8->lb 9->l, 10 lf, 11lf


# 345, 15 -> front; 15-75 right-front; 75->105 right; 105-165->right back; 165-195->back; 195->255->left back; 255->285 left; 285, 345->left front
#