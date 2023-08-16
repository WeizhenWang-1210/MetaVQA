import objaverse
import asyncio
import trimesh
#car packs 20f9af9b8a404d5cb022ac6fe87f21f5
#landrover ffba8330b24d42daac8905fa0102eb97
#2021 Lamborghini Countach LPI 800-4 d76b94884432422b966d1a7f8815afb5
#Porsche 911 Carrera 4S d01b254483794de3819786d93e0e1ebf
#pickup 40c94d8b31f94df3bd80348cac8624f1
#Mahindra thar 4by4 f045413d71d743c58682881cb7421d64
if __name__ == "__main__":
    #uids = objaverse.load_uids()
    objects = [
        "ffba8330b24d42daac8905fa0102eb97",
    ]
    objects = objaverse.load_objects(objects,1)
    trimesh.load(list(objects.values())[0]).show()
    #trimesh.load("metadrive/assets/models/lambo/vehicle.glb").show()

    car_spec = {
        "lambo":{
            'length':4.87,
            'width':2.099,
            'height':1.139,
            'mass':1595,
            'wheelbase':2.7,
            'max_speed':355,
            'front_wheel_width':0.255,
            'back_wheel_width':0.355
            #'front_wheel_diameter': 2 * 25.5 * 0.3 + 20*2.54 15.3 + 50.8
            #'back_wheel_diameter':  2 * 25.5 * 0.25 + 21*2.54 
        }
    }


    """
    LAMBO Spec
    TIRE_RADIUS = 0.3305#0.313
    TIRE_WIDTH = 0.255#0.25
    MASS = 1595#1100
    LATERAL_TIRE_TO_CENTER = 1#0.815
    FRONT_WHEELBASE = 1.36#1.05234
    REAR_WHEELBASE = 1.45#1.4166
    #path = ['ferra/vehicle.gltf', (1, 1, 1), (0, 0.075, 0.), (0, 0, 0)]
    path = ['lambo/vehicle.glb', (0.5,0.5,0.5), (1.09, 0, 0.6), (0, 0, 0)]"""

   