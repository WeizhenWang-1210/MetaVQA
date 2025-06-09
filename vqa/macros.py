NUSC_PATH = '/bigdata/datasets/nuscenes'

NUSC_VERSION = 'v1.0-trainval'

NUSC_EGO_SHAPE = (1.730, 4.084, 1.562)

IGNORED_NUSC_TYPE = (
    "noise", "human.pedestrian.personal_mobility", "movable_object.pushable_pullable", "movable_object.debris",
    "static_object.bicycle_rack", "flat.driveable_surface", "flat.sidewalk", "flat.terrain", "flat.other",
    "static.manmade",
    "static.vegetation", "static.other")

ALL_NUSC_TYPE = {
    "noise": 'noise',
    "human.pedestrian.adult": 'Pedestrian',
    "human.pedestrian.child": 'Pedestrian',
    "human.pedestrian.wheelchair": 'Wheelchair',
    "human.pedestrian.stroller": 'Stroller',
    "human.pedestrian.personal_mobility": 'p.mobility',
    "human.pedestrian.police_officer": 'Police_officer',
    "human.pedestrian.construction_worker": 'Construction_worker',
    "animal": 'Animal',
    "vehicle.car": 'Car',
    "vehicle.motorcycle": 'Motorcycle',
    "vehicle.bicycle": 'Bike',
    "vehicle.bus.bendy": 'Bus',
    "vehicle.bus.rigid": 'Bus',
    "vehicle.truck": 'Truck',
    "vehicle.construction": 'Construction_vehicle',
    "vehicle.emergency.ambulance": 'Ambulance',
    "vehicle.emergency.police": 'Policecar',
    "vehicle.trailer": 'Trailer',
    "movable_object.barrier": 'Barrier',
    "movable_object.trafficcone": 'Cone',
    "movable_object.pushable_pullable": 'push/pullable',
    "movable_object.debris": 'debris',
    "static_object.bicycle_rack": 'bicycle racks',
    "flat.driveable_surface": 'driveable',
    "flat.sidewalk": 'sidewalk',
    "flat.terrain": 'terrain',
    "flat.other": 'flat.other',
    "static.manmade": 'manmade',
    "static.vegetation": 'vegetation',
    "static.other": 'static.other',
    "vehicle.ego": "ego"
}

PAIRING_PATH = "/bigdata/weizhen/data_nusc_multiview_trainval.json"

NUSCENES_SN_PATH = "/bigdata/datasets/scenarionet/nuscenes/trainval"