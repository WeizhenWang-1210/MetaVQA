# TODO Cautious
CFG_GRAMMAR = {
    "<o>": [
        ["<s>", "<p>", "<t>", "<dir>", "<a>"],
        ["us"]
    ],
    "<s>": [
        ["nil"], ["parked"], ["accelerating"], ["decelerating"], ["turn_left"], ["turn_right"]
    ],
    "<a>": [
        ["nil"], ["<passive_deed>", "<o>"], ["<active_deed>", "<o>"]
    ],
    "<passive_deed>": [
        ["followed"], ["passed_by"], ["headed_toward"], ["accompanied_by"]
    ],
    "<active_deed>": [
        ["follow"], ["pass_by"], ["head_toward"], ["move_alongside"],
    ],
    "<p>": [
        ["nil"], ["Red"], ["Blue"], ["Green"], ["Yellow"], ["Black"], ["White"], ["Purple"], ["Orange"], ["Brown"],
        ["Gray"],
    ],
    "<t>": [
        ["nil"], ["Hatchback"], ["Pickup"], ["Policecar"], ["SUV"], ["Sedan"], ["SportCar"], ["Truck"], ["Pedestrian"], ["vehicle"]
    ],
    "<dir>": [
        ["nil"], ["<tdir>", "<o>"]
    ],
    "<tdir>": [
        ["l"], ["r"], ["f"], ["b"], ["lf"], ["rf"], ["lb"], ["rb"],
    ]
}

NO_STATE_CFG = {
    "<o>": [
        ["<s>", "<p>", "<t>", "<dir>", "<a>"],
        ["us"]
    ],
    "<s>": [
        ["nil"]
    ],
    "<a>": [
        ["nil"], ["<passive_deed>", "<o>"], ["<active_deed>", "<o>"]
    ],
    "<passive_deed>": [
        ["followed"], ["passed_by"], ["headed_toward"], ["accompanied_by"]
    ],
    "<active_deed>": [
        ["follow"], ["pass_by"], ["head_toward"], ["move_alongside"],
    ],
    "<p>": [
        ["nil"], ["Red"], ["Blue"], ["Green"], ["Yellow"], ["Black"], ["White"], ["Purple"], ["Orange"], ["Brown"],
        ["Gray"],
    ],
    "<t>": [
        ["nil"], ["Hatchback"], ["Pickup"], ["Policecar"], ["SUV"], ["Sedan"], ["SportCar"], ["Truck"], ["Pedestrian"],
        ["vehicle"]
    ],
    "<dir>": [
        ["nil"], ["<tdir>", "<o>"]
    ],
    "<tdir>": [
        ["l"], ["r"], ["f"], ["b"], ["lf"], ["rf"], ["lb"], ["rb"],
    ]
}

NO_COLOR_CFG = {
    "<o>": [
        ["<s>", "<p>", "<t>", "<dir>", "<a>"],
        ["us"]
    ],
    "<ox>": [
        ["<s>", "<px>", "<t>", "<dir>", "<a>"],
        ["us"]
    ],
    "<s>": [
        ["nil"], ["parked"], ["accelerating"], ["decelerating"], ["turn_left"], ["turn_right"]
    ],
    "<a>": [
        ["nil"], ["<passive_deed>", "<ox>"], ["<active_deed>", "<ox>"]
    ],
    "<passive_deed>": [
        ["followed"], ["passed_by"], ["headed_toward"], ["accompanied_by"]
    ],
    "<active_deed>": [
        ["follow"], ["pass_by"], ["head_toward"], ["move_alongside"],
    ],
    "<p>": [
        ["nil"]
    ],
    "<px>": [
        ["nil"], ["Red"], ["Blue"], ["Green"], ["Yellow"], ["Black"], ["White"], ["Purple"], ["Orange"], ["Brown"],
        ["Gray"],
    ],
    "<t>": [
        ["nil"], ["Hatchback"], ["Pickup"], ["Policecar"], ["SUV"], ["Sedan"], ["SportCar"], ["Truck"], ["Pedestrian"], ["vehicle"]
    ],
    "<dir>": [
        ["nil"], ["<tdir>", "<ox>"]
    ],
    "<tdir>": [
        ["l"], ["r"], ["f"], ["b"], ["lf"], ["rf"], ["lb"], ["rb"],
    ]
}



NO_TYPE_CFG = {
    "<o>": [
        ["<s>", "<p>", "<t>", "<dir>", "<a>"],
        ["us"]
    ],
    "<ox>": [
        ["<s>", "<p>", "<tx>", "<dir>", "<a>"],
        ["us"]
    ],
    "<s>": [
        ["nil"], ["parked"], ["accelerating"], ["decelerating"], ["turn_left"], ["turn_right"]
    ],
    "<a>": [
        ["nil"], ["<passive_deed>", "<ox>"], ["<active_deed>", "<ox>"]
    ],
    "<passive_deed>": [
        ["followed"], ["passed_by"], ["headed_toward"], ["accompanied_by"]
    ],
    "<active_deed>": [
        ["follow"], ["pass_by"], ["head_toward"], ["move_alongside"],
    ],
    "<p>": [
        ["nil"], ["Red"], ["Blue"], ["Green"], ["Yellow"], ["Black"], ["White"], ["Purple"], ["Orange"], ["Brown"],
        ["Gray"],
    ],
    "<t>": [
        ["nil"]
    ],
    "<tx>": [
        ["nil"], ["Hatchback"], ["Pickup"], ["Policecar"], ["SUV"], ["Sedan"], ["SportCar"], ["Truck"], ["Pedestrian"],
        ["vehicle"]
    ],
    "<dir>": [
        ["nil"], ["<tdir>", "<ox>"]
    ],
    "<tdir>": [
        ["l"], ["r"], ["f"], ["b"], ["lf"], ["rf"], ["lb"], ["rb"],
    ]
}


STATIC_GRAMMAR = {
    "<o>": [
        ["<s>", "<p>", "<t>", "<dir>", "<a>"],
        ["us"]
    ],
    "<s>": [
        ["nil"]
    ],
    "<p>": [
        ["nil"], ["Red"], ["Blue"], ["Green"], ["Yellow"], ["Black"], ["White"], ["Purple"], ["Orange"], ["Brown"],
        ["Gray"],
    ],
    "<t>": [
        ["nil"], ["Hatchback"], ["Pickup"], ["Policecar"], ["SUV"], ["Sedan"], ["SportCar"], ["Truck"], ["Pedestrian"], ["vehicle"]
    ],
    "<dir>": [
        ["nil"], ["<tdir>", "<o>"]
    ],
    "<tdir>": [
        ["l"], ["r"], ["f"], ["b"], ["lf"], ["rf"], ["lb"], ["rb"],
    ],
    "<a>": [
        ["nil"]
    ],
}

NO_COLOR_STATIC = {
    "<o>": [
        ["<s>", "<p>", "<t>", "<dir>", "<a>"],
        ["us"]
    ],
    "<s>": [
        ["nil"]
    ],
    "<p>": [
        ["nil"]
    ],
    "<t>": [
        ["nil"], ["Hatchback"], ["Pickup"], ["Policecar"], ["SUV"], ["Sedan"], ["SportCar"], ["Truck"], ["Pedestrian"],
        ["vehicle"]
    ],
    "<dir>": [
        ["nil"], ["<tdir>", "<ox>"]
    ],
    "<ox>": [
        ["<s>", "<px>", "<t>", "<dir>", "<a>"],
        ["us"]
    ],
    "<px>": [
        ["nil"], ["Red"], ["Blue"], ["Green"], ["Yellow"], ["Black"], ["White"], ["Purple"], ["Orange"], ["Brown"],
        ["Gray"],
    ],
    "<tdir>": [
        ["l"], ["r"], ["f"], ["b"], ["lf"], ["rf"], ["lb"], ["rb"],
    ],
    "<a>": [
        ["nil"]
    ],
}

NO_TYPE_STATIC = {
    "<o>": [
        ["<s>", "<p>", "<t>", "<dir>", "<a>"],
        ["us"]
    ],
    "<s>": [
        ["nil"]
    ],
    "<p>": [
        ["nil"], ["Red"], ["Blue"], ["Green"], ["Yellow"], ["Black"], ["White"], ["Purple"], ["Orange"], ["Brown"],
        ["Gray"],
    ],
    "<t>": [
        ["nil"]
    ],
    "<dir>": [
        ["nil"], ["<tdir>", "<ox>"]
    ],
    "<ox>": [
        ["<s>", "<p>", "<tx>", "<dir>", "<a>"],
        ["us"]
    ],
    "<tx>": [
        ["nil"], ["Hatchback"], ["Pickup"], ["Policecar"], ["SUV"], ["Sedan"], ["SportCar"], ["Truck"], ["Pedestrian"],
        ["vehicle"]
    ],
    "<tdir>": [
        ["l"], ["r"], ["f"], ["b"], ["lf"], ["rf"], ["lb"], ["rb"],
    ],
    "<a>": [
        ["nil"]
    ],
}
