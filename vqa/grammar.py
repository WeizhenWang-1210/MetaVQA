CFG_GRAMMAR = {
    "<o>":[
        ["<s>", "<p>", "<t>", "<dir>", "<a>"],
        ["us"]
    ],
    "<s>":[
        ["nil"],["visible"],["parked"],["moving"],["accelerating"],["decelerating"],["turning"]
    ],
    "<a>":[
        ["nil"], ["<deed_with_o>","<o>"],["<deed_without_o>"]
    ],
    "<deed_with_o>":[
        ["follow"],["pass by"],["collide with"],["head toward"],["drive alongside"]
    ],
    "<deed_without_o>":[
        ["turn right"], ["turn left"]
    ],
    "<p>":[
        ["nil"],["Red"],["Blue"],["Green"],["Yellow"],["Black"],["White"],["Purple"],["Orange"],["Brown"],["Gray"],
        ["Cyan"],["Lime"],["Pink"],["Gold"],["Teal"],["Maroon"],["Navy"],["Olive"],["Silver"],["Violet"]
    ],
    "<t>":[
        ["nil"], ["Bus"],["Caravan"],["Coupe"],["FireTruck"],["Hatchback"],["Jeep"],["Pickup"],["Policecar"],["SUV"],
        ["SchoolBus"], ["Sedan"], ["SportCar"],["Truck"]
    ],
    "<dir>":[
        ["nil"],["<tdir>", "<o>"]
    ],
    "<tdir>":[
        ["l"], ["r"], ["f"], ["b"], ["lf"],["rf"],["lb"], ["rb"],
    ]
}


STATIC_GRAMMAR = {
    "<o>":[
        ["<p>", "<t>", "<dir>"],
        ["us"]
    ],
    "<p>":[
        ["nil"],["Red"],["Blue"],["Green"],["Yellow"],["Black"],["White"],["Purple"],["Orange"],["Brown"],["Gray"],
        ["Cyan"],["Lime"],["Pink"],["Gold"],["Teal"],["Maroon"],["Navy"],["Olive"],["Silver"],["Violet"]
    ],
    "<t>":[
        ["nil"], ["Bus"],["Caravan"],["Coupe"],["FireTruck"],["Hatchback"],["Jeep"],["Pickup"],["Policecar"],["SUV"],
        ["SchoolBus"], ["Sedan"], ["SportCar"],["Truck"]
    ],
    "<dir>":[
        ["nil"],["<tdir>", "<o>"]
    ],
    "<tdir>":[
        ["l"], ["r"], ["f"], ["b"], ["lf"],["rf"],["lb"], ["rb"],
    ]
}