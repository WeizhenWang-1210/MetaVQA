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
        ["follow"],["pass by"],["collide"],["head toward"],["drive alongside"]
    ],
    "<deed_without_o>":[
        ["turn right"], ["turn left"]
    ],
    "<p>":[
        ["nil"],["red"],["yellow"],["green"],["blue"],["black"],["orange"],["white"]
    ],
    "<t>":[
        ["nil"], ["car"],["policecar"]
    ],
    "<dir>":[
        ["nil"],["<tdir>", "<o>"]
    ],
    "<tdir>":[
        ["left"], ["right"], ["front"], ["back"], ["left and front"],["right and front"],["left and back"], ["right and back"],
    ]
}