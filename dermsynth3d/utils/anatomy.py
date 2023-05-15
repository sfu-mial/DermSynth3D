class Anatomy:
    background = 0
    head = 1
    upper_torso = 2
    lower_torso = 3
    hips = 4
    upper_leg_left = 5
    upper_leg_right = 6
    lower_leg_left = 7
    lower_leg_right = 8
    feet_left = 9
    feet_right = 10
    upper_arm_left = 11
    upper_arm_right = 12
    lower_arm_left = 13
    lower_arm_right = 14
    hand_left = 15
    hand_right = 16


class SimpleAnatomy:
    background = 0
    head = 1
    torso = 2
    hips = 3
    legs = 4
    feet = 5
    arms = 6
    hands = 7

    n_labels = 8

    @staticmethod
    def to_string(idx):
        if idx == SimpleAnatomy.background:
            return "BG"
        if idx == SimpleAnatomy.head:
            return "head"
        if idx == SimpleAnatomy.torso:
            return "torso"
        if idx == SimpleAnatomy.hips:
            return "hips"
        if idx == SimpleAnatomy.legs:
            return "legs"
        if idx == SimpleAnatomy.feet:
            return "feet"
        if idx == SimpleAnatomy.arms:
            return "arms"
        if idx == SimpleAnatomy.hands:
            return "hands"

    @staticmethod
    def to_simple_anatomy(x):
        x[x == Anatomy.background] = SimpleAnatomy.background
        x[x == Anatomy.head] = SimpleAnatomy.head
        x[x == Anatomy.upper_torso] = SimpleAnatomy.torso
        x[x == Anatomy.lower_torso] = SimpleAnatomy.torso
        x[x == Anatomy.hips] = SimpleAnatomy.hips
        x[x == Anatomy.upper_leg_left] = SimpleAnatomy.legs
        x[x == Anatomy.upper_leg_right] = SimpleAnatomy.legs
        x[x == Anatomy.lower_leg_left] = SimpleAnatomy.legs
        x[x == Anatomy.lower_leg_right] = SimpleAnatomy.legs
        x[x == Anatomy.feet_left] = SimpleAnatomy.feet
        x[x == Anatomy.feet_right] = SimpleAnatomy.feet
        x[x == Anatomy.upper_arm_left] = SimpleAnatomy.arms
        x[x == Anatomy.upper_arm_right] = SimpleAnatomy.arms
        x[x == Anatomy.lower_arm_left] = SimpleAnatomy.arms
        x[x == Anatomy.lower_arm_right] = SimpleAnatomy.arms
        x[x == Anatomy.hand_left] = SimpleAnatomy.hands
        x[x == Anatomy.hand_right] = SimpleAnatomy.hands
