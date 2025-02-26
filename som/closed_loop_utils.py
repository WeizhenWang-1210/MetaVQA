from metadrive.envs.base_env import BaseEnv
import numpy as np
from vqa.functionals import identify_angle
from som.embodied_utils import classify_distance, l2_distance, find_sector, get_end_sector, classify_speed, \
    describe_speed, ACTION, determine_collisions
from som.static_question_generation import angle2sector, SECTORS, TYPES_WITHOUT_HEADINGS
from vqa.object_node import extrapolate_bounding_boxes, box_trajectories_overlap, transform_vec
from vqa.configs.NAMESPACE import POSITION2CHOICE
from vqa.annotation_utils import annotate_type


def computeADE(traj1, traj2):
    """
    Traj1 is Ground Truth
    Traj2 is the generated trajectory, which can be shorter than GT
    """
    traj1 = traj1[:traj2.shape[0], :]
    distances = np.linalg.norm((traj1 - traj2), axis=1)
    ade = np.mean(distances)
    return float(ade)


def computeFDE(traj1, traj2):
    """
    Traj1 is Ground Truth
    Traj2 is the generated trajectory, which can be shorter than GT
    """
    t = traj2.shape[0]
    assert traj1.shape[-1]==2, traj1.shape
    assert not (traj1[t - 1, :] == 0.0).all()
    assert not (traj2[t - 1, :] == 0.0).all()
    return float(np.linalg.norm(traj1[t - 1, :] - traj2[t - 1, :]))


def absoluteFDE(traj1, traj2):
    return float(np.linalg.norm(traj1[- 1, :] - traj2[- 1, :]))


def option_strings(l):
    str_list = []
    ascii_offset = ord("A")
    for idx, item in enumerate(l):
        str_list.append(
            f"({chr(ascii_offset + idx)}) {item}"
        )
    option_str = "; ".join(str_list)
    return option_str


def CoT_prompts(env: BaseEnv, label2id: dict):
    def prompt_distance(label):
        obj_id = label2id[label]
        object = env.engine.get_object(obj_id)[obj_id]
        ego = env.agent
        question = f"Please tell me how far object <{label}> is from us. Classify the answer into: (A) Very close(0-2m); (B) Close(2-10m); (C) Medium(10-30m); (D) Far(30m-)."
        options = ["very close", "close", "medium", "far"]
        answer = classify_distance(l2_distance(ego.position, object.position))
        ground_truth = chr(ord("A") + options.index(answer))
        option2answer = dict(
            A="Very close(0-2m)",
            B="Close(2-10m)",
            C="Medium(10-30m)",
            D="Far(30m-)"
        )
        return question, ground_truth, answer, option2answer

    def prompt_position(label):
        obj_id = label2id[label]
        object = env.engine.get_object(obj_id)[obj_id]
        ego = env.agent
        options = list(POSITION2CHOICE.values())
        option_str = option_strings(options)
        question = f"Please tell me the relative position of object <{label}> with respect to us. Select the best option from: {option_str}."
        answer = POSITION2CHOICE[find_sector(object.bounding_box, ego.bounding_box, ego.heading)]
        ground_truth = chr(ord("A") + options.index(answer))

        option2answer = {
            chr(ord("A") + options.index(opt)): opt for opt in options
        }

        return question, ground_truth, answer, option2answer

    def prompt_heading(label):
        obj_id = label2id[label]
        object = env.engine.get_object(obj_id)[obj_id]
        type = annotate_type(object)
        if type in TYPES_WITHOUT_HEADINGS:
            return None, None, None, None
        ego = env.agent
        options = [POSITION2CHOICE[sector] for sector in SECTORS]
        option_str = option_strings(options)
        question = f"Please describe the heading direction of object <{label}>. Select the best option from: {option_str}."
        object_angle = identify_angle(ego.position, ego.heading)([[object]])[object.id]
        heading = angle2sector(object_angle)
        answer = SECTORS.index(heading)
        ground_truth = chr(ord("A") + answer)

        option2answer = {
            chr(ord("A") + SECTORS.index(sector)): POSITION2CHOICE[sector] for sector in SECTORS
        }

        return question, ground_truth, POSITION2CHOICE[heading], option2answer

    def prompt_collision(label):
        obj_id = label2id[label]
        object = env.engine.get_object(obj_id)[obj_id]
        type = annotate_type(object)
        if type in TYPES_WITHOUT_HEADINGS:
            return None, None, None, None
        ego = env.agent
        options = ["Yes", "No"]
        option_str = option_strings(options)
        question = f"Suppose object <{label}> proceed along its current heading. Will it collides into us if we stay still? Select the best option from: {option_str}."
        init_center = object.position
        extrapolated_centers = [list(np.array(object.heading) * i + np.array(init_center)) for i in range(50)]
        extrapolated_boxes = extrapolate_bounding_boxes(extrapolated_centers,
                                                        np.arctan2(object.heading[1], object.heading[0]),
                                                        object.bounding_box)
        ego_boxes = [ego.bounding_box for i in range(50)]
        crash = box_trajectories_overlap(extrapolated_boxes, ego_boxes)
        answer = "Yes" if crash else "No"
        ground_truth = "A" if crash else "B"
        option2answer = dict(A="Yes", B="No")
        return question, ground_truth, answer, option2answer

    def prompt_action_distance(action, duration, speed):
        """
        action in 0-4, duration in step, speed in m/s
        """
        speed_class = classify_speed(speed)
        speed_str = describe_speed(speed_class)
        action_str = ACTION.get_action(action)
        question = f"Our current speed is {speed_class}{speed_str}, and we perform action \"{action_str}\" for {round(duration / 10, 1)} seconds. How far will we end up from our current position? Select the best option from: (A) Very close(0-2m); (B) Close(2-10m); (C) Medium(10-30m); (D) Far(30m-)."
        options = ["very close", "close", "medium", "far"]
        end_distance, _ = get_end_sector(action=action, duration=duration, speed=speed)
        ground_truth = chr(ord("A") + options.index(end_distance))

        option2answer = dict(
            A="Very close(0-2m)",
            B="Close(2-10m)",
            C="Medium(10-30m)",
            D="Far(30m-)"
        )

        return question, ground_truth, end_distance, option2answer

    def prompt_action_position(action, duration, speed):
        """
        action in 0-4, duration in step, speed in m/s
        """
        speed_class = classify_speed(speed)
        speed_str = describe_speed(speed_class)
        action_str = ACTION.get_action(action)
        question = f"Our current speed is {speed_class}{speed_str}, and we perform action \"{action_str}\" for {round(duration / 10, 1)} seconds. Which sector will we end up? Select the best option from: (A) left-front; (B) front; (C) right-front."
        options = ["left-front", "front", "right-front"]
        _, end_side = get_end_sector(action=action, duration=duration, speed=speed)
        if end_side == "m":
            end_side = "f"
        answer = POSITION2CHOICE[end_side]
        ground_truth = chr(ord("A") + options.index(answer))
        option2answer = {
            chr(ord("A") + options.index(opt)): opt for opt in options
        }
        return question, ground_truth, answer, option2answer

    def prompt_action_collision(action, duration, speed, label):
        """
        action in 0-4, duration in step, speed in m/s
        """
        obj_id = label2id[label]
        object = env.engine.get_object(obj_id)[obj_id]
        ego = env.agent
        object_box = object.bounding_box
        object_box_ego = transform_vec(ego.position, ego.heading, object_box)
        speed_class = classify_speed(speed)
        speed_str = describe_speed(speed_class)
        action_str = ACTION.get_action(action)
        question = f"Our current speed is {speed_class}{speed_str}, and we perform action \"{action_str}\" for {round(duration / 10, 1)} seconds. Will we run into object <{label}>, provided that it remains still? Select the best option from: (A) Yes; (B) No."
        will_collide, collision_time = determine_collisions(obj_box=object_box_ego, action=action, speed=speed,
                                                            duration=duration)
        answer = "Yes" if will_collide else "No"
        ground_truth = "A" if will_collide else "B"
        return question, ground_truth, answer, dict(A="Yes", B="No")

    questions = []
    situational_type2func = {
        "distance": prompt_distance,
        "position": prompt_position,
        "heading": prompt_heading,
        "collision": prompt_collision,
    }
    situational_dicts = dict()

    indexed_dict = dict()

    for label in label2id.keys():
        #print(label)
        #tmp = dict()
        for question_type, func in situational_type2func.items():
            question, option, answer, opt2answer = func(label)
            #tmp[question_type] = dict(
            #    question=question, option=option, answr=answer
            #)
            #questions.append(question)
            if not question is None:
                indexed_dict[f"{label}_{question_type}"] = dict(
                    question=question, option=option, answer=answer, option2answer=opt2answer
                )
        #situational_dicts[label] = tmp

    embodied_type2func = {
        "distance": prompt_action_distance,
        "position": prompt_action_position,
        "collision": prompt_action_collision,
    }
    actions = [ACTION.TURN_LEFT, ACTION.TURN_RIGHT, ACTION.SLOW_DOWN, ACTION.BRAKE, ACTION.KEEP_STRAIGHT]

    embodied_dicts = dict()
    for question_type, func in embodied_type2func.items():
        if question_type != "collision":
            tmp = dict()
            for action in actions:

                question, option, answer, opt2answer = func(action, 20, env.agent.speed)
                #tmp[action] = dict(
                #    question=question, option=option, answr=answer
                #)
                #questions.append(question)
                if not question is None:
                    indexed_dict[f"ego_{question_type}_{action}"] = dict(
                        question=question, option=option, answer=answer, option2answer=opt2answer
                    )

            #embodied_dicts[question_type] = tmp
        else:
            for label in label2id.keys():
                #tmp = dict()
                for action in actions:
                    question, option, answer, opt2answer = prompt_action_collision(action, 20, env.agent.speed, label)
                    #tmp[action] = dict(
                    #    question=question, option=option, answr=answer
                    #)
                    #questions.append(question)
                    if not question is None:
                        indexed_dict[f"{label}_ego-{question_type}_{action}"] = dict(
                            question=question, option=option, answer=answer, option2answer=opt2answer
                        )

                #embodied_dicts[question_type][label] = tmp
    return indexed_dict
