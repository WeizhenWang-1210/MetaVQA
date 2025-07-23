import json
super_categories = dict(
    identify_color="spatial",
    identify_type="spatial",
    identify_distance="spatial",
    identify_position="spatial",
    identify_heading="spatial",
    pick_closer="spatial",
    predict_crash_ego_still="embodied",
    describe_scenario="spatial",
    predict_crash_ego_dynamic="embodied",
    relative_distance="spatial",
    relative_position="spatial",
    relative_heading="spatial",
    relative_predict_crash_still="spatial",
    relative_predict_crash_dynamic="spatial",
    identify_closest="spatial",
    identify_leftmost="spatial",
    identify_rightmost="spatial",
    identify_frontmost="spatial",
    identify_backmost="spatial",
    order_closest="spatial",
    order_leftmost="spatial",
    order_rightmost="spatial",
    order_frontmost="spatial",
    order_backmost="spatial",
    describe_sector="spatial",
    describe_distance="spatial",
    embodied_distance="embodied",
    embodied_sideness="embodied",
    embodied_collision="embodied",
    grounding="grounding"
)

stats = json.load(open("/home/weizhen/vqa_compositions.json","r"))



from collections import defaultdict
supertypes = defaultdict(lambda:0)

for type, count in stats["type"].items():
    supertypes[super_categories[type]] += count

stats["supertype"] = supertypes
stats["type2super"] = super_categories
from pprint import pprint
pprint(stats)
json.dump(
    stats,
    open("/home/weizhen/vqa_compositions.json","w"),
    indent=1
)
#print(supertypes)
