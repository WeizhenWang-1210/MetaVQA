import json
from os.path import join
stats =[
    "/home/weizhen/experiments/main/InternVL_test_results.json",
    "/home/weizhen/experiments/main/InternVL_trainval_test_results.json",
]

preserved = dict()
learned = dict()
unlearned = dict()
forget = dict()

zeroshot = json.load(open(stats[0]))
finetuned = json.load(open(stats[1]))


for key in zeroshot.keys():
    assert zeroshot[key]["question"] == finetuned[key]["question"]
    answer_zeroshot = zeroshot[key]["final_choice"]
    answer_finetuned = finetuned[key]["final_choice"]
    gt = zeroshot[key]["answer"]

    tup = (answer_zeroshot, answer_finetuned, gt)
    if answer_zeroshot != gt and answer_finetuned == gt:
        learned[key] = tup
    elif answer_zeroshot != gt and answer_finetuned != gt:
        unlearned[key] = tup
    elif answer_zeroshot == gt and answer_finetuned == gt:
        preserved[key] = tup
    else:
        forget[key] = tup


base = "/home/weizhen/experiments/demo"

json.dump(
    preserved, open(join(base, "preserved.json"),"w"), indent=2
)
json.dump(
    learned, open(join(base, "learned.json"),"w"), indent=2
)
json.dump(
    unlearned, open(join(base, "unlearned.json"),"w"), indent=2
)
json.dump(
    forget, open(join(base, "forget.json"),"w"), indent=2
)







