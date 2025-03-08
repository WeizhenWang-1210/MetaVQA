import json
import glob
import re

path_template = "/home/chenda/evaluations/vqa/InternVL_37500/*_test_results.json"
merged_path = "/home/weizhen/experiments/main/InternVL_37500_test_results.json"
stat_path = "/home/weizhen/experiments/main/InternVL_37500_test_results_stats.json"
results = glob.glob(path_template)

final = dict()

NOT_YET_PARSED = ["describe_scenario"]
domained_path = "/data_weizhen/metavqa_cvpr/datasets/test/test/test.json"
original_qa = json.load(open(domained_path, "r"))

for result in results:
    qas = json.load(open(result, "r"))
    for key, value in qas.items():
        assert key not in final.keys(), "Repetitive keys found!"
        assert key in original_qa.keys(), "You should have domain information"
        final[key] = value
        final[key]["domain"] = original_qa[key]["domain"]


def parse_response_llava(all_response, zero_shot=False):
    pattern = r'"ANSWER":(.*?)"(.*?)"'
    response = all_response.split("assistant")[-1]
    #print(response)
    if zero_shot:
        matches = re.findall(pattern, response)
        if len(matches) > 0:
            #print(f"Match:{matches[-1]}")
            #print("___________")
            for choice in ["(A)", "(B)", "(C)", "(D)", "(a)", "(b)", "(c)", "(d)","A)","B)","C)","D)","a)","b)","c)","d)"]:
                #print(choice, matches[-1][-1]])
                if choice in matches[-1][-1]:
                    print("here")
                    return choice[-2]#matches[-1][-1][1]
            return matches[-1][-1]
        else:
            #print(f"No Match")
            #print("___________")
            return ""
    else:
        return ""

def parse_response_finetuned(all_response):
    pattern = r'\t"ANSWER":(.*?)"(.*?)"'
    response = all_response.split("assistant")[-1]
    matches = re.findall(pattern, response)
    if len(matches) > 0:
        return matches[-1][-1]
    else:
        return ""

def parse_response(response, zero_shot=False):
    if zero_shot:
        valid_choices = ["(A)", "(B)", "(C)", "(D)", "(a)", "(b)", "(c)", "(d)", " A", " B", "C", "D"]
        for valid_choice in valid_choices:
            if valid_choice in response:
                return valid_choice
        return ""
    else:
        return response[0]

def accuracy_analysis(qa_records, zero_shot=False):
    statistics = dict()
    total_correct = total = 0
    for qid, record in qa_records.items():
        if record["type"] in NOT_YET_PARSED:
            continue
        if record["type"] not in statistics.keys():
            statistics[record["type"]] = dict(
                total=0, correct=0
            )
        statistics[record["type"]]["total"] += 1
        if zero_shot:
            statistics[record["type"]]["correct"] += 1 if record["final_choice"].upper() == "({})".format(
                record["answer"]).upper() else 0
        else:
            statistics[record["type"]]["correct"] += 1 if record["final_choice"].upper() == record[
                "answer"].upper() else 0
    for type, stat in statistics.items():
        stat["accuracy"] = stat["correct"] / stat["total"]
        total += stat["total"]
        total_correct += stat["correct"]
    return statistics, total, total_correct


def accuracy_analysis_llava(qa_records, zero_shot=False):
    statistics = dict()
    total_correct = total = 0
    for qid, record in qa_records.items():
        if record["type"] in NOT_YET_PARSED:
            continue
        if record["type"] not in statistics.keys():
            statistics[record["type"]] = dict(
                total=0, correct=0
            )
        statistics[record["type"]]["total"] += 1
        if zero_shot:
            statistics[record["type"]]["correct"] += 1 if record["final_choice"].upper() == record[
                "answer"].upper() else 0
        else:
            statistics[record["type"]]["correct"] += 1 if record["final_choice"].upper() == record[
                "answer"].upper() else 0
    for type, stat in statistics.items():
        stat["accuracy"] = stat["correct"] / stat["total"]
        total += stat["total"]
        total_correct += stat["correct"]
    return statistics, total, total_correct


def accuracy_by_domain(qa_records, zero_shot=False):
    statistics = dict()
    total_correct = total = 0

    for qid, record in qa_records.items():
        if record["type"] in NOT_YET_PARSED:
            continue
        identifier = "_".join([record["domain"], record["type"]])

        if identifier not in statistics.keys():
            statistics[identifier] = dict(
                total=0, correct=0
            )
        statistics[identifier]["total"] += 1
        if zero_shot:
            statistics[identifier]["correct"] += 1 if record["final_choice"].upper() == record[
                "answer"].upper() else 0
        else:
            statistics[identifier]["correct"] += 1 if record["final_choice"].upper() == record[
                "answer"].upper() else 0
    for type, stat in statistics.items():
        stat["accuracy"] = stat["correct"] / stat["total"]
        total += stat["total"]
        total_correct += stat["correct"]
    return statistics, total, total_correct


def parse_internvl(response):
    matches = list(re.finditer(r'\(A\)|\(B\)|\(C\)|\(D\)', response))
    # Get the last match if it exists
    if matches:
        last_occurrence = matches[-1]
        print("Last occurrence:", last_occurrence.group())
        #print("Position:", last_occurrence.start())
        print("Context:", response[last_occurrence.start()-200:])
        return last_occurrence.group()[1]
    else:
        print("No matches found")
        return ""

def parse_llavanext(response, reasoning):
    matches = list(re.finditer(r'\(A\)|\(B\)|\(C\)|\(D\)', response))
    # Get the last match if it exists
    if matches:
        last_occurrence = matches[-1]
        print("Last occurrence:", last_occurrence.group())
        # print("Position:", last_occurrence.start())
        print("Context:", response[last_occurrence.start() - 200:])
        return last_occurrence.group()[1]
    else:
        print("No matches found")
        return ""



for qid in final.keys():
    #print(final[qid]["model_response"])
    #print("______")
    #choice = parse_response_llava(final[qid]["model_response"], True)
    choice = parse_response_finetuned(final[qid]["model_response"])
    print(final[qid]["model_response"])
    print(choice)
    print("______")
    final[qid]["final_choice"] = choice

statistics, total, total_correct = accuracy_by_domain(final, True)

sim_correct = real_correct = sim_total = real_total = 0

for key, record in statistics.items():
    if key.startswith("sim"):
        sim_total += record["total"]
        sim_correct += record["correct"]
    elif key.startswith("real"):
        real_total += record["total"]
        real_correct += record["correct"]

stats = dict(
    total_questions=total, total_correct=total_correct, total_sims=sim_total, total_reals=real_total,
    sim_correct=sim_correct, real_correct=real_correct, stats=statistics
)
json.dump(final, open(merged_path, "w"), indent=1)
json.dump(stats, open(stat_path, "w"), indent=1)
