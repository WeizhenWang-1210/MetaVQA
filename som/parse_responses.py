import json
import re

def parse_response(response, answer2opt):
    if "assistant" in response:
        generated = response.split("assistant")[-1][1:]
    elif "ASSISTANT" in response:
        generated = response.split("ASSISTANT")[-1][1:]
    else:
        generated = response
    #print("!!!!!!!!!!!!!!!!")
    #print(generated)
    #print("!!!!!!!!!!!!!!!!")
    answer = ""
    if len(generated) == 1:
        #Generated only a single token. Use that as an option.
        answer = generated[-1]
    else:
        #try match option string.
        #print("!!!!!!!!!!!!!!!!")
        answer = parse_option(generated)
        #print(generated)
        #print(f"Answer:{answer}")
        #print("!!!!!!!!!!!!!!!!")
        if answer == "":
            #try to match key world
            #print(answer2opt)
            for keyword in answer2opt.keys():
                if keyword in generated:
                    answer = answer2opt[keyword]
            if answer == "":
                #try to match answer field.
                pattern = r'Answer(.*?):(.*?)([A-Z])([^a-zA-Z0-9]*?)'
                matches = re.findall(pattern, generated)
                #print(generated)
                #print(matches)
                if len(matches) > 0:
                    answer = matches[-1][-2][-1]
    return answer.capitalize()
import random

def parse_response_safe(response, answer2opt):
    if "assistant" in response:
        generated = response.split("assistant")[-1][1:]
    elif "ASSISTANT" in response:
        generated = response.split("ASSISTANT")[-1][1:]
    else:
        generated = response
    #print("!!!!!!!!!!!!!!!!")
    #print(generated)
    #print("!!!!!!!!!!!!!!!!")
    answer = ""
    if len(generated) == 1:
        #Generated only a single token. Use that as an option.
        answer = generated[-1]
    else:
        #try match option string.
        #print("!!!!!!!!!!!!!!!!")
        answer = parse_option(generated)
        #print(generated)
        #print(f"Answer:{answer}")
        #print("!!!!!!!!!!!!!!!!")
        if answer == "":
            #try to match key world
            #print(answer2opt)
            for keyword in answer2opt.keys():
                if keyword in generated:
                    answer = answer2opt[keyword]
            if answer == "":
                #try to match answer field.
                pattern = r'Answer(.*?):(.*?)([A-Z])([^a-zA-Z0-9]*?)'
                matches = re.findall(pattern, generated)
                #print(generated)
                #print(matches)
                if len(matches) > 0:
                    answer = matches[-1][-2][-1]
    answer = answer.upper()
    if answer not in ["A","B","C","D","E","F","G","H"]:
        answer = random.choice(list(answer2opt.values()))
    return answer.upper()


def parse_option(response):
    print("here")
    matches = list(re.finditer(r'\(A\)|\(B\)|\(C\)|\(D\)|\(E\)|\(F\)|\(G\)|\(H\)|\(a\)|\(b\)|\(c\)|\(d\)|A\)|B\)|C\)|D\)|a\)|b\)|c\)|d\)', response))
    # Get the last match if it exists
    if matches:
        last_occurrence = matches[-1]
        #print("Last occurrence:", last_occurrence.group())
        #print("Position:", last_occurrence.start())
        #print("Context:", response[last_occurrence.start() - 200:])
        return last_occurrence.group()[1]
    else:
        #print("No matches found")
        return ""



def parse_gpt(response):
    pattern = r'"ANSWER":(.*?)"(.*?)"'
    matches = re.findall(pattern, response)
    if len(matches) > 0:
        # print(f"Match:{matches[-1]}")
        # print("___________")
        for choice in ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(a)", "(b)", "(c)", "(d)","(e)", "(f)", "(g)", "(h)", "A)", "B)", "C)", "D)","E)", "F)", "G)", "H)", "a)", "b)",
                       "c)", "d)", "e)", "f)", "g)", "h)"]:
            # print(choice, matches[-1][-1]])
            if choice in matches[-1][-1]:
                return choice[-2]  # matches[-1][-1][1]
        return matches[-1][-1]
    else:
        # print(f"No Match")
        # print("___________")
        return ""

