import argparse
import os
import random

def main(args):
    dirs = [file_ for file_ in os.listdir(args.test_cases_path) \
        if os.path.isdir(os.path.join(args.test_cases_path, file_)) and file_ not in ['.', ".."]]
    for i, dir in enumerate(dirs):
        print("{}- {}".format(i, dir))
    #example input would: 0, 20
    case_ix_num = input("Specify the case number, and number of cases you would like to inspect from the shown list:")
    case_ix, case_num = tuple(int(x) for x in case_ix_num.split())
    cases_path = os.path.join(args.test_cases_path, dirs[case_ix])
    files = [x for x in os.listdir(cases_path) if x not in [',', ",,", "prob_cases"]]
    effect_exists = 0
    case_num = min(len(files), case_num)
    # print(files)
    for i, file_ in enumerate(random.sample(files, k=case_num)):
        try:
            print("Case number: {}, File Name: {}".format(i + 1, file_))
            adverse_file = open(os.path.join(cases_path, file_))
            orgnl_conv, mod_conv = adverse_file.read().split("modified_conversation")
            print("The original conversation:")
            print("--------------------------")
            print(orgnl_conv.split("original_conversation")[1].strip())
            print("The modified conversation:")
            print("--------------------------")
            print(mod_conv.strip())
            print("##########################")
            recvd_inp = input("input 1 if effect exists else 0, type exit to stop:")
            if recvd_inp == "exit":
                i -= 1
                break
            effect_exists += int(recvd_inp)
        except Exception as e:
            print("This is the file which caused the error: {}".format(file_))
            
    total_cases = min(case_num, i + 1)
    if total_cases:
        print("Proportion of cases which have the desired effect:", effect_exists/min(case_num, i + 1))
    else:
        print("No cases examinded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_cases_path", help="Path to the folder containing all the test cases.", default="./test_conversations/")
    args = parser.parse_args()

    main(args)