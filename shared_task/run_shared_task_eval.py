
import json
import numpy as np
import os

### read file
def readJSONLine(path):

    output = []
    with open(path, 'r') as f:
        for line in f:
            output.append(json.loads(line))

    return output

### evaluation script
def runEvaluation(system_predictions, golden_predictions, skip_sarcasm_micro):

    ## read in files
    golden_predictions_dict = {}
    for each_line in golden_predictions:
        golden_predictions_dict[each_line['id']] = each_line

    ## question tags
    question_tag = [i for i in golden_predictions[0]['golden_annotation'] if 'part2' in i]

    ## evaluation
    result = {}
    errors = {}
    for each_task in question_tag:

        # evaluate curr task
        curr_task = {}
        curr_task_errors = []
        TP, FP, FN = 0.0, 0.0, 0.0
        for each_line in system_predictions:
            curr_sys_pred = set([i.lower() for i in each_line['predicted_annotation'][each_task] if \
                             i != 'Not Specified' and i != 'not specified' and i != 'not_effective'])
            #             print(golden_predictions_dict[each_line['id']]['golden_annotation'][each_task])
            curr_golden_ann = [i.lower() for i in
                               golden_predictions_dict[each_line['id']]['golden_annotation'][each_task] \
                               if i != 'Not Specified' and i != 'not specified' and i != 'not_effective']
            #             print(curr_sys_pred, curr_golden_ann)
            if len(curr_golden_ann) > 0:
                for predicted_chunk in curr_sys_pred:
                    if predicted_chunk in curr_golden_ann:
                        TP += 1  # True positives are predicted spans that appear in the gold labels.
                    else:
                        FP += 1  # False positives are predicted spans that don't appear in the gold labels.
                        curr_task_errors.append(('FP', each_line['id'], predicted_chunk, curr_golden_ann))
                for gold_chunk in curr_golden_ann:
                    if gold_chunk not in curr_sys_pred:
                        FN += 1  # False negatives are gold spans that weren't in the set of spans predicted by the model.
                        curr_task_errors.append(('FN', each_line['id'], curr_sys_pred, gold_chunk))
            else:
                if len(curr_sys_pred) > 0:
                    for predicted_chunk in curr_sys_pred:
                        FP += 1  # False positives are predicted spans that don't appear in the gold labels.
                        curr_task_errors.append(('FP', each_line['id'], predicted_chunk, curr_golden_ann))

        # print
        if TP + FP == 0:
            P = 0.0
        else:
            P = TP / (TP + FP)

        if TP + FN == 0:
            R = 0.0
        else:
            R = TP / (TP + FN)

        if P + R == 0:
            F1 = 0.0
        else:
            F1 = 2.0 * P * R / (P + R)

        curr_task["F1"] = F1
        curr_task["P"] = P
        curr_task["R"] = R
        curr_task["TP"] = TP
        curr_task["FP"] = FP
        curr_task["FN"] = FN
        N = TP + FN
        curr_task["N"] = N

        # print(curr_task)
        task_name = each_task.replace('.Response', '').replace('part2-', '')
        result[task_name] = curr_task
        errors[task_name] = curr_task_errors

        # print
    #         print(each_task.replace('.Response', ''))
    #         print('P:', curr_task['P'], 'R:', curr_task['R'], 'F1:', curr_task['F1'])
    #         print('=======')

    if skip_sarcasm_micro:
        all_TP = np.sum([t_value['TP'] for t_name, t_value in result.items() if t_name != 'sarcasm'])
        all_FP = np.sum([t_value['FP'] for t_name, t_value in result.items() if t_name != 'sarcasm'])
        all_FN = np.sum([t_value['FN'] for t_name, t_value in result.items() if t_name != 'sarcasm'])
    else:
        ### calculate micro-F1
        all_TP = np.sum([i[1]['TP'] for i in result.items()])
        all_FP = np.sum([i[1]['FP'] for i in result.items()])
        all_FN = np.sum([i[1]['FN'] for i in result.items()])

    all_P = all_TP / (all_TP + all_FP)
    all_R = all_TP / (all_TP + all_FN)
    all_F1 = 2.0 * all_P * all_R / (all_P + all_R)

    ## append
    result['micro'] = {}
    result['micro']['TP'] = all_TP
    result['micro']['FP'] = all_FP
    result['micro']['FN'] = all_FN
    result['micro']['P'] = all_P
    result['micro']['R'] = all_R
    result['micro']['F1'] = all_F1
    result['micro']['N'] = all_TP + all_FN

    #     print('micro F1', all_F1)

    return result, errors


if __name__ == '__main__':

    ##### Attention: replace YOUR_TEAM_NAME with your actual team name
    ## YOUR_TEAM_NAME = 'OSU_NLP'
    input_path = '../data/' + 'HLTRI_SARCASM' +'/'
    golden_path = '../data/shared_task-test_set-eval/'
    errors_path = input_path + 'errors/'
    text_path = '../data/shared_task_test_set_final/'
    skip_sarcasm_micro = True
    if not os.path.exists(errors_path):
     os.mkdir(errors_path)

    team_name = input_path.split('/')[-2]
    print('team name:', team_name)

    ### score each category
    # category_flag = ['positive', 'negative', 'can_not_test', 'death', 'cure']
    category_flag = ['cure']

    curr_team = {}
    curr_team['team_name'] = team_name

    ## loop each category
    all_category_results = {}
    for each_category in category_flag:
        ## read in data
        curr_pred = readJSONLine(input_path + team_name + '-' + each_category + '-sarcasm.jsonl')
        curr_sol = readJSONLine(golden_path + each_category + '_sol-sarcasm.jsonl')
        err_file = errors_path + team_name + '-' + each_category + '.txt'
        tweets = readJSONLine(text_path + 'shared_task-test-' + each_category + '.jsonl')
        text_lookup = {tweet['id']: tweet['text'] for tweet in tweets}
        ## generate result
        curr_result, curr_errors = runEvaluation(curr_pred, curr_sol, skip_sarcasm_micro)
        with open(err_file, 'w') as f:
            for t, t_errs in curr_errors.items():
                for (err_type, tweet_id, predicted_chunk, golden_chunks) in t_errs:
                    f.write('-------------------\n')
                    f.write(f'Tweet ID: {tweet_id}\n')
                    text = text_lookup[tweet_id]
                    f.write(f'Tweet text: {text}\n')
                    f.write(f'Slot {t}\n')
                    f.write(f'Error type {err_type}\n')
                    f.write(f'Predicted Wrong: {predicted_chunk}\n')
                    f.write(f'Gold: {golden_chunks}\n')
                    f.write('-------------------\n')

        ## print
        t_p = curr_result["micro"]["P"]
        t_r = curr_result["micro"]["R"]
        t_f1 = curr_result["micro"]["F1"]
        print('----------------------------')
        print(f'{team_name+" "+each_category:<20}\tP: {t_p:.4f}\tR: {t_r:.4f}\tF1: {t_f1:.4f}')
        print('----------------------------')
        for t, t_results in curr_result.items():
            if t != 'micro':
                # print(f'{t:<20}\tP: {t_results["P"]:.4f}\tR: {t_results["R"]:.4f}\tF1: {t_results["F1"]:.4f}\tN: {t_results["N"]:.0f}')
                fixed_t = t.replace("_", "\\_")
                print(f'{fixed_t:<20} & {t_results["P"]*100:<5.2f} & {t_results["R"]*100:<5.2f} & {t_results["F1"]*100:<5.2f} & {t_results["N"]:<5.0f} \\\\')
        ## append result
        print('----------------------------')
        all_category_results[each_category] = curr_result

    ### overall
    all_cate_TP = np.sum([i[1]['micro']['TP'] for i in all_category_results.items()])
    all_cate_FP = np.sum([i[1]['micro']['FP'] for i in all_category_results.items()])
    all_cate_FN = np.sum([i[1]['micro']['FN'] for i in all_category_results.items()])

    # print(all_cate_TP + all_cate_FN)

    ### micro-F1
    all_cate_P = all_cate_TP / (all_cate_TP + all_cate_FP)
    all_cate_R = all_cate_TP / (all_cate_TP + all_cate_FN)
    all_cate_F1 = 2.0 * all_cate_P * all_cate_R / (all_cate_P + all_cate_R)

    curr_team['category_perf'] = all_category_results
    merged_performance = {}
    merged_performance['TP'] = all_cate_TP
    merged_performance['FP'] = all_cate_FP
    merged_performance['FN'] = all_cate_FN
    merged_performance['P'] = all_cate_P
    merged_performance['R'] = all_cate_R
    merged_performance['F1'] = all_cate_F1
    curr_team['overall_perf'] = merged_performance

    print('-----')
    print(f'{team_name + " overall":<20}\tP: {all_cate_P:.4f}\tR: {all_cate_R:.4f}\tF1: {all_cate_F1:.4f}\tN: {all_cate_TP+all_cate_FN:.0f}')
    print('======')
