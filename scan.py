import json
import subprocess

with open('params_globalpool.json') as f:
    params = json.load(f)

win_height = 7
for sec_num in range(7, 11):
    sec_size = 35 // sec_num + 1
    for inh_radius in range(1, 2):
        for threshold in range(22, 25):
            params['layers'][1]['win'][0] = win_height
            params['layers'][1]['sec_num'] = sec_num
            params['layers'][1]['sec_size'] = sec_size
            params['layers'][1]['inh_radius'] = inh_radius
            params['layers'][1]['threshold'] = threshold

            with open('params_globalpool.json', 'w') as f:
                json.dump(params, f)

            subprocess.run(['python', '-u', 'main.py', 'params_globalpool.json', '--noprogress'])
            print('win_height =', win_height, 'sec_num =', sec_num, 'sec_size =', sec_size, 'inh_radius =', inh_radius, 'threshold =', threshold)
            print()
