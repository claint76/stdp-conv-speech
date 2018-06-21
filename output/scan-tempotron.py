import subprocess


learning_rate = 0.01
while learning_rate < 0.09:
    for round_num in range(3,6):
        print(learning_rate, round_num)
        subprocess.call(['python', '-u', 'tempotron-classifier.py', str(learning_rate), str(round_num)])
    learning_rate += 0.01
