import os, argparse
from multiprocessing import Process

def job(dataid):
    print('process PID: {}'.format(os.getpid()))
    job_command = 'CUDA_VISIBLE_DEVICES=0 python3 stable-baselines3/main/train_imitate_pour.py ' \
                  '--objscale 1.5 '\
                  '--exp pour '\
                  '--reward 0 0.1 0.02 '\
                  '--dataid {} '\
                  '--iter 2000'.format(dataid)
    os.system(job_command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataid', type=int, default=0)
    parser.add_argument('--p', type=int, default=4)

    args = parser.parse_args()
    data_id = args.dataid
    proc_num = args.p
    proc = []

    for i in range(proc_num):
        proc.append(Process(target=job, args=(data_id+i,)))
    
    for i in range(proc_num):
        proc[i].start()
    
    for i in range(proc_num):
        proc[i].join()

    print('finished.')