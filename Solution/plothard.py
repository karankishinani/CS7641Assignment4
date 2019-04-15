'''
Submitted on Apr 14, 2018
Adapted from https://github.com/JonathanTay/CS-7641-assignment-4
@author: Karan Kishinani
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

outdir = 'out/plots/hard/'

lr_test = True
Q_test = True

vi_hard_f = 'out/hard/initial/Hard Value.csv'
pi_hard_f = 'out/hard/initial/Hard Policy.csv'
best_Q = 'out/hard/initial/Q/Hard Q-Learning L0.01 q0.0 E0.1.csv'

evi = pd.read_csv(vi_hard_f)
epi = pd.read_csv(pi_hard_f)
eQ = pd.read_csv(best_Q)

algos = {'VI': evi, 'PI': epi,'Q': eQ}

plt.title('HardGW Convergence over Iterations')

for key, val in algos.items():
    print (key)
    df = algos[key]
    plt.plot(df['iter'], df['convergence'], label=key)
plt.legend()
plt.xlim([0,100])
plt.ylabel('Convergence Delta')
plt.xlabel('Iterations')
plt.savefig(outdir + 'hardgw_convergence.png')
plt.close()

plt.title('HardGW Convergence over Iterations (Zoomed)')

for key, val in algos.items():
    print (key)
    df = algos[key]
    plt.plot(df['iter'], df['convergence'], label=key)
plt.legend()
plt.ylim([-1,20])
plt.xlim([0,80])
plt.ylabel('Convergence Delta')
plt.xlabel('Iterations')
plt.savefig(outdir + 'hardgw_convergence_zoomed.png')
plt.close()

plt.title('HardGW Time over Iterations')

for key, val in algos.items():
    print (key)
    df = algos[key]
    plt.plot(df['iter'], df['time'], label=key)
plt.legend()
plt.xlim([0,100])
plt.ylabel('Wall Time')
plt.xlabel('Iterations')
plt.savefig(outdir + 'hardgw_time.png')
plt.close()

plt.title('HardGW Time over Iterations Full')

for key, val in algos.items():
    print (key)
    df = algos[key]
    plt.plot(df['iter'], df['time'], label=key)
plt.legend()
#plt.xlim([-50,3800])
plt.ylabel('Wall Time')
plt.xlabel('Iterations')
plt.savefig(outdir + 'hardgw_time_full.png')
plt.close()

if lr_test:

    lr = 'L0.01'

    eQ_files = []
    for subdir, dirs, files in os.walk('out/hard/initial/Q/'):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".csv") and lr in filepath:
                eQ_files.append(filepath)

    f = best_Q
    eQ = pd.read_csv(f)
    lab = f.upper()[35:-4]
    plt.plot(eQ['iter'], eQ['convergence'], label=lab)

    '''
    plt.title('HardGW Convergence over Iterations - Learning Rate of ' + lr)
    plt.ylim([0, 20])
    plt.ylabel('Convergence Delta')
    plt.xlabel('Iterations')
    plt.legend(loc='best')
    plt.savefig(outdir + 'lr0.01.png')
    plt.close()
    '''
    convergence = np.array([])
    lr = 'L0.1'

    eQ_files = []
    for subdir, dirs, files in os.walk('out/hard/initial/Q/'):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".csv") and lr in filepath:
                eQ_files.append(filepath)

    for f in eQ_files:
        eQ = pd.read_csv(f)
        # lab = f.upper()[41:-4]
        convergence = np.append(convergence, eQ['convergence'].min())

    f = eQ_files[np.argmin(convergence)]
    eQ = pd.read_csv(f)
    lab = f.upper()[35:-4]
    plt.plot(eQ['iter'], eQ['convergence'], label=lab)

    convergence = np.array([])
    lr = 'L0.9'

    eQ_files = []
    for subdir, dirs, files in os.walk('out/hard/initial/Q/'):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".csv") and lr in filepath:
                eQ_files.append(filepath)

    for f in eQ_files:
        eQ = pd.read_csv(f)
        # lab = f.upper()[41:-4]
        convergence = np.append(convergence, eQ['convergence'].min())

    f = eQ_files[np.argmin(convergence)]
    eQ = pd.read_csv(f)
    lab = f.upper()[35:-4]
    plt.plot(eQ['iter'], eQ['convergence'], label=lab)

    plt.title('HardGW Convergence over Iterations - Learning Rate')
    plt.legend(loc='best')
    plt.ylabel('Convergence Delta')
    plt.xlabel('Iterations')
    plt.savefig(outdir + 'lr.png')
    plt.close()


if Q_test:
    lr = 'L0.01'
    initQ = 'q0.0'

    eQ_files = []
    for subdir, dirs, files in os.walk('out/hard/initial/Q/'):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".csv") and lr in filepath and initQ in filepath:
                eQ_files.append(filepath)

    for f in eQ_files:

        eQ = pd.read_csv(f)
        lab = f.upper()[41:-4]
        plt.plot(eQ['iter'], eQ['convergence'], label=lab)

    plt.title('HardGW Convergence over Iterations - inital Q of ' + initQ)
    plt.ylim([0, 12])
    plt.legend(loc='best')
    plt.ylabel('Convergence Delta')
    plt.xlabel('Iterations')
    plt.savefig(outdir + 'q0.png')
    plt.close()

    lr = 'L0.01'
    initQ = 'q-100'

    eQ_files = []
    for subdir, dirs, files in os.walk('out/hard/initial/Q/'):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".csv") and lr in filepath and initQ in filepath:
                eQ_files.append(filepath)

    for f in eQ_files:

        eQ = pd.read_csv(f)
        lab = f.upper()[41:-4]
        plt.plot(eQ['iter'], eQ['convergence'], label=lab)

    plt.title('HardGW Convergence over Iterations - inital Q of ' + initQ)
    plt.ylim([0, 12])
    plt.legend(loc='best')
    plt.ylabel('Convergence Delta')
    plt.xlabel('Iterations')
    plt.savefig(outdir + 'q-100.png')
    plt.close()

    lr = 'L0.01'
    initQ = 'q100'

    eQ_files = []
    for subdir, dirs, files in os.walk('out/hard/initial/Q/'):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".csv") and lr in filepath and initQ in filepath:
                eQ_files.append(filepath)

    for f in eQ_files:

        eQ = pd.read_csv(f)
        lab = f.upper()[41:-4]
        plt.plot(eQ['iter'], eQ['convergence'], label=lab)

    plt.title('HardGW Convergence over Iterations - inital Q of ' + initQ)
    plt.ylim([0, 12])
    plt.legend(loc='best')
    plt.ylabel('Convergence Delta')
    plt.xlabel('Iterations')
    plt.savefig(outdir + 'q100.png')
    plt.close()

lr = 'L0.01'

eQ_files = []
for subdir, dirs, files in os.walk('out/hard/initial/Q/'):
    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith(".csv") and lr in filepath:
            eQ_files.append(filepath)

for f in eQ_files:

    eQ = pd.read_csv(f)
    lab = f.upper()[41:-4]
    print (eQ['convergence'].min())
    if (eQ['convergence'].min() < 1.6):
        plt.plot(eQ['iter'], eQ['convergence'], label=lab)

plt.title('HardGW Convergence over Iterations - Learning Rate of ' + lr)
plt.ylim([0, 20])
plt.legend(loc='best')
plt.ylabel('Convergence Delta')
plt.xlabel('Iterations')
plt.savefig(outdir + 'best_lr_summary.png')
plt.close()