import os
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import pandas as pd

n_row = 2900
pd_seed_gamma_alg = []
fig_path = '../plot/figures'

# log_paths = ['../logs/workflow_scheduling-v2/20220318163834295900',
#              '../logs/workflow_scheduling-v2/20220318163846200263',
#              '../logs/workflow_scheduling-v2/20220309165550153085']

log_paths = ['../logs/workflow_scheduling-v2/20220318163846200263',
             '../logs/workflow_scheduling-v2/20220321174939748956']
# name_list = ['ES-gamma-2.5-lr-0.05-sigma-0.05', 'ES-gamma-2.5-lr-0.01-sigma-0.05', 'ES-gamma-2.5-lr-0.001-sigma-0.01']
name_list = ['ES-gamma-2.5-lr-0.01-sigma-0.05', 'ES-gamma-2.5-lr-0.001-sigma-0.1']

all_csv_files = []
EXT = "training_record.csv"

for log_path in log_paths:
    all_csv_files += [file
                      for path, subdir, files in os.walk(log_path)
                      for file in glob(os.path.join(path, EXT))]

print(all_csv_files)

for name, filename in zip(name_list, all_csv_files):
    if os.path.exists(filename):
        df = pd.read_csv(filename, nrows=n_row)
        df.columns = ['policy_id', 'rewards', 'VM_execHour', 'VM_totHour', 'VM_cost', 'SLA_penalty', 'missDeadlineNum']
        # df['params'] = f"$\\run$_{alphas}"
        df['method'] = name
        df['source'] = filename
        df['generation'] = df.index + 1
        pd_seed_gamma_alg.append(df)
    else:
        print('missing alg, alpha, sigma, gamma, seed')

data_combined = pd.concat(pd_seed_gamma_alg, ignore_index=True)
data_combined['rewards'] = data_combined['rewards'].abs()


# draw figs for one method with different gamma
def draw_figs(data):
    plt.figure()
    # label_size = 15
    # matplotlib.rcParams['xtick.labelsize'] = label_size
    # matplotlib.rcParams['ytick.labelsize'] = label_size
    # matplotlib.rcParams.update({'font.size': label_size, 'legend.fontsize': label_size})
    # data['gamma'].astype(str)
    sns.lineplot(data=data, x='generation', y='rewards', hue='method', palette=sns.color_palette("Set1", 2))
    plt.ylim(0, 200)
    # plt.show()
    plt.savefig(os.path.join(fig_path, 'fitness settings'))


draw_figs(data_combined)

# def draw_figs_compared(x, y, data, g):
#     plt.figure()
#     # label_size = 15
#     # matplotlib.rcParams['xtick.labelsize'] = label_size
#     # matplotlib.rcParams['ytick.labelsize'] = label_size
#     # matplotlib.rcParams.update({'font.size': label_size, 'legend.fontsize': label_size})
#     data['gamma'].astype(str)
#     sns.lineplot(data=data, x=x, y=y, hue='params')  # palette=sns.color_palette("Set1", 1)
#     plt.ylim(0, 200)
#     # plt.ylim(-90, -25)
#     # plt.xlim(3500, 5000)
#     plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
#     plt.title(r'$\gamma$' + "=%s with diff seeds" % g)
#     plt.tight_layout()
#     plt.savefig("%s_gamma_%s.pdf" % (y, g))

# data_combined = read_CSV()

# draw_figs(data_combined)

# for g in gammas:
#     draw_figs_compared('generation', 'rewards', data_combined[data_combined['gamma'] == g], g)
#     # draw_figs_compared('generation', 'VM_cost', data_combined[data_combined['gamma'] == g], g)
