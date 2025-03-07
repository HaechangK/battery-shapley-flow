from utils import *
from Virtual_dataset import *
from matplotlib.patches import Patch

comparison_dir = os.path.join(figure_dir, '[Paper7]Comparison-explanations')
os.makedirs(comparison_dir, exist_ok=True)

font_size = 22
plt.rcParams['axes.titlesize'] = font_size+2
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['xtick.labelsize'] = font_size-3
plt.rcParams['ytick.labelsize'] = font_size-2
plt.rcParams['legend.fontsize'] = font_size-2
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "dejavuserif"

# %% Main arguments
TARGET = 'Virtual - battery'
NOISE = 0.4

# %% Retrieve and unpickle the lists
list_of_files = [file for file in os.listdir(data_dir + f'/{TARGET}_{NOISE}')]
list_of_files = [file for file in list_of_files if 'DS_Store' not in file]
lists = [pickle.load(open(data_dir + f'/{TARGET}_{NOISE}/{file_name}', 'rb')) for file_name in list_of_files]

# Initialize lists to store the errors for all imported lists
r2s = []
errors_shapflow_direct = []
errors_shap_direct = []
errors_lime_direct = []
errors_ig_direct = []

errors_shapflow_indirect = []
errors_shap_indirect = []
errors_lime_indirect = []
errors_ig_indirect = []

distances_ranks_shf = []
distances_ranks_shap = []
distances_ranks_lime = []
distances_ranks_ig = []

# Loop over each retrieved list and calculate the errors
for lst in lists:
    (features, r2, gt_direct, gt_indirect, shf_direct_norm, shf_indirect_norm,
     shap_values_mean_norm, lime_values_mean_norm, ig_values_mean_norm) = lst
    gt_direct, gt_indirect = np.array(gt_direct), np.array(gt_indirect)

    def normalize(array):
        return array / array.sum()

    gt_direct = normalize(gt_direct)
    gt_indirect = normalize(gt_indirect)
    shf_direct_norm = normalize(shf_direct_norm)
    shf_indirect_norm = normalize(shf_indirect_norm)
    shap_values_mean_norm = normalize(shap_values_mean_norm)
    lime_values_mean_norm = normalize(lime_values_mean_norm)
    ig_values_mean_norm = normalize(ig_values_mean_norm)

    r2s.append(r2)

    # Direct errors
    errors_shapflow_direct.append(shf_direct_norm - gt_direct)
    errors_shap_direct.append(shap_values_mean_norm - gt_direct)
    errors_lime_direct.append(lime_values_mean_norm - gt_direct)
    errors_ig_direct.append(ig_values_mean_norm - gt_direct)

    # Indirect errors
    errors_shapflow_indirect.append(shf_indirect_norm - gt_indirect)
    errors_shap_indirect.append(shap_values_mean_norm - gt_indirect)
    errors_lime_indirect.append(lime_values_mean_norm - gt_indirect)
    errors_ig_indirect.append(ig_values_mean_norm - gt_indirect)

    distance_ranks_shf, distance_ranks_shap, distance_ranks_lime, distance_ranks_ig,  = (
        compare_rank(features, gt_indirect,
                     shf_indirect_norm,
                     shap_values_mean_norm,
                     lime_values_mean_norm,
                     ig_values_mean_norm,
                     dist = 'kendalltau'))

    # Distances
    distances_ranks_shf.append(distance_ranks_shf)
    distances_ranks_shap.append(distance_ranks_shap)
    distances_ranks_lime.append(distance_ranks_lime)
    distances_ranks_ig.append(distance_ranks_ig)

# Calculate the mean and std of the errors - direct
mean_error_shapflow_direct = np.mean(errors_shapflow_direct, axis=0)
std_error_shapflow_direct = np.std(errors_shapflow_direct, axis=0)

mean_error_shap_direct = np.mean(errors_shap_direct, axis=0)
std_error_shap_direct = np.std(errors_shap_direct, axis=0)

mean_error_lime_direct = np.mean(errors_lime_direct, axis=0)
std_error_lime_direct = np.std(errors_lime_direct, axis=0)

mean_error_ig_direct = np.mean(errors_ig_direct, axis=0)
std_error_ig_direct = np.std(errors_ig_direct, axis=0)

# Calculate the mean and std of the errors - indirect
mean_error_shapflow_indirect = np.mean(errors_shapflow_indirect, axis=0)
std_error_shapflow_indirect = np.std(errors_shapflow_indirect, axis=0)

mean_error_shap_indirect = np.mean(errors_shap_indirect, axis=0)
std_error_shap_indirect = np.std(errors_shap_indirect, axis=0)

mean_error_lime_indirect = np.mean(errors_lime_indirect, axis=0)
std_error_lime_indirect = np.std(errors_lime_indirect, axis=0)

mean_error_ig_indirect = np.mean(errors_ig_indirect, axis=0)
std_error_ig_indirect = np.std(errors_ig_indirect, axis=0)

features = ['$PP_{An1}$', '$PP_{An2}$', '$PP_{Cat1}$', '$PP_{Cat2}$',
            '$PP_{B1}$', '$PP_{B2}$', '$PP_{C1}$', '$PP_{C2}$', '$PP_{D1}$', '$PP_{D2}$',
            '$IPF_{An1}$', '$IPF_{Cat1}$', '$IPF_{B1}$', '$IPF_{C1}$']


# %%
errors_shapflow_direct = pd.DataFrame(errors_shapflow_direct)
errors_shap_direct = pd.DataFrame(errors_shap_direct)
errors_lime_direct = pd.DataFrame(errors_lime_direct)
errors_ig_direct = pd.DataFrame(errors_ig_direct)

errors_shapflow_indirect = pd.DataFrame(errors_shapflow_indirect)
errors_shap_indirect = pd.DataFrame(errors_shap_indirect)
errors_lime_indirect = pd.DataFrame(errors_lime_indirect)
errors_ig_indirect = pd.DataFrame(errors_ig_indirect)

# Plot configuration
num_features = errors_shap_direct.shape[1]
positions_flow = [1 + 6*i for i in range(num_features)]  # Box positions for Shapley Flow
positions_shap = [2 + 6*i for i in range(num_features)]  # Box positions for SHAP
positions_lime = [3 + 6*i for i in range(num_features)]  # Box positions for LIME
positions_ig = [4 + 6*i for i in range(num_features)]  # Box positions for IG


# %%
# Figure 1: Boxplot for Direct Errors
c1, c2, c3, c4 = 'slateblue', 'coral', 'greenyellow', 'gold'

plt.figure(figsize=(20, 10))
plt.subplot(2,1,1)
plt.boxplot(errors_shapflow_direct, positions=positions_flow, widths=0.6, patch_artist=True,
            boxprops=dict(facecolor=c1),
            medianprops = dict(color = 'black'))
plt.boxplot(errors_shap_direct, positions=positions_shap, widths=0.6, patch_artist=True,
            boxprops=dict(facecolor=c2),
            medianprops = dict(color = 'black'))
plt.boxplot(errors_lime_direct, positions=positions_lime, widths=0.6, patch_artist=True,
            boxprops=dict(facecolor=c3),
            medianprops = dict(color = 'black'))
plt.boxplot(errors_ig_direct, positions=positions_ig, widths=0.6, patch_artist=True,
            boxprops=dict(facecolor=c4),
            medianprops = dict(color = 'black'))
plt.hlines(0, xmin = 0, xmax = positions_ig[-1])
plt.xticks([2.5 + 6*i for i in range(num_features)], features)
plt.title('Explanation boundary: \'Direct\'')
plt.ylabel('Mean error')
plt.grid(visible = True)
shapflow_patch = Patch(color=c1, label='Shapley Flow')
shap_patch = Patch(color=c2, label='SHAP')
lime_patch = Patch(color=c3, label='LIME')
ig_patch = Patch(color=c4, label='IG')
plt.legend(handles=[shapflow_patch, shap_patch, lime_patch, ig_patch],
           # loc='lower left'
           )

# Figure 2: Boxplot for Indirect Errors
plt.subplot(2,1,2)
plt.boxplot(errors_shapflow_indirect, positions=positions_flow, widths=0.6, patch_artist=True,
            boxprops=dict(facecolor=c1),
            medianprops = dict(color = 'black'))
plt.boxplot(errors_shap_indirect, positions=positions_shap, widths=0.6, patch_artist=True,
            boxprops=dict(facecolor=c2),
            medianprops = dict(color = 'black'))
plt.boxplot(errors_lime_indirect, positions=positions_lime, widths=0.6, patch_artist=True,
            boxprops=dict(facecolor=c3),
            medianprops = dict(color = 'black'))
plt.boxplot(errors_ig_indirect, positions=positions_ig, widths=0.6, patch_artist=True,
            boxprops=dict(facecolor=c4),
            medianprops = dict(color = 'black'))
plt.hlines(0, xmin = 0, xmax = positions_ig[-1])
plt.xticks([2.5 + 6*i for i in range(num_features)], features)
plt.title('Explanation boundary: \'Root cause\'')
# plt.xlabel('Features')
plt.ylabel('Mean error')
plt.grid(visible = True)
shapflow_patch = Patch(color=c1, label='Shapley Flow')
shap_patch = Patch(color=c2, label='SHAP')
lime_patch = Patch(color=c3, label='LIME')
ig_patch = Patch(color=c4, label='IG')
plt.legend(handles=[shapflow_patch, shap_patch, lime_patch, ig_patch],
           # loc='lower left'
           )

plt.tight_layout()
savename = comparison_dir + f'/[battery][{NOISE}] Box_{len(list_of_files)} experiments.png'
plt.savefig(savename)
plt.show()

# %% Violin plot of rank
plt.figure(figsize = (10,7))
vp1 = plt.violinplot(distances_ranks_shf, positions=[1], widths=1.0, showmedians=True)
for pc in vp1['bodies']:
    pc.set_facecolor(c1)
    pc.set_edgecolor('black')
vp1['cmedians'].set_color('black')
vp1['cbars'].set_color(c1)
vp1['cmaxes'].set_color(c1)
vp1['cmins'].set_color(c1)

vp2 = plt.violinplot(distances_ranks_shap, positions=[2], widths=1.0, showmedians=True)
for pc in vp2['bodies']:
    pc.set_facecolor(c2)
    pc.set_edgecolor('black')
vp2['cmedians'].set_color('black')
vp2['cbars'].set_color(c2)
vp2['cmaxes'].set_color(c2)
vp2['cmins'].set_color(c2)

vp3 = plt.violinplot(distances_ranks_lime, positions=[3], widths=1.0, showmedians=True)
for pc in vp3['bodies']:
    pc.set_facecolor(c3)
    pc.set_edgecolor('black')
vp3['cmedians'].set_color('black')
vp3['cbars'].set_color(c3)
vp3['cmaxes'].set_color(c3)
vp3['cmins'].set_color(c3)

vp4 = plt.violinplot(distances_ranks_ig, positions=[4], widths=1.0, showmedians=True)
for pc in vp4['bodies']:
    pc.set_facecolor(c4)
    pc.set_edgecolor('black')
vp4['cmedians'].set_color('black')
vp4['cbars'].set_color(c4)
vp4['cmaxes'].set_color(c4)
vp4['cmins'].set_color(c4)

shapflow_patch = Patch(color=c1, label='Shapley Flow')
shap_patch = Patch(color=c2, label='SHAP')
lime_patch = Patch(color=c3, label='LIME')
ig_patch = Patch(color=c4, label='IG')
plt.legend(handles=[shapflow_patch, shap_patch, lime_patch, ig_patch], loc='lower left')

plt.xticks([1,2,3,4], labels = ['Shapley flow', 'SHAP', 'LIME', 'IG'])
plt.xlabel("Methods")
plt.ylabel("Kendall-tau correlation")
plt.grid(visible = True)
plt.title('Explanation boundary: \'Root cause\'')
plt.tight_layout()
savename = comparison_dir + f'/[battery][{NOISE}] Distances based on ranks.png'
plt.savefig(savename)
plt.show()

# %%
shapflow_abs_direct = np.absolute(errors_shapflow_direct.values).sum(axis = 0)
shap_abs_direct = np.absolute(errors_shap_direct.values).sum(axis = 0)
lime_abs_direct = np.absolute(errors_lime_direct.values).sum(axis = 0)
ig_abs_direct = np.absolute(errors_ig_direct.values).sum(axis = 0)

shapflow_abs_indirect = np.absolute(errors_shapflow_indirect.values).sum(axis = 0)
shap_abs_indirect = np.absolute(errors_shap_indirect.values).sum(axis = 0)
lime_abs_indirect = np.absolute(errors_lime_indirect.values).sum(axis = 0)
ig_abs_indirect = np.absolute(errors_ig_indirect.values).sum(axis = 0)

print("Errors:")
print("[Direct] SHAPflow    : {:.3f}".format(shapflow_abs_direct.mean()))
print("[Direct] SHAP        : {:.3f}".format(shap_abs_direct.mean()))
print("[Direct] LIME        : {:.3f}".format(lime_abs_direct.mean()))
print("[Direct] IG          : {:.3f}".format(ig_abs_direct.mean()))

print("[Indirect] SHAPflow  : {:.3f}".format(shapflow_abs_indirect.mean()))
print("[Indirect] SHAP      : {:.3f}".format(shap_abs_indirect.mean()))
print("[Indirect] LIME      : {:.3f}".format(lime_abs_indirect.mean()))
print("[Indirect] IG      : {:.3f}".format(ig_abs_indirect.mean()))

# %%
results = [r2s,
           shapflow_abs_direct, shap_abs_direct, lime_abs_direct, ig_abs_direct,
           shapflow_abs_indirect, shap_abs_indirect, lime_abs_indirect, ig_abs_indirect,
           distances_ranks_shf, distances_ranks_shap, distances_ranks_lime, distances_ranks_ig,
           ]

with open(data_dir + f'/virtual_dataset_overall/[battery][{NOISE}] Comparison_given_noise.pickle', 'wb') as f:
    pickle.dump(results, f)
