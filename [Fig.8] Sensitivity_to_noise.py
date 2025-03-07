from utils import *
from matplotlib.patches import Patch

comparison_dir = os.path.join(figure_dir, '[Paper7]Comparison-explanations')
sensitivity_dir = os.path.join(figure_dir, '[Paper8,9]Sensitivity-to-noise')
os.makedirs(sensitivity_dir, exist_ok = True)

font_size = 22
plt.rcParams['axes.titlesize'] = font_size+2
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['xtick.labelsize'] = font_size-1
plt.rcParams['ytick.labelsize'] = font_size-1
plt.rcParams['legend.fontsize'] = font_size-2
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Times New Roman'

# %% Retrieve and unpickle the lists
list_of_files = [file for file in os.listdir(data_dir + f'/virtual_dataset_overall')]
lists = [pickle.load(open(data_dir + f'/virtual_dataset_overall/{file_name}', 'rb')) for file_name in list_of_files]

# Initialize lists to store the errors for all imported lists
r2s_means = []
r2s_stds = []

shapflow_abs_direct_ = []
shap_abs_direct_ = []
lime_abs_direct_ = []
ig_abs_direct_ = []

shapflow_abs_indirect_ = []
shap_abs_indirect_ = []
lime_abs_indirect_ = []
ig_abs_indirect_ = []

shapflow_abs_direct_means = []
shap_abs_direct_means = []
lime_abs_direct_means = []
ig_abs_direct_means = []

shapflow_abs_indirect_means = []
shap_abs_indirect_means = []
lime_abs_indirect_means = []
ig_abs_indirect_means = []

shapflow_abs_direct_stds = []
shap_abs_direct_stds = []
lime_abs_direct_stds = []
ig_abs_direct_stds = []

shapflow_abs_indirect_stds = []
shap_abs_indirect_stds = []
lime_abs_indirect_stds = []
ig_abs_indirect_stds = []

distances_ranks_shf_ = []
distances_ranks_shap_ = []
distances_ranks_lime_ = []
distances_ranks_ig_ = []

# Loop over each retrieved list and calculate the errors
for lst in lists:
    (r2s,
     shapflow_abs_direct, shap_abs_direct, lime_abs_direct, ig_abs_direct,
     shapflow_abs_indirect, shap_abs_indirect, lime_abs_indirect, ig_abs_indirect,
     distances_ranks_shf, distances_ranks_shap, distances_ranks_lime, distances_ranks_ig,
     ) = lst

    r2s_mean = np.mean(r2s)
    r2s_std = np.std(r2s)
    r2s_means.append(r2s_mean)
    r2s_stds.append(r2s_std)

    shapflow_abs_direct_.append(shapflow_abs_direct)
    shap_abs_direct_.append(shap_abs_direct)
    lime_abs_direct_.append(lime_abs_direct)
    ig_abs_direct_.append(ig_abs_direct)

    shapflow_abs_indirect_.append(shapflow_abs_indirect)
    shap_abs_indirect_.append(shap_abs_indirect)
    lime_abs_indirect_.append(lime_abs_indirect)
    ig_abs_indirect_.append(ig_abs_indirect)

    # Means of absolute error
    shapflow_abs_direct_mean = np.mean(shapflow_abs_direct)
    shap_abs_direct_mean = np.mean(shap_abs_direct)
    lime_abs_direct_mean = np.mean(lime_abs_direct)
    ig_abs_direct_mean = np.mean(ig_abs_direct)

    shapflow_abs_indirect_mean = np.mean(shapflow_abs_indirect)
    shap_abs_indirect_mean = np.mean(shap_abs_indirect)
    lime_abs_indirect_mean = np.mean(lime_abs_indirect)
    ig_abs_indirect_mean = np.mean(ig_abs_indirect)

    shapflow_abs_direct_means.append(shapflow_abs_direct_mean)
    shap_abs_direct_means.append(shap_abs_direct_mean)
    lime_abs_direct_means.append(lime_abs_direct_mean)
    ig_abs_direct_means.append(ig_abs_direct_mean)

    shapflow_abs_indirect_means.append(shapflow_abs_indirect_mean)
    shap_abs_indirect_means.append(shap_abs_indirect_mean)
    lime_abs_indirect_means.append(lime_abs_indirect_mean)
    ig_abs_indirect_means.append(ig_abs_indirect_mean)

    # STDs of absolute error
    shapflow_abs_direct_std = np.std(shapflow_abs_direct)
    shap_abs_direct_std = np.std(shap_abs_direct)
    lime_abs_direct_std = np.std(lime_abs_direct)
    ig_abs_direct_std = np.std(ig_abs_direct)

    shapflow_abs_indirect_std = np.std(shapflow_abs_indirect)
    shap_abs_indirect_std = np.std(shap_abs_indirect)
    lime_abs_indirect_std = np.std(lime_abs_indirect)
    ig_abs_indirect_std = np.std(ig_abs_indirect)

    shapflow_abs_direct_stds.append(shapflow_abs_direct_std)
    shap_abs_direct_stds.append(shap_abs_direct_std)

    shapflow_abs_indirect_stds.append(shapflow_abs_indirect_std)
    shap_abs_indirect_stds.append(shap_abs_indirect_std)
    lime_abs_indirect_stds.append(lime_abs_indirect_std)
    ig_abs_indirect_stds.append(ig_abs_indirect_std)

    # Distances
    distances_ranks_shf_.append(distances_ranks_shf)
    distances_ranks_shap_.append(distances_ranks_shap)
    distances_ranks_lime_.append(distances_ranks_lime)
    distances_ranks_ig_.append(distances_ranks_ig)

shapflow_abs_direct_means = np.array(shapflow_abs_direct_means)
shap_abs_direct_means = np.array(shap_abs_direct_means)
lime_abs_direct_means = np.array(lime_abs_direct_means)
ig_abs_direct_means = np.array(ig_abs_direct_means)

shapflow_abs_indirect_means = np.array(shapflow_abs_indirect_means)
shap_abs_indirect_means = np.array(shap_abs_indirect_means)
lime_abs_indirect_means = np.array(lime_abs_indirect_means)
ig_abs_indirect_means = np.array(ig_abs_indirect_means)

shapflow_abs_direct_stds = np.array(shapflow_abs_direct_stds)
shap_abs_direct_stds = np.array(shap_abs_direct_stds)
lime_abs_direct_stds = np.array(lime_abs_direct_stds)
ig_abs_direct_stds = np.array(ig_abs_direct_stds)

shapflow_abs_indirect_stds = np.array(shapflow_abs_indirect_stds)
shap_abs_indirect_stds = np.array(shap_abs_indirect_stds)
lime_abs_indirect_stds = np.array(lime_abs_indirect_stds)
ig_abs_indirect_stds = np.array(ig_abs_indirect_stds)

# noises = [0.0, 0.05, 0.1, 0.15, 0.2]
noises = [0.0, 0.1, 0.2, 0.3, 0.4]

# %% Direct errors plot
plt.figure(figsize=(9, 6))
alpha = 0.2
plt.plot(noises, r2s_means, label='$R^2$ score mean')
plt.fill_between(noises,
                 r2s_means - 1.96 * np.array(r2s_stds),
                 r2s_means + 1.96 * np.array(r2s_stds),
                 alpha=alpha )
plt.xticks(noises)
plt.xlabel('Std of noise')
plt.ylabel('$R^2$')
plt.legend()
plt.grid()
plt.tight_layout()
savename = sensitivity_dir + f'/[battery] R2 score.png'
plt.savefig(savename)
plt.show()


# %%
positions_flow = [1 + 6*i for i in range(len(noises))]  # Box positions for Shapley Flow
positions_shap = [2 + 6 * i for i in range(len(noises))]  # Box positions for SHAP
positions_lime = [3 + 6*i for i in range(len(noises))]  # Box positions for LIME
positions_ig = [4 + 6*i for i in range(len(noises))]  # Box positions for IG
c1, c2, c3, c4 = 'slateblue', 'coral', 'greenyellow', 'gold'

# %%
plt.figure(figsize = (12,10))
# Subplot 1: MAE of direct boundary according to noise
plt.subplot(3,1,1)
vp1 = plt.violinplot(shapflow_abs_direct_, positions=positions_flow, widths=0.8, showmedians=True)
for pc in vp1['bodies']:
    pc.set_facecolor(c1)
    pc.set_edgecolor('black')
vp1['cmedians'].set_color('black')
vp1['cbars'].set_color(c1)
vp1['cmaxes'].set_color(c1)
vp1['cmins'].set_color(c1)

vp2 = plt.violinplot(shap_abs_direct_, positions=positions_shap, widths=0.8, showmedians=True)
for pc in vp2['bodies']:
    pc.set_facecolor(c2)
    pc.set_edgecolor('black')
vp2['cmedians'].set_color('black')
vp2['cbars'].set_color(c2)
vp2['cmaxes'].set_color(c2)
vp2['cmins'].set_color(c2)

vp3 = plt.violinplot(lime_abs_direct_, positions=positions_lime, widths=0.8, showmedians=True)
for pc in vp3['bodies']:
    pc.set_facecolor(c3)
    pc.set_edgecolor('black')
vp3['cmedians'].set_color('black')
vp3['cbars'].set_color(c3)
vp3['cmaxes'].set_color(c3)
vp3['cmins'].set_color(c3)

vp4 = plt.violinplot(ig_abs_direct_, positions=positions_ig, widths=0.8, showmedians=True)
for pc in vp4['bodies']:
    pc.set_facecolor(c4)
    pc.set_edgecolor('black')
vp4['cmedians'].set_color('black')
vp4['cbars'].set_color(c4)
vp4['cmaxes'].set_color(c4)
vp4['cmins'].set_color(c4)

plt.xticks([2.5 + 6*i for i in range(len(noises))], noises)
plt.ylabel("MAE")
plt.ylim(-0.2,6)
plt.grid(visible = True)
plt.title('(a) \'Direct\' explanation boundary: MAE')

# Subplot 2: MAE of root cause boundary according to noise
plt.subplot(3,1,2)
vp1 = plt.violinplot(shapflow_abs_indirect_, positions=positions_flow, widths=0.8, showmedians=True)
for pc in vp1['bodies']:
    pc.set_facecolor(c1)
    pc.set_edgecolor('black')
vp1['cmedians'].set_color('black')
vp1['cbars'].set_color(c1)
vp1['cmaxes'].set_color(c1)
vp1['cmins'].set_color(c1)

vp2 = plt.violinplot(shap_abs_indirect_, positions=positions_shap, widths=0.8, showmedians=True)
for pc in vp2['bodies']:
    pc.set_facecolor(c2)
    pc.set_edgecolor('black')
vp2['cmedians'].set_color('black')
vp2['cbars'].set_color(c2)
vp2['cmaxes'].set_color(c2)
vp2['cmins'].set_color(c2)

vp3 = plt.violinplot(lime_abs_indirect_, positions=positions_lime, widths=0.8, showmedians=True)
for pc in vp3['bodies']:
    pc.set_facecolor(c3)
    pc.set_edgecolor('black')
vp3['cmedians'].set_color('black')
vp3['cbars'].set_color(c3)
vp3['cmaxes'].set_color(c3)
vp3['cmins'].set_color(c3)

vp4 = plt.violinplot(ig_abs_indirect_, positions=positions_ig, widths=0.8, showmedians=True)
for pc in vp4['bodies']:
    pc.set_facecolor(c4)
    pc.set_edgecolor('black')
vp4['cmedians'].set_color('black')
vp4['cbars'].set_color(c4)
vp4['cmaxes'].set_color(c4)
vp4['cmins'].set_color(c4)

plt.xticks([2.5 + 6*i for i in range(len(noises))], noises)
plt.ylabel("MAE")
plt.ylim(-0.2,6)
plt.grid(visible = True)
plt.title('(b) \'Root cause\' explanation boundary: MAE')

# Subplot 3: KT coefficient of direct boundary according to noise
plt.subplot(3,1,3)
vp1 = plt.violinplot(distances_ranks_shf_, positions=positions_flow, widths=0.8, showmedians=True)
for pc in vp1['bodies']:
    pc.set_facecolor(c1)
    pc.set_edgecolor('black')
vp1['cmedians'].set_color('black')
vp1['cbars'].set_color(c1)
vp1['cmaxes'].set_color(c1)
vp1['cmins'].set_color(c1)

vp2 = plt.violinplot(distances_ranks_shap_, positions=positions_shap, widths=0.8, showmedians=True)
for pc in vp2['bodies']:
    pc.set_facecolor(c2)
    pc.set_edgecolor('black')
vp2['cmedians'].set_color('black')
vp2['cbars'].set_color(c2)
vp2['cmaxes'].set_color(c2)
vp2['cmins'].set_color(c2)

vp3 = plt.violinplot(distances_ranks_lime_, positions=positions_lime, widths=0.8, showmedians=True)
for pc in vp3['bodies']:
    pc.set_facecolor(c3)
    pc.set_edgecolor('black')
vp3['cmedians'].set_color('black')
vp3['cbars'].set_color(c3)
vp3['cmaxes'].set_color(c3)
vp3['cmins'].set_color(c3)

vp4 = plt.violinplot(distances_ranks_ig_, positions=positions_ig, widths=0.8, showmedians=True)
for pc in vp4['bodies']:
    pc.set_facecolor(c4)
    pc.set_edgecolor('black')
vp4['cmedians'].set_color('black')
vp4['cbars'].set_color(c4)
vp4['cmaxes'].set_color(c4)
vp4['cmins'].set_color(c4)

plt.xticks([2.5 + 6*i for i in range(len(noises))], noises)
plt.xlabel("Std of noise")
plt.ylabel("Kendall-tau correlation")
plt.grid(visible = True)
plt.title('(c) \'Root cause\' explanation boundary: Kendall-tau')

shapflow_patch = Patch(color=c1, label='Shapley Flow')
shap_patch = Patch(color=c2, label='SHAP')
lime_patch = Patch(color=c3, label='LIME')
ig_patch = Patch(color=c4, label='IG')

plt.figlegend(handles=[shapflow_patch, shap_patch, lime_patch, ig_patch],
              loc='lower center',
              ncol=4,
              fontsize=24,
              bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.05, 1, 1])
savename = sensitivity_dir + f'/[battery] Robustness_to_noise.png'
plt.savefig(savename)
plt.show()
