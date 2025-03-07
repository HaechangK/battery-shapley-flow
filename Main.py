import time
from utils import *
from datetime import datetime
from Virtual_dataset import *
now = time.time()

# %% Basic arguments
BENCHMARK = False
SHAPFLOW_INTERPRET = True
TARGET = 'Virtual - battery'

# Arguments for generating synthetic dataset
NUM_DATA = 1000
NOISE = 0.4
DROPOUT = 0.0

# %% Random seed for reproducibility
np.random.seed(21)
seeds = np.random.randint(10000, size=20)
seed = seeds[0]

# %%
# if __name__ == '__main__':
for seed in seeds:
    print("Number of synthetic data     : ", NUM_DATA)
    print("Benchmark with other methods : ", BENCHMARK)
    print("Task                         : ", f'{TARGET} with noise {NOISE} and dropout {DROPOUT}')

    # %%
    """
    1. Generate virtual dataset
    """
    X, y, gt_direct, gt_indirect = generate_dataset(num_data=NUM_DATA, sigma=NOISE, seed=seed)

    # %%
    """
    2. Interpret with baseline XAI approaches.
    Three benchmark algorithms are used: SHAP, LIME, Integrated gradients.
    XGBoost models are used and trained for SHAP and LIME, while torch DNN is used for Integrated gradients.
    """
    if BENCHMARK:
        from XAI.SHAP import SHAP
        SH = SHAP(X, y, target = TARGET, interventional = False)
        shap_values = SH(load_data = False)
        # SH.SHAP_plot(visuals=[])

        from XAI.LIME import LIME
        lime = LIME(X, y, model = None)
        lime_values = lime.explain()

        from XAI.IntegratedGradients import IntegratedGradientsExplainer
        explainer = IntegratedGradientsExplainer(X, y,
                                                 n_bg = 100,
                                                 model = None)
        ig_values = explainer.explain()

    # %%
    """
    3. Interpret with Shapley flow.
    XGBoost models are used and trained in this process.
    """
    if SHAPFLOW_INTERPRET:
        from XAI.Shapflow import SHAPflow
        SHF = SHAPflow(X, y,
                       n_bg = 100,
                       nsamples = 100,
                       nruns = 1,
                       target = TARGET
                       )
        SHF.build_graph()

        edge_credit = SHF.Graph_explain()
        importance_matrix = SHF.importance_matrix(edge_credit)
        SHF.draw_graph(max_display = 20)

        # %%
        rank_boundaries = [0, -1]
        boundary_importances = []
        for rank_boundary in rank_boundaries:
            boundary_importance = SHF.explanation_boundary(rank_boundary = rank_boundary)
            boundary_importances.append(boundary_importance)
            SHF.SHAPFLOW_plot(boundary_importance,
                              visuals = [])

    # %%
    """
    4. Comparing two explanations
    """
    if BENCHMARK and SHAPFLOW_INTERPRET:
        features = SHF.nodenames[:-1]
        compare_explanation(features, gt_direct, gt_indirect, boundary_importances,
                            shap_values, lime_values, ig_values, config="raw")
        compare_explanation(features, gt_direct, gt_indirect, boundary_importances,
                            shap_values, lime_values, ig_values, config="error")

        gt_direct_norm = gt_direct / sum(gt_direct)
        gt_indirect_norm = gt_indirect / sum(gt_indirect)

        shap_values_mean = np.absolute(shap_values).mean(axis=0)
        shap_values_mean_norm = shap_values_mean / shap_values_mean.sum()
        lime_values_mean = np.absolute(lime_values).mean(axis=0)
        lime_values_mean_norm = lime_values_mean / lime_values_mean.sum()
        ig_values_mean = np.absolute(ig_values).mean(axis=0)
        ig_values_mean_norm = ig_values_mean / ig_values_mean.sum()

        shf_direct = np.absolute(boundary_importances[1]).mean(axis=0)
        shf_direct_norm = shf_direct / shf_direct.sum()
        shf_indirect = np.absolute(boundary_importances[0]).mean(axis=0)
        shf_indirect_norm = shf_indirect / shf_indirect.sum()

        results = [features, SHF.r2mean, gt_direct, gt_indirect, shf_direct_norm, shf_indirect_norm,
                   shap_values_mean_norm, lime_values_mean_norm, ig_values_mean_norm]
        virtual_dataset_dir = os.path.join(data_dir, f'{TARGET}_{NOISE}')
        os.makedirs(virtual_dataset_dir, exist_ok = True)
        with open(virtual_dataset_dir + f'/result_{datetime.now().strftime("%H%M%S")}.pickle', 'wb') as f:
            pickle.dump(results, f)

        del SHF

# %%
"""
4. Reporting calculation time
"""
print("{:.2f}s taken".format(time.time() - now))
