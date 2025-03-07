from utils import *

# %%
def generate_dataset(num_data, sigma, seed = 1234):
    """
    For easy identification of ground truth, we link the nodes with linear equations
    """
    np.random.seed(seed)
    print(f"Generating virtual dataset...", end = '')

    # Function to simulate IPF calculation for a generic process stage
    def calculate_ipf(*args):
        # args are the PPs and previous IPFs, with the last argument being the noise scale
        noise_scale = args[-1]
        weights = np.random.normal(1.0, scale = 0.5, size = (len(args)-1))
        weights = np.clip(weights, 0, 2)
        PPs = np.vstack(list(args[:-1])).T
        IPF = PPs @ weights + np.random.normal(0, noise_scale, num_data)
        return IPF, weights

    # Initialize PPs for 4 stages with random values
    data_pps = np.vstack([np.random.rand(num_data) for _ in range(10)]).T  # Assuming a total of 10 PPs across 4 stages

    data_ipfA_An, wA_A = calculate_ipf(data_pps[:,0], data_pps[:,1], sigma)
    data_ipfA_Cat, wA_C = calculate_ipf(data_pps[:,2], data_pps[:,3], sigma)
    data_ipfB, wB = calculate_ipf(data_pps[:,4], data_pps[:,5], data_ipfA_An, data_ipfA_Cat, sigma)
    data_ipfC, wC = calculate_ipf(data_pps[:,6], data_pps[:,7], data_ipfB, sigma)

    data_fpp, wD = calculate_ipf(data_pps[:,8], data_pps[:,9], data_ipfC, sigma)

    # Create DataFrame with PPs and IPFs
    X = pd.DataFrame({
        'PP_A_An1': data_pps[:,0], 'PP_A_An2': data_pps[:, 1],
        'PP_A_Cat1': data_pps[:,2], 'PP_A_Cat2': data_pps[:,3],
        'PP_B1': data_pps[:,4], 'PP_B2': data_pps[:,5],
        'PP_C1': data_pps[:,6], 'PP_C2': data_pps[:,7],
        'PP_D1': data_pps[:,8], 'PP_D2': data_pps[:,9],
        'IPF_A_An1': data_ipfA_An, 'IPF_A_Cat1': data_ipfA_Cat,
        'IPF_B1': data_ipfB,
        'IPF_C1': data_ipfC
    })

    # Assuming the final product quality metric is based on the last IPFs
    y = pd.Series(data_fpp)

    # Returning the ground truth, using linear coefficients
    args = wA_A, wA_C, wB, wC, wD

    gt_direct = get_gt_direct(args)
    gt_indirect = get_gt_indirect(args)

    print("Done!")
    return X, y, gt_direct, gt_indirect


def get_gt_direct(args):
    wA_A, wA_C, wB, wC, wD = args
    sum_wA_A = sum(wA_A)
    sum_wA_C = sum(wA_C)
    sum_wB = sum([w for w in wB[:-2]] + [sum_wA_A * wB[-2]] + [sum_wA_C * wB[-1]])
    sum_wC = sum([w for w in wC[:-1]] + [sum_wB * wC[-1]])
    gt_direct = [w for w in wD[:-1]] + [sum_wC * wD[-1]]

    gt_direct = [0, 0, 0, 0, 0, 0, 0, 0, gt_direct[0], gt_direct[1], 0, 0, 0, gt_direct[2]]
    # In the order of ['PP_A_Anode_1', 'PP_A_Anode_2', 'PP_A_Cathode_1', 'PP_A_Cathode_2',
    #                 'PP_B1', 'PP_B2', 'PP_C1', 'PP_C2', 'PP_D1', 'PP_D2',
    #                 'IPF_A_Anode_1', 'IPF_A_Cathode_1', 'IPF_B1', 'IPF_C1']

    return gt_direct

def get_gt_indirect(args):
    wA_A, wA_C, wB, wC, wD = args

    wDC = [wD[-1] * c for c in wC]
    wDCB = [wD[-1] * wC[-1] * b for b in wB]
    wDCBA_A = [wD[-1] * wC[-1] * wB[-2] * a_a for a_a in wA_A]
    wDCBA_C = [wD[-1] * wC[-1] * wB[-1] * a_c for a_c in wA_C]
    gt_indirect = wDCBA_A + wDCBA_C + wDCB[:-2] + wDC[:-1] + list(wD[:-1]) + [0,0,0,0]
    # In the order of ['PP_A_Anode_1', 'PP_A_Anode_2', 'PP_A_Cathode_1', 'PP_A_Cathode_2',
    #                 'PP_B1', 'PP_B2', 'PP_C1', 'PP_C2', 'PP_D1', 'PP_D2',
    #                 'IPF_A_Anode_1', 'IPF_A_Cathode_1', 'IPF_B1', 'IPF_C1']

    return gt_indirect


def compare_explanation(features, gt_direct, gt_indirect,
                        boundary_importances, shap_values, lime_values, ig_values, config = 'raw'):
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

    if config == 'raw':
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.plot(features, gt_direct_norm, label='Ground truth - direct')
        plt.plot(features, shf_direct_norm, label='Shapley flow - direct')
        plt.plot(features, shap_values_mean_norm, label='SHAP')
        plt.plot(features, lime_values_mean_norm, label='LIME')
        plt.plot(features, ig_values_mean_norm, label='Integrated gradients')
        plt.xlabel('Features')
        plt.ylabel('Importances')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(features, gt_indirect_norm, label='Ground truth - indirect')
        plt.plot(features, shf_indirect_norm, label='Shapley flow - indirect')
        plt.plot(features, shap_values_mean_norm, label='SHAP')
        plt.plot(features, lime_values_mean_norm, label='LIME')
        plt.plot(features, ig_values_mean_norm, label='Integrated gradients')
        plt.xlabel('Features')
        plt.ylabel('Importances')
        plt.legend()

        plt.tight_layout()
        plt.savefig(figure_dir + '/Comparison.png')
        plt.show()

    elif config == 'error':
        error_shap_direct = np.absolute(shap_values_mean_norm - gt_direct_norm)
        error_shapflow_direct = np.absolute(shf_direct_norm - gt_direct_norm)
        error_lime_direct = np.absolute(lime_values_mean_norm - gt_direct_norm)
        error_ig_direct = np.absolute(ig_values_mean_norm - gt_direct_norm)

        error_shap_indirect = np.absolute(shap_values_mean_norm - gt_indirect_norm)
        error_shapflow_indirect = np.absolute(shf_indirect_norm - gt_indirect_norm)
        error_lime_indirect = np.absolute(lime_values_mean_norm - gt_indirect_norm)
        error_ig_indirect = np.absolute(ig_values_mean_norm - gt_indirect_norm)

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.plot(features, error_shapflow_direct, label='Shapley flow - direct')
        plt.plot(features, error_shap_direct, label='SHAP')
        plt.plot(features, error_lime_direct, label='LIME')
        plt.plot(features, error_ig_direct, label='Integrated Gradients')
        plt.xlabel('Features')
        plt.ylabel('Errors')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(features, error_shapflow_indirect, label='Shapley flow - indirect')
        plt.plot(features, error_shap_indirect, label='SHAP')
        plt.plot(features, error_lime_indirect, label='LIME')
        plt.plot(features, error_ig_indirect, label='Integrated Gradients')
        plt.xlabel('Features')
        plt.ylabel('Errors')
        plt.legend()

        plt.tight_layout()
        plt.savefig(figure_dir + '/Comparison_errors.png')
        plt.show()

    else:
        pass

def compare_rank(features, gt_indirect,
                 shf_indirect_norm,
                 shap_values_mean_norm,
                 lime_values_mean_norm,
                 ig_values_mean_norm,
                 dist = 'kendalltau'):
    feature_pp_indices = ['PP' in feature for feature in features]

    gt_indirect = np.array(gt_indirect)[feature_pp_indices].tolist()
    shf_indirect_norm = np.array(shf_indirect_norm)[feature_pp_indices].tolist()
    shap_values_mean_norm = np.array(shap_values_mean_norm)[feature_pp_indices].tolist()
    lime_values_mean_norm = np.array(lime_values_mean_norm)[feature_pp_indices].tolist()
    ig_values_mean_norm = np.array(ig_values_mean_norm)[feature_pp_indices].tolist()

    from scipy.stats import rankdata
    gt_ranks = len(gt_indirect) + 1 - rankdata(gt_indirect)
    shf_ranks = len(shf_indirect_norm) + 1 - rankdata(shf_indirect_norm)
    shap_ranks = len(shap_values_mean_norm) + 1 - rankdata(shap_values_mean_norm)
    lime_ranks = len(lime_values_mean_norm) + 1 - rankdata(lime_values_mean_norm)
    ig_ranks = len(ig_values_mean_norm) + 1 - rankdata(ig_values_mean_norm)

    if dist == 'kendalltau':
        def rank_distance(rank1, rank2):
            from scipy.stats import kendalltau
            tau, p_value = kendalltau(rank1, rank2)
            return tau

    else:
        def rank_distance(rank1, rank2):
            return np.sum(np.abs(rank1 - rank2))

    distance_ranks_shf = rank_distance(gt_ranks, shf_ranks)
    distance_ranks_shap = rank_distance(gt_ranks, shap_ranks)
    distance_ranks_lime = rank_distance(gt_ranks, lime_ranks)
    distance_ranks_ig = rank_distance(gt_ranks, ig_ranks)

    print("Distance ranks [Shapflow]: {:.3f}".format(distance_ranks_shf))
    print("Distance ranks [SHAP]    : {:.3f}".format(distance_ranks_shap))
    print("Distance ranks [LIME]    : {:.3f}".format(distance_ranks_lime))
    print("Distance ranks [IG]      : {:.3f}".format(distance_ranks_ig))

    return distance_ranks_shf, distance_ranks_shap, distance_ranks_lime, distance_ranks_ig
