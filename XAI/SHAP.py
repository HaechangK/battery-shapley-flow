import os
import shap
from xgboost import XGBRegressor
from utils import *

# %% loading directory
shap_dir = os.path.join(figure_dir, 'SHAP')
os.makedirs(shap_dir, exist_ok=True)
water_dir = os.path.join(shap_dir, 'Waterfall')
bee_dir = os.path.join(shap_dir, 'Beeswarm')
decision_dir = os.path.join(shap_dir, 'Decision')
bar_dir = os.path.join(shap_dir, 'Bar')
inter_dir = os.path.join(shap_dir, 'Interaction')
scatter_dir = os.path.join(shap_dir, 'Scatter')
dependence_dir = os.path.join(shap_dir, 'Dependence')
integrated_dir = os.path.join(shap_dir, 'Integrated')

os.makedirs(water_dir, exist_ok=True)
os.makedirs(bee_dir, exist_ok=True)
os.makedirs(decision_dir, exist_ok=True)
os.makedirs(bar_dir, exist_ok=True)
os.makedirs(inter_dir, exist_ok=True)
os.makedirs(scatter_dir, exist_ok=True)
os.makedirs(dependence_dir, exist_ok=True)
os.makedirs(integrated_dir, exist_ok=True)

data_shap_dir = os.path.join(data_dir, 'Shap values')
os.makedirs(data_shap_dir, exist_ok=True)

# %% SHAP module
class SHAP:
    def __init__(self, X, y, target, interventional = True):
        """
        :argument
            X: [pd.DataFrame] Data to be interpreted
            y: [pd.Series] Vector of target variables
            target: [str] Target variable string
            interventional: [bool] Whether to perform interventional SHAP. If false, perform tree_path_dependent SHAP
        """
        self.X = X
        self.y = y
        self.target = target
        self.interventional = interventional
        self.clustered_shap = False

        # The model is trained on all training data
        print("Training new model")
        self.model = XGBRegressor()
        self.model.fit(self.X, self.y)
        print("Done!")


    def __call__(self, load_data = False):
        """
        Computes SHAP values for all features
        :argument
            load_data: [bool] Whether to load the saved SHAP data. If false, compute the SHAP values
        :returns
            self.result.values: [np.ndarray] Matrix containing SHAP values of all features and instances
        """
        # returns SHAP values matrix
        print("Waiting for SHAP analysis...", end='')

        if load_data:
            with open(data_shap_dir + '/SHAP_{}.pickle'.format(self.target), 'rb') as handle:
                self.result =pickle.load(handle)

        else:
            if self.interventional == True:
                self.explainer = shap.TreeExplainer(model=self.model,
                                                    data=self.X,
                                                    feature_perturbation="interventional")
            else:
                self.explainer = shap.TreeExplainer(model=self.model,
                                                    feature_perturbation="tree_path_dependent")

            # Descaling SHAP values
            self.result = self.explainer(self.X)
            self.result.data = self.X
            self.result.values = self.result.values
            self.result.base_value = self.result.base_values[0]
            self.result.base_values = self.result.base_value * np.ones(shape=(self.result.shape[0],))

            # Saves SHAP values into pickle format
            with open(data_shap_dir + '/SHAP_{}.pickle'.format(self.target), 'wb') as handle:
                pickle.dump(self.result, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return self.result.values

    def SHAP_plot(self, visuals, add_info = '', max_display = 10):
        """
        Provides visual aids for the explanation.
        :argument
            visuals: [List] List of visual aids preferred to be drawn.
            add_info: [str] Additional information for saving plots
            max_display: [int] Maximum number of features to display
        Additional Info (Types of visualizations):
            Bar: Mean absolute values of attributions for every feature (global)
            Beeswarm: Absolute values and directions of attributions (global)
            Waterfall: Absolute values and directions of attributions (local)
            Force: Absolute values and directions of attributions (local)
            Decision: Directions of attributions (global)
            Scatter: Attributions against feature values (global)
            Dependence: Attributions against feature values, colored by other feature values(global)
        """
        savename = bar_dir + f'/[{self.target}] Bar{add_info}.png'
        feature_order = shap.plots.bar(self.result,
                                       savename=savename,
                                       max_display = max_display,
                                       )

        if 'Bar' in visuals:
            print("Visualizing Bar plots...", end='')
            savename = bar_dir + f'/[{self.target}] Bar{add_info}.png'
            shap.plots.bar(self.result,
                           order=feature_order,
                           savename=savename,
                           max_display=max_display
                           )

            if self.clustered_shap:
                for label in self.label_set:
                    savename = bar_dir + f'/[{self.target}] Feature importance_Group_{label}{add_info}.png'
                    result = self.result.copy()
                    result.values = self.values[label]
                    result.base_values = self.base_values[label]
                    shap.plots.bar(result,
                                   order=feature_order,
                                   savename=savename,
                                   max_display=max_display
                                   )

        if 'Beeswarm' in visuals:
            print("Visualizing Beeswarm plots...", end='')
            savename = bee_dir + f'/[{self.target}] Beeswarm{add_info}.png'
            shap.plots.beeswarm(self.result,
                                show=True,
                                order=feature_order,
                                max_display = max_display,
                                savedir=savename
                                )
            if self.clustered_shap:
                for label in self.label_set:
                    result = self.result.copy()
                    result.data = self.Xs[label]
                    result.values = self.values[label]
                    result.base_values = self.base_values[label]
                    savename = bee_dir + f'/[{self.target}] Group {label} Beeswarm{add_info}.png'
                    shap.plots.beeswarm(result,
                                        show=True,
                                        order=feature_order,
                                        xlim=(-0.2, 0.2),
                                        savedir=savename)

        if 'Waterfall' in visuals:
            print("Visualizing Waterfall plots...", end='')
            try:
                for label in self.label_set:
                    for i in range(self.Xs[label].shape[0]):
                        result = self.result.copy()
                        result.data = self.Xs[label][i]
                        result.values = self.values[label][i]
                        group_dir = os.path.join(water_dir, f'Group_{label}')
                        os.makedirs(group_dir, exist_ok=True)
                        savename = group_dir + f'/Sample_{i}{add_info}.png'
                        shap.plots.waterfall(result,
                                             show=True,
                                             title=f'Sample_{i}',
                                             savedir=savename)
            except:
                for i in range(100):
                    result = copy.deepcopy(self.result)
                    result.data = self.X.iloc[i,:]
                    result.values = self.result.values[i]
                    result.base_value = self.result.base_value[i]
                    result.base_values = self.result.base_values[i]
                    savename = water_dir + f'/Sample_{i}{add_info}.png'
                    shap.plots.waterfall(result,
                                         show=True,
                                         title=f'Sample_{i}',
                                         savedir=savename)

        if 'Force' in visuals:
            print("Visualizing Force plots...", end='')
            for label in self.label_set:
                for i in range(self.Xs[label].shape[0]):
                    result = self.result.copy()
                    result.data = self.Xs[label][i]
                    result.values = self.values[label][i]
                    group_dir = os.path.join(water_dir, f'Group_{label}')
                    os.makedirs(group_dir, exist_ok=True)
                    savename = group_dir + f'/Sample_{i}{add_info}.png'
                    shap.plots.force(result,
                                     matplotlib=True,
                                     show=True)

        if 'Decision' in visuals:
            print("Visualizing Decision plots...", end='')
            savename = decision_dir + f'/[{self.target}] Decision{add_info}.png'
            shap.plots.decision(self.result.base_value[0],
                                self.result.values,
                                feature_order=feature_order,
                                feature_display_range=range(20, -1, -1),
                                feature_names=self.X.columns.tolist(),
                                title='Groups',
                                savedir=savename,
                                ignore_warnings=True)
            if self.clustered_shap:
                for label in self.label_set:
                    result = self.result.copy()
                    result.values = self.values[label]
                    result.base_values = self.base_values[label]
                    savename = decision_dir + f'/[{self.target}] Group {label} Decision{add_info}.png'
                    shap.plots.decision(result.base_value,
                                        result.values,
                                        feature_order=feature_order,
                                        feature_display_range=range(20, -1, -1),
                                        feature_names=self.X.columns.tolist(),
                                        title='Group {}'.format(label),
                                        savedir=savename,
                                        ignore_warnings=True)

        if 'Scatter' in visuals:
            print("Extracting scatter plot...", end='')
            y_dir = os.path.join(scatter_dir, '[{}]'.format(self.target))
            os.makedirs(y_dir, exist_ok=True)
            for i, feature in enumerate(self.X.columns):
                savename = y_dir + f'/[{self.target}] Scatter_{feature}{add_info}.png'
                shap.plots.scatter(self.result[:, i],
                                   savedir=savename,
                                   show=True)

            if self.clustered_shap:
                for label in self.label_set:
                    result = self.result.copy()
                    result.data = self.Xs[label]
                    result.values = self.values[label]
                    result.base_values = self.base_values[label]
                    group_dir = os.path.join(y_dir, 'Group {}'.format(label))
                    os.makedirs(group_dir, exist_ok=True)
                    for i, feature in enumerate(self.X.columns):
                        savename = group_dir + f'/[{self.target}] Scatter_{feature}{add_info}.png'
                        shap.plots.scatter(result[:, feature],
                                           savedir=savename,
                                           show=False)

        if 'Dependence' in visuals:
            print("Extracting Dependence plot...", end='')
            group_dir = os.path.join(dependence_dir, '[{}]'.format(self.target))
            os.makedirs(group_dir, exist_ok=True)
            for i, feature in enumerate(self.X.columns):
                savename = group_dir + f'/[{self.target}] Dependence_{feature}{add_info}.png'
                shap.dependence_plot(feature,
                                     self.result.values,
                                     self.X,
                                     savedir = savename)

        print("Done!")
