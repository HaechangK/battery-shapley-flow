from shapflow.flow3 import GraphExplainer, translator
from shapflow.flow3 import build_feature_graph
from shapflow.flow3 import CausalLinks, create_xgboost_f
from shapflow.flow3 import edge_credits2edge_credit, node_dict2str_dict
from utils import *
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# %%
shapflow_dir = os.path.join(figure_dir, 'SHAPflow')
data_graph_dir = os.path.join(data_dir, 'Causal graph')
data_shapflow_dir = os.path.join(data_dir, 'Shapflow values')
os.makedirs(shapflow_dir, exist_ok=True)
os.makedirs(data_graph_dir, exist_ok=True)
os.makedirs(data_shapflow_dir, exist_ok=True)

font_size = 22
plt.rcParams['axes.titlesize'] = font_size
plt.rcParams['axes.labelsize'] = font_size
plt.rcParams['xtick.labelsize'] = font_size - 2
plt.rcParams['ytick.labelsize'] = font_size - 2
plt.rcParams['legend.fontsize'] = font_size - 2
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "dejavuserif"


# %% SHAP flow class
class SHAPflow():
    def __init__(self, X, y, n_bg, nsamples, nruns, target, feature_mask = False):
        """
        Args:
            X: [pd.DataFrame] Data to be interpreted
            y: [pd.Series] Vector of target variables
            n_bg: [int] Number of background samples, for comparison
            nsamples: [int] Number of foreground samples to be interpreted
            nruns: [int] Number of independent runs of Shapley flow calculations (Number of monte carlo samplings)
            target: [str] Target variable string
            feature_mask: [bool] Whether to mask features
        """
        model = None
        self.X = X
        self.y = y
        self.model = model
        self.feature_mask = feature_mask

        if isinstance(model, str): # When using pre-saved model
            print("Loading regression model...", end='')
            savename = model_dir + "/{}".format(model)
            self.model = joblib.load(savename)
            print("Done!")

        print("Loading shapflow explanations...")
        self.n_bg = n_bg
        self.nsamples = nsamples
        self.nruns = nruns
        self.bg = X.fillna(X.mean()).sample(self.n_bg, random_state = 21)  # background samples
        self.fg = X.sample(n = self.nsamples, random_state = 21)  # foreground samples to explain
        self.fg_y = y.sample(n = self.nsamples, random_state = 21)  # foreground samples to explain
        self.sample_ind = -1  # sample to show
        self.target = target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)

    def build_graph(self):
        """
        This function makes dataset generated from graph structure that resembles the sequential batch process.
        For easy clarification, linear relation is preferred.
        Returns:
            self.causal_graph
        """
        PPs_A_Cathode = [name for name in self.X.columns if 'PP_A_Cat' in name]
        IPFs_A_Cathode = [name for name in self.X.columns if 'IPF_A_Cat' in name]

        PPs_A_Anode = [name for name in self.X.columns if 'PP_A_An' in name]
        IPFs_A_Anode = [name for name in self.X.columns if 'IPF_A_An' in name]

        PPs_B = [name for name in self.X.columns if 'PP_B' in name]
        IPFs_B = [name for name in self.X.columns if 'IPF_B' in name]

        PPs_C = [name for name in self.X.columns if 'PP_C' in name]
        IPFs_C = [name for name in self.X.columns if 'IPF_C' in name]

        PPs_D = [name for name in self.X.columns if 'PP_D' in name]

        causal_links = CausalLinks()

        m1 = XGBRegressor(n_estimators=10)
        m1.fit(self.X_train[PPs_A_Cathode], self.X_train['IPF_A_Cat1'])
        r21 = r2_score(self.X_test['IPF_A_Cat1'], m1.predict(self.X_test[PPs_A_Cathode]))
        print("R2 Score Train: {:.3f}".format(
            r2_score(self.X_train['IPF_A_Cat1'], m1.predict(self.X_train[PPs_A_Cathode]))))
        print("R2 Score Test : {:.3f}".format(r21))
        causal_links.add_causes_effects(causes=PPs_A_Cathode, effects=['IPF_A_Cat1'],
                                        models=create_xgboost_f(PPs_A_Cathode, m1, output_margin=True))

        m2 = XGBRegressor(n_estimators=10)
        m2.fit(self.X_train[PPs_A_Anode], self.X_train['IPF_A_An1'])
        r22 = r2_score(self.X_test['IPF_A_An1'], m2.predict(self.X_test[PPs_A_Anode]))
        print(
            "R2 Score Train: {:.3f}".format(r2_score(self.X_train['IPF_A_An1'], m2.predict(self.X_train[PPs_A_Anode]))))
        print("R2 Score Test : {:.3f}".format(r22))
        causal_links.add_causes_effects(causes=PPs_A_Anode, effects=['IPF_A_An1'],
                                        models=create_xgboost_f(PPs_A_Anode, m2, output_margin=True))

        m3 = XGBRegressor(n_estimators=10)
        m3.fit(self.X_train[IPFs_A_Cathode + IPFs_A_Anode + PPs_B], self.X_train['IPF_B1'])
        r23 = r2_score(self.X_test['IPF_B1'], m3.predict(self.X_test[IPFs_A_Cathode + IPFs_A_Anode + PPs_B]))
        print("R2 Score Train: {:.3f}".format(
            r2_score(self.X_train['IPF_B1'], m3.predict(self.X_train[IPFs_A_Cathode + IPFs_A_Anode + PPs_B]))))
        print("R2 Score Test : {:.3f}".format(r23))
        causal_links.add_causes_effects(causes=IPFs_A_Cathode + IPFs_A_Anode + PPs_B, effects=['IPF_B1'],
                                        models=create_xgboost_f(IPFs_A_Cathode + IPFs_A_Anode + PPs_B, m3,
                                                                output_margin=True))

        m4 = XGBRegressor(n_estimators=10)
        m4.fit(self.X_train[IPFs_B + PPs_C], self.X_train['IPF_C1'])
        r24 = r2_score(self.X_test['IPF_C1'], m4.predict(self.X_test[IPFs_B + PPs_C]))
        print(
            "R2 Score Train: {:.3f}".format(r2_score(self.X_train['IPF_C1'], m4.predict(self.X_train[IPFs_B + PPs_C]))))
        print("R2 Score Test : {:.3f}".format(r24))
        causal_links.add_causes_effects(causes=IPFs_B + PPs_C, effects=['IPF_C1'],
                                        models=create_xgboost_f(IPFs_B + PPs_C, m4, output_margin=True))

        m5 = XGBRegressor(n_estimators=10)
        m5.fit(self.X_train[IPFs_C + PPs_D], self.y_train)
        r25 = r2_score(self.y_test, m5.predict(self.X_test[IPFs_C + PPs_D]))
        print("R2 Score Train: {:.3f}".format(r2_score(self.y_train, m5.predict(self.X_train[IPFs_C + PPs_D]))))
        print("R2 Score Test : {:.3f}".format(r25))
        causal_links.add_causes_effects(causes=IPFs_C + PPs_D, effects=['FPP'],
                                        models=create_xgboost_f(IPFs_C + PPs_D, m5, output_margin=True))

        self.r2mean = np.mean(np.array([r21, r22, r23, r24, r25]))

        categorical_feature_names = []
        display_translator = translator(self.X.columns, self.X, self.X.copy())

        self.causal_graph = build_feature_graph(self.X.fillna(self.X.mean()),
                                                causal_links,
                                                categorical_feature_names,
                                                display_translator,
                                                target_name='FPP')
        self.causal_graph.draw()

        return self.causal_graph

    def dropout_links(self, causal_links, dropout_rate=0.2):
        """
        Randomly mutes edges in the causal_links.
        Args:
            causal_links: [list] A list of tuples representing causal links (e.g., [('A', 'B'), ('B', 'C')]).
            dropout_rate: [float] The probability of each edge being 'muted'.
        Returns:
            new_causal_links: [list] A new list of causal_links with some edges 'muted'.
        """
        num_causal_links = len(causal_links.items)-1
        num_links_to_mute = int(dropout_rate * num_causal_links)
        index_to_mute = random.sample(range(num_causal_links), num_links_to_mute)
        new_causal_links = [causal_links.items[index] for index in range(len(causal_links.items)) if index not in index_to_mute]
        return new_causal_links

    # %% Graph Explainer
    def Graph_explain(self, source = None):
        """
        Assigns attributions through edges of causal graphs
        Args:
            source: [list] List that contains strings of source nodes
        """
        # Multiple background result with individual run
        causal_edge_credits = []
        for i in tqdm(range(len(self.bg))):
            self.E = GraphExplainer(self.causal_graph, self.bg[i:i+1])
            self.E.prepare_graph(self.fg)
            self.G = copy.deepcopy(self.E.graph)
            savename = f'./figures/[{self.target}] Causal_graph.png'
            self.G.draw(savename=savename, rank = self.rank2)

            self.explainer = GraphExplainer(self.G, self.bg[i:i + 1], nruns=self.nruns, silent = True)
            self.cf_c = self.explainer.shap_values(self.fg, skip_prepare=True, source=source)
            causal_edge_credits.append(node_dict2str_dict(self.cf_c.edge_credit))

        self.edge_credit = edge_credits2edge_credit(causal_edge_credits, self.cf_c.graph)

        return self.edge_credit


    def draw_graph(self, max_display = 10, source = ''):
        """
        Draws the causal graph computed edge attributions
        Args:
            max_display: [int] Maximum number of edges displayed
            source: [str] Name of the source node to be interpreted
        """
        savename = shapflow_dir + f'/[{self.target}][{source}] bg num{self.n_bg}'
        self.credit_G = self.cf_c.draw(self.sample_ind, max_display=max_display, show_fg_val=False,
                                       edge_credit=self.edge_credit,
                                       format_str = "{:.3f}",
                                       savename=savename,
                                       rank = self.rank2)
        self.save_agraph(self.credit_G)
        print(f"Figure saved as {savename}")


    def importance_matrix(self, edge_credit, remove_noise = True):
        """
        Returns importance matrix given attribution values of edges.
        Args:
            edge_credit: [Edge credit object] Embodies attributions of each edge
            remove_noise: [Bool] Whether to remove noise in importance matrix calculation
        Returns:
            self.edge_importance_matrix: [np.ndarray] Importance matrix
        """
        print("Extracting importance matrix...", end = '')
        # Extract unique nodes from the edge_credit structure
        nodenames = (set(node.name for node in edge_credit.keys())|
                     set(child.name for node in edge_credit.values() for child in node.keys()))
        self.nodenames = sorted(nodenames, key = lambda s: (self.rank2(s), s))  # Sort the nodes to have a consistent order

        # Create a mapping of node names to indices
        self.node_to_index = {node: i for i, node in enumerate(self.nodenames)}

        # Initialize a square matrix of zeros with a size equal to the number of unique nodes
        self.edge_importance_matrix = np.zeros((self.nsamples, len(self.nodenames), len(self.nodenames)))

        # Populate the matrix with the importance values
        for node, child_nodes in edge_credit.items():
            for child, importance in child_nodes.items():
                if isinstance(child_nodes, dict):
                    node_index = self.node_to_index[node.name]
                    child_index = self.node_to_index[child.name]
                    self.edge_importance_matrix[:, node_index, child_index] += importance

        if remove_noise:
            nodenames = self.nodenames.copy()
            self.nodenames = [nodenames[i] for i in range(len(nodenames)) if
                              'noise' not in nodenames[i]]
            not_noise_index = [i for i in range(len(nodenames)) if
                              'noise' not in nodenames[i]]
            self.node_to_index = {node: i for i, node in enumerate(self.nodenames)}
            self.edge_importance_matrix = self.edge_importance_matrix[:,not_noise_index][:,:,not_noise_index]

        # Save the matrix as an attribute for later access
        savename = data_shapflow_dir + '/[{}] SHAPflow importance matrix.npz'.format(self.target)
        np.savez(savename, array = self.edge_importance_matrix, list = self.nodenames)

        print("Done!")

        return self.edge_importance_matrix


    def load_importance_matrix(self):
        """
        Loads saved importance matrix
        Returns:
            self.edge_importance_matrix: [np.ndarray] Importance matrix
        """
        print("Loading importance matrix...", end = '')
        data = np.load(f'./datas/Shapflow values/[{self.target}] SHAPflow importance matrix.npz')
        self.edge_importance_matrix = data['array']
        self.nodenames = data['list'].tolist()
        self.node_to_index = {node: i for i, node in enumerate(self.nodenames)}
        print("Done!")
        return self.edge_importance_matrix


    def plot_importance_matrix(self, maxshow = None):
        """
        Plots importance matrix
        Args:
            maxshow: [None or int] Maximum number of features to show, sorted by sum of importances
        """
        mean_matrix = np.mean(np.absolute(self.edge_importance_matrix), axis=0)
        max_val = np.max(np.abs(mean_matrix))

        if maxshow is not None:
            font_size = 22
            plt.rcParams['axes.titlesize'] = font_size
            plt.rcParams['axes.labelsize'] = font_size
            plt.rcParams['xtick.labelsize'] = font_size - 2
            plt.rcParams['ytick.labelsize'] = font_size - 2
            plt.rcParams['legend.fontsize'] = font_size - 2
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams["mathtext.fontset"] = "dejavuserif"

            matrix_sum = mean_matrix.sum(axis=1)
            ind = np.argsort(matrix_sum)[-maxshow:][::-1]
            ind = np.sort(ind)
            ind = np.append(ind, matrix_sum.shape[0] - 1)
            mean_matrix_reduced = mean_matrix[ind][:, ind]
            self.nodenames_reduced = np.array(self.nodenames)[ind].tolist()

            plt.figure(figsize=(10, 9))
            ax = sns.heatmap(mean_matrix_reduced,
                             xticklabels=self.nodenames_reduced,
                             yticklabels=self.nodenames_reduced,
                             cmap='GnBu',
                             # center=0,
                             linewidths=1.0,
                             vmin=0,
                             vmax=max_val
                             )
            plt.plot(range(len(ind) + 1), range(len(ind) + 1), color='lightgray', linewidth=2)

            cbar = ax.collections[0].colorbar
            cbar.set_label('Feature attribution', rotation=270, labelpad=25)

            savename = shapflow_dir + f'/[{self.target}] Importance matrix top {maxshow} features.png'
            plt.xlabel("Child node")
            plt.ylabel("Parent node")
            plt.tight_layout()
            plt.savefig(savename)
            plt.show()

        else:
            plt.figure(figsize=(30, 30))
            sns.heatmap(mean_matrix,
                        xticklabels=self.nodenames,
                        yticklabels=self.nodenames,
                        cmap='vlag',
                        center=max_val / 2,
                        linewidths=.5,
                        vmin=-max_val,
                        vmax=max_val
                        )
            savename = shapflow_dir + f'/[{self.target}] Importance matrix.png'
            plt.tight_layout()
            plt.savefig(savename)
            plt.show()


    def explanation_boundary(self, rank_boundary):
        """
        Returns 1D vector of importance from importance matrix, based on definition of rank boundary
        Args:
            rank_boundary: [int] Rank boundary. e.g) 0: Root cause boundary, -1: Direct boundary
        Returns:
            boundary_importance: [np.ndarray] 1D vector of attribution results
        """
        print("Extracting explanations for certain boundary..." , end = '')
        if rank_boundary == -1:
            rank_boundary = max([self.rank2(nodename) for nodename in self.nodenames])-1

        self.rank_boundary = rank_boundary
        feature_indices = [self.node_to_index[nodename] for nodename in self.nodenames if self.rank2(nodename) <= self.rank_boundary]
        min_feature_indices, max_feature_indices = min(feature_indices), max(feature_indices)
        cause_nodes = [self.nodenames[i] for i in feature_indices if 'noise' not in self.nodenames[i]]

        boundary_matrix = self.edge_importance_matrix[:,:max_feature_indices+1, max_feature_indices+1:]

        self.fg_boundary = self.fg[cause_nodes]
        boundary_importance = np.sum(boundary_matrix, axis = 2)
        # boundary_importance = pd.DataFrame(boundary_importance, columns = feature_indices).sort_index(axis = 1).values
        boundary_importance = np.hstack((boundary_importance, np.zeros((boundary_importance.shape[0], len(self.nodenames) - len(feature_indices) - 1))))

        print("Done!")

        return boundary_importance


    def SHAPFLOW_plot(self, importance, load_data = False, visuals = [], max_display = 10):
        """
        Borrows the module of shap to feature figures.
        """
        print("Visualizing boundary explanation results...", end = '')
        from XAI.SHAP import SHAP

        self.SH = SHAP(
            X = self.fg,
            y = self.fg_y,
            target = self.target,
            )
        self.SH(load_data=load_data)
        self.SH.result.data = self.fg.to_numpy()
        self.SH.result.values = importance

        with open(data_shapflow_dir + f'/[{self.target}] SHAPflow_rank {self.rank_boundary}.pickle', 'wb') as handle:
            pickle.dump(self.SH.result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.add_info = f' SHAPflow_boundary_{self.rank_boundary}'

        nodenames_splitted = [nodename.split('_', 1) for nodename in self.SH.result.feature_names]
        nodenames = [f'${prefix}_{{{num}}}$' for prefix, num in nodenames_splitted]
        self.SH.result.feature_names = nodenames

        self.SH.SHAP_plot(visuals=visuals,
                          add_info=self.add_info,
                          max_display = max_display)
        print("Done!")

    # %% Saving and Loading AGraph object
    def save_agraph(self, G):
        """
        Saves AGraph object
        Args:
            G: [AGraph object]
        """
        print("Saving AGraph object...", end = '')
        graph_str = G.string()
        savename = data_graph_dir + f'/[{self.target}] Causal graph bg num{self.n_bg}.pkl'
        with open(savename, 'wb') as file:
            pickle.dump(graph_str, file)
        print("Done!")


    def load_agraph(self, savename):
        """
        Loads AGraph object
        Args:
            savename: [str] Saved directory
        Returns:
            loaded_graph: [AGraph object]
        """
        from pygraphviz import AGraph
        print("Loading AGraph object...", end = '')
        with open(savename, 'rb') as file:
            loaded_graph_str = pickle.load(file)
        loaded_graph = AGraph(string=loaded_graph_str)
        print("Done!")
        return loaded_graph


    def node2str(self, graph):
        """
        Generates dictionary that brings nodes from node names
        Args:
            graph: [AGraph object] Causal graph
        Returns:
            n2s: [dict] Dictionary that brings nodes from node names
        """
        n2s = dict()
        for node in graph.nodes:
            n2s[node.name] = node
        return n2s


    # %% Rank functions
    def rank2(self, nodename):
        if nodename[0:2] == 'PP':
            rank = 0
        elif nodename[0:5] == 'IPF_A':
            rank = 2
            if 'noise' in nodename:
                rank = 1
        elif nodename[0:5] == 'IPF_B':
            rank = 4
            if 'noise' in nodename:
                rank = 3
        elif nodename[0:5] == 'IPF_C':
            rank = 6
            if 'noise' in nodename:
                rank = 5
        elif nodename[0:3] == 'FPP':
            rank = 7
        return rank