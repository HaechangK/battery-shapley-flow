from lime.lime_tabular import LimeTabularExplainer
from utils import *

class LIME:
    def __init__(self, X, y, model = None):
        self.X = X
        self.y = y

        if model == None:
            print("Training new model")
            from xgboost import XGBRegressor
            model = XGBRegressor()
            model.fit(self.X, self.y)
            model.__call__ = model.predict
            print("Done!")

        self.model = model

    def explain(self):
        print("Explaining key variables via LIME...")
        explainer = LimeTabularExplainer(
            self.X.values,
            feature_names=self.X.columns,
            mode="regression"
        )
        predict_fn = lambda x: self.model.predict(x).reshape(-1)

        explanations = []
        for i in range(self.X.values.shape[0]):
            explanation = explainer.explain_instance(
                self.X.values[i],
                predict_fn,
                num_features=len(self.X.columns)
            )
            explanations.append(explanation)

        feature_importance = {feature: [] for feature in self.X.columns}

        # Collect feature attributions from every explanation
        for explanation in explanations:
            local_importance = explanation.local_exp[0]
            for feature_idx, weight in local_importance:
                feature = self.X.columns[feature_idx]
                feature_importance[feature].append(abs(weight))

        lime_values = pd.DataFrame(feature_importance)
        return lime_values
