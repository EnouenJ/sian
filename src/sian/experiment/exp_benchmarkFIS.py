from .experiment import *
from .experiment_mixins import CompareFISHyperparametersMixin, DatasetSetupMixin

class CompareFISHyperparametersExperiment(Experiment, CompareFISHyperparametersMixin, DatasetSetupMixin):
	def __init__(self):
		super().__init__()

	def run(self, full_experiment_details):
		print(10 * "=", "Running Compare FIS Hyperparameters Experiment", 10 * "=")
		exp_folder = self.setup_experiment_folder(full_experiment_details["dataset_str"])
		return self.compare_fis_hyperparameters(full_experiment_details, exp_folder)
	
