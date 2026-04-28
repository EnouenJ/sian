from .experiment import Experiment
from .experiment_mixins import MLPTrainingBenchmarkMixin, FISExecutionMixin, DatasetSetupMixin

class CombinedMLPandFISExperiment(Experiment, MLPTrainingBenchmarkMixin, FISExecutionMixin, DatasetSetupMixin):
	def __init__(self):
		super().__init__()

	def run(self, full_experiment_details):
		print(10 * "=", "Running Combined MLP and FIS Experiment", 10 * "=")
		exp_folder = self.setup_experiment_folder(full_experiment_details["dataset_str"])
		mlp_location_dict = self.train_mlp_benchmark(full_experiment_details, exp_folder)
		full_experiment_details["mlp_location_dict"] = mlp_location_dict
		return self.run_fis(full_experiment_details, mlp_location_dict)
	
