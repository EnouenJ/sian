from .experiment import *
from .experiment_mixins import SIANTrainingSweepMixin, SIANTrainingBenchmarkMixin, DatasetSetupMixin

class TrainASingleSIANModelBenchmark(Experiment, SIANTrainingBenchmarkMixin, DatasetSetupMixin):
	def __init__(self):
		super().__init__()
		self.SIAN_saving_settings_results_to_save = ['best_net', 'final_net']

	def run(self, full_experiment_details):
		print(10 * "=", "Running SIAN Only Experiment", 10 * "=")
		exp_folder = self.setup_experiment_folder(full_experiment_details["dataset_str"])
		return self.train_and_save_sian(
			full_experiment_details,
			exp_folder,
			fis_json_location=full_experiment_details["fis_json_location"],
			MAX_K=full_experiment_details["MAX_K"],
		)

class TrainASingleSIANModelSweep(Experiment, SIANTrainingSweepMixin, DatasetSetupMixin):
	def __init__(self):
		super().__init__()
		self.SIAN_saving_settings_results_to_save = ['best_net', 'final_net']

	def run(self, full_experiment_details):
		print(10 * "=", "Running SIAN Only Experiment", 10 * "=")
		exp_folder = self.setup_experiment_folder(full_experiment_details["dataset_str"])
		return self.train_and_save_sian(
			full_experiment_details,
			exp_folder,
			fis_json_location=full_experiment_details["fis_json_location"],
			MAX_K=full_experiment_details["MAX_K"],
		)

