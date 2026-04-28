from .experiment import *
from .experiment_mixins import SyntheticSweepMixin, DatasetSetupMixin, RealWorldDatasetExperimentMixin


class SyntheticSweepExperiment(Experiment, SyntheticSweepMixin, DatasetSetupMixin):
    def __init__(self):
        super().__init__()

    def run(self, full_experiment_details):
        print(10 * "=", "Running Synthetic Sweep Experiment", 10 * "=")
        exp_folder = self.setup_experiment_folder(full_experiment_details["dataset_str"])
        return self.run_synthetic_sweep(full_experiment_details, exp_folder)
    

class RealWorldDatasetExperiment(Experiment, DatasetSetupMixin, RealWorldDatasetExperimentMixin):
    def __init__(self):
        super().__init__()
        self.default_target_sample_sizes = [800]

    def run(self, full_experiment_details):
        exp_folder = self.setup_experiment_folder(full_experiment_details["dataset_str"])
        print(f"Experiment folder: {exp_folder}")

        self.run_realworld_experiment(full_experiment_details, exp_folder)
	
