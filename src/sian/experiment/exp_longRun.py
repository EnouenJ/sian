from .experiment import *
from .experiment_mixins import LongRunMixin, DatasetSetupMixin


class LongRunExperiment(Experiment, LongRunMixin, DatasetSetupMixin):
    def __init__(self):
        super().__init__()

    def run(self, full_experiment_details):
        print(10 * "=", "Running LongRun Experiment", 10 * "=")
        exp_folder = self.setup_experiment_folder(full_experiment_details["dataset_str"])
        return self.run_longrun(full_experiment_details, exp_folder)
	
