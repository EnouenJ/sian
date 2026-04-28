from .experiment import *
from .experiment_mixins import FISExecutionMixin, DatasetSetupMixin

class RunFISOnlyExperiment(Experiment, FISExecutionMixin, DatasetSetupMixin):
    def __init__(self):
        super().__init__()
        self.default_target_sample_sizes = [800]

    def run(self, full_experiment_details):
        print(10 * "=", "Running FIS Only Experiment", 10 * "=")
        mlp_location_dict = full_experiment_details["mlp_location_dict"]
        return self.run_fis(full_experiment_details, mlp_location_dict)
	
