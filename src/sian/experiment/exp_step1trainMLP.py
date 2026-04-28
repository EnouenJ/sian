from .experiment import *
from .experiment_mixins import MLPTrainingBenchmarkMixin, MLPTrainingSweepMixin, DatasetSetupMixin

class TrainAnMLPExperimentBenchmark(Experiment, MLPTrainingBenchmarkMixin, DatasetSetupMixin):
    def __init__(self):
        super().__init__()

    def run(self, full_experiment_details):
        print(10 * "=", "Running MLP Only Experiment", 10 * "=")
        exp_folder = self.setup_experiment_folder(full_experiment_details["dataset_str"])
        return self.train_mlp_benchmark(full_experiment_details, exp_folder)

class TrainAnMLPExperimentSweep(Experiment, MLPTrainingSweepMixin, DatasetSetupMixin):
    def __init__(self):
        super().__init__()

    def run(self, full_experiment_details):
        print(10 * "=", "Running MLP Only Experiment", 10 * "=")
        exp_folder = self.setup_experiment_folder(full_experiment_details["dataset_str"])
        return self.train_mlp_sweep(full_experiment_details, exp_folder)

