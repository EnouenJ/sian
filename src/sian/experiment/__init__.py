



from .experiment import Experiment


# from .experiment import SyntheticSweepExperiment
# from .experiment import LongRunExperiment
# from .experiment import CompareFISHyperparametersExperiment
# from .experiment import InflationTestExperiment
# from .experiment import RealWorldDatasetExperiment
# from .experiment import RunFISOnlyExperiment
# from .experiment import TrainASingleSIANModel

# from .experiment import TrainAnMLPExperiment








from .exp_sampleSizeSweep import SyntheticSweepExperiment, RealWorldDatasetExperiment
from .exp_longRun import LongRunExperiment
from .exp_benchmarkFIS import CompareFISHyperparametersExperiment
from .exp_benchmarkInflation import InflationTestExperiment


from .exp_step1trainMLP import TrainAnMLPExperimentBenchmark, TrainAnMLPExperimentSweep
from .exp_step2runFIS import RunFISOnlyExperiment
from .exp_step3trainSIAN import TrainASingleSIANModelBenchmark, TrainASingleSIANModelSweep


from .exp_combined import CombinedExperiment




#10/02/2025 -- TODO: remove and reorganize
from .helpers import initalize_the_explainer
from .helpers import train_mlp_final, do_the_fis_final, train_sian_final


#10/06/2025 -- TODO: remove and reorganize
from .experiment import return_score_type_name, prepare_for_FIS_v2, really_do_the_fis

from .experiment_mixins import DatasetSetupMixin, MLPTrainingSweepMixin, MLPTrainingBenchmarkMixin, FISExecutionMixin, CompareFISHyperparametersMixin, SIANTrainingBenchmarkMixin, SIANTrainingSweepMixin, LongRunMixin, SyntheticSweepMixin, RealWorldDatasetExperimentMixin

from .exp_step1_and_2 import CombinedMLPandFISExperiment