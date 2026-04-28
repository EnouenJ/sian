import json
import argparse
from configurate import setup_config, CommonConfig
from sian.experiment import (
    CompareFISHyperparametersExperiment,
    TrainAnMLPExperimentBenchmark,
    TrainAnMLPExperimentSweep,
    RunFISOnlyExperiment,
    TrainASingleSIANModelBenchmark, 
    TrainASingleSIANModelSweep,
    CombinedMLPandFISExperiment,
    LongRunExperiment,
    SyntheticSweepExperiment,
    RealWorldDatasetExperiment,
    InflationTestExperiment
)

CONFIGURATION_FLAG = setup_config()

parser = argparse.ArgumentParser(description="Unified Experiment Runner")
parser.add_argument(
    '--experiment_details_file_path',
    type=str,
    default="",
    help="JSON File Path for the experiment configuration file"
)

args = parser.parse_args()
experiment_details_file_path = args.experiment_details_file_path
with open(experiment_details_file_path, 'r') as file:
    experiment_details_json = json.load(file)

type_of_experiment = experiment_details_json.get("type_of_experiment")

# exp = CombinedExperiment()
# exp.run() #logical conclusion of the current way CombinedExperiment() is implemented



if type_of_experiment == "compare_fis":
    exp = CompareFISHyperparametersExperiment()
elif type_of_experiment == "train_mlp_and_fis":
    exp = CombinedMLPandFISExperiment()
elif type_of_experiment == "run_exp_realworld.py":
    exp = TrainAnMLPExperimentBenchmark()
elif type_of_experiment == "train_mlp_sweep":
    exp = TrainAnMLPExperimentSweep()
elif type_of_experiment == "train_sian_benchmark":
    exp = TrainASingleSIANModelBenchmark()
elif type_of_experiment == "train_sian_sweep":
    exp = TrainASingleSIANModelSweep()
elif type_of_experiment in "longrun":
    exp = LongRunExperiment()
elif type_of_experiment == "synthetic_sweep_BS_and_LR" or type_of_experiment == "exp_bugcheck_sweep":
    exp = SyntheticSweepExperiment()
elif type_of_experiment == "real_world_dataset_experiment":
    exp = RealWorldDatasetExperiment()
elif type_of_experiment == "test_inflation":
    exp = InflationTestExperiment()
else:
    raise ValueError(f"Unknown type_of_experiment: {type_of_experiment}")

exp.run( experiment_details_json )