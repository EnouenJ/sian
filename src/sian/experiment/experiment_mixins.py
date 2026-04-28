from abc import ABC
import os
import torch
import numpy as np
from sian.data import Final_TabularDataset
from sian.utils import gettimestamp, convert_dict_keys_to_int
from .helpers import train_mlp_final, prepare_for_FIS_v2
from .experiment import *




class DatasetSetupMixin(ABC):
    def setup_dataset(self, full_dataset_details):
        data_base_path = full_dataset_details["data_base_path"]
        dataset_str = full_dataset_details["dataset_str"]
        preproc_owner = full_dataset_details["preproc_owner"] if full_dataset_details["preproc_owner"] != "None" else None
        load_dataset_path = data_base_path
        save_dataset_path = data_base_path + dataset_str + "/"
        
        dataset_obj = Final_TabularDataset(
            dataset_str,
            preproc_owner=preproc_owner,
            load_dataset_path=load_dataset_path,
            save_dataset_path=save_dataset_path
        )
        return dataset_obj, load_dataset_path, save_dataset_path, preproc_owner

    def shuffle_dataset_object(self, dataset_obj, trn_reduc_perc=1.0):
        DEFAULT_trnval_shuffle_seed = 0
        DEFAULT_trnval_split_percentage = 0.7 #TODO: integrate better
        dataset_obj.shuffle_and_split_trnval(trnval_shuffle_seed=0, trnval_split_percentage=0.7, trnval_reduc_percentage=trn_reduc_perc)
        pass




class MLPTraining(ABC):
    
    def train_one_mlp(self,   dataset_obj, mlp_training_args, experiment_data):
        mlp_results = train_mlp_final(dataset_obj, mlp_training_args)

        gpu_details = self.get_device_details()

        self.add_training_params(experiment_data, mlp_training_args)
        self.add_model_config_params(experiment_data, mlp_training_args.model_config, prefix="mlp")
        
        experiment_data.update({
            "MLP_total_training_time": round(mlp_results["total_training_time"], 2),
            "mlp_results.step_data": mlp_results["step_data"],
            "mlp_results.epoch_data": mlp_results["epoch_data"],
            "mlp_location": mlp_results["final_model_path"],
        })
        
        dataset_str = experiment_data["dataset_str"]
        exp_folder = self.setup_experiment_folder(dataset_str)

        TARGET_TRN_N = experiment_data["TARGET_TRN_N"]
        
        saved_results_path = self.save_experiment_data(exp_folder, TARGET_TRN_N, experiment_data, suffix="MLP")

        return mlp_results




class MLPTrainingSweepMixin(MLPTraining):
    def train_mlp_sweep(self, full_experiment_details, exp_folder):
        _, _, _, model_init_seed = self.collect_the_seeds(full_experiment_details)
        full_experiment_details["model_init_seed"] = model_init_seed

        #TODO - rahil remove this because where does it get used
        dataset_str = full_experiment_details["dataset_str"]
        if True:
            full_dataset_details = full_experiment_details #TODO: CREATE THIS AS A SUBDICTIONARY
            data_seed = model_init_seed #TODO: change this
            full_dataset_details['data_seed'] = data_seed

        
        dataset_obj, __, __, preproc_owner = self.setup_dataset(full_dataset_details)
        self.shuffle_dataset_object(dataset_obj) #NOTE: probably unnecessary - trnval not needed yet
        print(f"Using model_init_seed={model_init_seed} and data_seed={data_seed}")


        
        target_sample_sizes = full_experiment_details["target_sample_sizes"]
        target_sample_sizes = [dataset_obj.get_N() if str(target_sample_size) == "get_N" else target_sample_size for target_sample_size in target_sample_sizes]
        target_sample_sizes = [int(target_sample_size) for target_sample_size in target_sample_sizes]
        print('target_sample_sizes', target_sample_sizes)


        mlp_training_args = self.get_training_configuration(full_experiment_details, exp_folder, type_of_model="MLP")
        mlp_training_args.model_config.model_init_seed = model_init_seed

        mlp_location_dict = {}
        for TARGET_TRN_N in target_sample_sizes:
            print(10 * "-", f"Training MLP for sample size: {TARGET_TRN_N}", 10 * "-")
            experiment_data = {}


            #TODO: this will break everything unless we pull out the last line of setup_dataset()
            dataset_obj, _, _, _ = self.setup_dataset(full_dataset_details) 
            self.shuffle_dataset_object(dataset_obj) #NOTE: probably unnecessary - trnval not needed yet
            BASE_TRN_N = dataset_obj.get_N()
            trn_reduc_perc = TARGET_TRN_N / BASE_TRN_N

            gpu_details = self.get_device_details()
            self.add_base_experiment_data(experiment_data, dataset_str, preproc_owner, model_init_seed, data_seed, gpu_details)
            
            experiment_data.update({
                "TARGET_TRN_N": int(TARGET_TRN_N),
            })
            mlp_training_args.trainval_reduction_percentage = trn_reduc_perc
            dataset_obj, _, _, preproc_owner = self.setup_dataset(full_experiment_details)
            self.shuffle_dataset_object(dataset_obj, trn_reduc_perc=trn_reduc_perc)


            mlp_results = self.train_one_mlp(dataset_obj, mlp_training_args, experiment_data)

            
            mlp_location_dict[TARGET_TRN_N] = mlp_results["final_model_path"]
            experiment_data['exp_folder'] = exp_folder
            saved_results_path = self.save_experiment_data(exp_folder, TARGET_TRN_N, experiment_data, suffix="MLP")
            print("MLP JSON results saved successfully")


        return mlp_location_dict




class MLPTrainingBenchmarkMixin(MLPTraining):
    def train_mlp_benchmark(self, full_experiment_details, exp_folder):
        _, _, _, model_init_seed = self.collect_the_seeds(full_experiment_details)
        full_experiment_details["model_init_seed"] = model_init_seed
        data_seed = model_init_seed #TODO: change this
        print(f"Using model_init_seed={model_init_seed} and data_seed={data_seed}")

        dataset_str = full_experiment_details["dataset_str"]
        preproc_owner = full_experiment_details["preproc_owner"] if full_experiment_details["preproc_owner"] != "None" else None

        mlp_training_args = self.get_training_configuration(full_experiment_details, exp_folder, type_of_model="MLP")
        mlp_training_args.model_config.model_init_seed = model_init_seed

        mlp_location_dict = {}
        if True:
            experiment_data = {}
            gpu_details = self.get_device_details()

            self.add_base_experiment_data(experiment_data, dataset_str, preproc_owner, model_init_seed, data_seed, gpu_details)

            dataset_obj, _, _, _ = self.setup_dataset(full_experiment_details)
            self.shuffle_dataset_object(dataset_obj) #NOTE: probably unnecessary - trnval not needed yet
            
            TARGET_TRN_N = int(dataset_obj.get_N())
            experiment_data.update({
                "TARGET_TRN_N": int(TARGET_TRN_N),
            })
            trn_reduc_perc = 1.0
            mlp_training_args.trainval_reduction_percentage = trn_reduc_perc

            mlp_results = self.train_one_mlp(dataset_obj, mlp_training_args, experiment_data)

            
            mlp_location_dict[TARGET_TRN_N] = mlp_results["final_model_path"]
            experiment_data['exp_folder'] = exp_folder
            saved_results_path = self.save_experiment_data(exp_folder, TARGET_TRN_N, experiment_data, suffix="MLP")
            print("MLP JSON results saved successfully")

        return mlp_location_dict







class FISExecutionMixin(ABC):
    def run_fis(self, full_experiment_details, mlp_location_dict):
        is_masked_mlp = full_experiment_details["is_masked_mlp"]
        dataset_str = full_experiment_details["dataset_str"]
        preproc_owner = full_experiment_details["preproc_owner"] if full_experiment_details["preproc_owner"] != "None" else None
        exp_folder = self.setup_experiment_folder(dataset_str)
        print(exp_folder)

        dataset_obj, _, _, _ = self.setup_dataset(full_experiment_details)
        self.shuffle_dataset_object(dataset_obj) #NOTE: probably unnecessary - trnval not needed yet

        _, _, _, model_init_seed = self.collect_the_seeds(full_experiment_details)
        data_seed = model_init_seed
        hyperparameter_combo_list = full_experiment_details["hyperparameter_combo_list"]

        full_interaction_dict = {}
        if True:
            TARGET_TRN_N = dataset_obj.get_N()
            full_interaction_dict[TARGET_TRN_N] = {}
            mlp_location = mlp_location_dict[TARGET_TRN_N]
            mlp_best_net = torch.load(mlp_location, weights_only=False)
            print("Loaded MLP model successfully")

            trn_reduc_perc = 1.0
            dataset_obj, _, _, preproc_owner = self.setup_dataset(full_experiment_details)
            self.shuffle_dataset_object(dataset_obj, trn_reduc_perc=trn_reduc_perc)

            dataset_obj.shuffle_and_split_trnval(trnval_shuffle_seed=0, trnval_split_percentage=0.7, trnval_reduc_percentage=trn_reduc_perc)
            val_tensor = dataset_obj.pull_val_tensor(self.device)
            fis_valX = val_tensor if is_masked_mlp else val_tensor.detach().cpu().numpy()

            for hyperparam_idx, hyperparam_combo in enumerate(hyperparameter_combo_list):
                datetimestr = gettimestamp()
                FIS_style = hyperparam_combo["FIS_style"]
                MAX_K = hyperparam_combo["MAX_K"]
                explainer_score_type = hyperparam_combo["explainer_score_type"]

                jam_arch = prepare_for_FIS_v2(mlp_best_net, is_masked_mlp, dataset_obj, explainer_score_type)

                print(10 * "=", f"Processing hyperparameter combination {hyperparam_idx+1}/{len(hyperparameter_combo_list)} for sample size {TARGET_TRN_N}", 10 * "=")

                if FIS_style == "layerwise":
                    tau_thresholds = convert_dict_keys_to_int(hyperparam_combo["tau_thresholds"])
                    theta_thresholds = convert_dict_keys_to_int(hyperparam_combo["theta_thresholds"])
                    number_of_rounds = None
                    inters_per_round = None
                elif FIS_style == "batchwise":
                    tau_thresholds = convert_dict_keys_to_int(hyperparam_combo["tau_thresholds"])
                    number_of_rounds = hyperparam_combo["number_of_rounds"]
                    inters_per_round = hyperparam_combo["inters_per_round"]
                    theta_thresholds = None
                else:
                    raise ValueError(f"FIS_style={FIS_style} not recognized")

                FIS_interactions, FIS_algorithm_time_taken, FIS_other_stuff, saved_results_path = really_do_the_fis(
                    FIS_style, MAX_K, hyperparam_combo, jam_arch, fis_valX, number_of_rounds, inters_per_round,
                    explainer_score_type, exp_folder, datetimestr, TARGET_TRN_N
                )
                print("FIS_algorithm_time_taken", FIS_algorithm_time_taken)

                batch_round_times_taken = FIS_other_stuff['times_per_round'] if FIS_style == 'batchwise' else None
                gpu_details = self.get_device_details()
                experiment_data = {}
                self.add_base_experiment_data(experiment_data, dataset_str, preproc_owner, model_init_seed, data_seed, gpu_details)
                self.add_fis_params(experiment_data, FIS_style, MAX_K, number_of_rounds, inters_per_round, tau_thresholds, theta_thresholds, explainer_score_type)
                
                experiment_data.update({
                    "TARGET_TRN_N": int(TARGET_TRN_N),
                    "FIS_algorithm_time_taken": FIS_algorithm_time_taken,
                    "FIS_interactions": FIS_interactions,
                    "mlp_location": mlp_location,
                    "batch_round_times_taken": batch_round_times_taken,
                })
                
                saved_results_path = self.save_experiment_data(exp_folder, TARGET_TRN_N, experiment_data, suffix="FIS")
                print("JSON results saved successfully")
                full_interaction_dict[TARGET_TRN_N][hyperparam_idx] = FIS_interactions

        return full_interaction_dict



class CompareFISHyperparametersMixin(ABC):
    def compare_fis_hyperparameters(self, full_experiment_details, exp_folder):
        _, _, _, model_init_seed = self.collect_the_seeds(full_experiment_details)
        full_experiment_details["model_init_seed"] = model_init_seed
        data_seed = model_init_seed
        print(f"Using model_init_seed={model_init_seed} and data_seed={data_seed}")

        dataset_str = full_experiment_details["dataset_str"]
        preproc_owner = full_experiment_details["preproc_owner"] if full_experiment_details["preproc_owner"] != "None" else None
        is_masked_mlp = full_experiment_details["is_masked_mlp"]

        dataset_obj, _, _, _ = self.setup_dataset(full_experiment_details)
        self.shuffle_dataset_object(dataset_obj) #NOTE: probably unnecessary - trnval not needed yet

        target_sample_sizes = full_experiment_details["target_sample_sizes"]
        target_sample_sizes = [int(target_sample_size) for target_sample_size in target_sample_sizes]
        print('target_sample_sizes', target_sample_sizes)

        mlp_training_args = self.get_training_configuration(full_experiment_details, exp_folder, type_of_model="MLP")
        mlp_training_args.model_init_seed = model_init_seed
        mlp_training_args.model_config.model_init_seed = model_init_seed

        hyperparameter_combo_list = full_experiment_details["hyperparameter_combo_list"]
        print('hyperparameter_combo_list', hyperparameter_combo_list)

        full_interaction_dict = {}
        for TARGET_TRN_N in target_sample_sizes:
            full_interaction_dict[TARGET_TRN_N] = {}
            print(10 * "-", f"Processing sample size: {TARGET_TRN_N}", 10 * "-")
            BASE_TRN_N = dataset_obj.get_N()
            trn_reduc_perc = TARGET_TRN_N / BASE_TRN_N
            mlp_training_args.trainval_reduction_percentage = trn_reduc_perc

            dataset_obj, _, _, preproc_owner = self.setup_dataset(full_experiment_details)
            self.shuffle_dataset_object(dataset_obj, trn_reduc_perc=trn_reduc_perc)
            are_we_traning_the_mlp = True
            explainer_score_type = hyperparameter_combo_list[0]["explainer_score_type"] #TODO - wrong
            jam_arch, fis_valX, mlp_results = prepare_for_FIS(are_we_traning_the_mlp, dataset_obj, mlp_training_args, explainer_score_type)

            for hyperparam_idx, hyperparam_combo in enumerate(hyperparameter_combo_list):
                datetimestr = gettimestamp()
                FIS_style = hyperparam_combo["FIS_style"]
                MAX_K = hyperparam_combo["MAX_K"]
                explainer_score_type = hyperparam_combo["explainer_score_type"]
                jam_arch.score_type_name = return_score_type_name(is_masked_mlp, explainer_score_type)

                print(10 * "=", f"Processing hyperparameter combination {hyperparam_idx+1}/{len(hyperparameter_combo_list)} for sample size {TARGET_TRN_N}", 10 * "=")

                if FIS_style == "layerwise":
                    tau_thresholds = convert_dict_keys_to_int(hyperparam_combo["tau_thresholds"])
                    theta_thresholds = convert_dict_keys_to_int(hyperparam_combo["theta_thresholds"])
                    number_of_rounds = None
                    inters_per_round = None
                elif FIS_style == "batchwise":
                    tau_thresholds = convert_dict_keys_to_int(hyperparam_combo["tau_thresholds"])
                    number_of_rounds = hyperparam_combo["number_of_rounds"]
                    inters_per_round = hyperparam_combo["inters_per_round"]
                    theta_thresholds = None
                else:
                    raise ValueError(f"FIS_style={FIS_style} not recognized")

                FIS_interactions, FIS_algorithm_time_taken, FIS_other_stuff, saved_results_path = really_do_the_fis(
                    FIS_style, MAX_K, hyperparam_combo, jam_arch, fis_valX, number_of_rounds, inters_per_round,
                    explainer_score_type, exp_folder, datetimestr, TARGET_TRN_N
                )
                print("FIS_algorithm_time_taken", FIS_algorithm_time_taken)

                gpu_details = self.get_device_details()
                experiment_data = {}
                self.add_base_experiment_data(experiment_data, dataset_str, preproc_owner, model_init_seed, data_seed, gpu_details)
                self.add_training_params(experiment_data, mlp_training_args)
                self.add_model_config_params(experiment_data, mlp_training_args.model_config, prefix="mlp")
                self.add_fis_params(experiment_data, FIS_style, MAX_K, number_of_rounds, inters_per_round, tau_thresholds, theta_thresholds, explainer_score_type)
                
                experiment_data.update({
                    "TARGET_TRN_N": int(TARGET_TRN_N),
                    "FIS_algorithm_time_taken": FIS_algorithm_time_taken,
                    "FIS_interactions": FIS_interactions,
                    "MLP_total_training_time": round(mlp_results["total_training_time"], 2),
                    "mlp_results.step_data": mlp_results["step_data"],
                    "mlp_results.epoch_data": mlp_results["epoch_data"],
                })
                
                saved_results_path = self.save_experiment_data(exp_folder, TARGET_TRN_N, experiment_data, suffix="CompareFIS")
                print("JSON results saved successfully")
                full_interaction_dict[TARGET_TRN_N][hyperparam_idx] = FIS_interactions

        return full_interaction_dict


class SIANTrainingSweepMixin(ABC):
    def train_and_save_sian(self, full_experiment_details, exp_folder, fis_json_location, MAX_K):
        _, _, _, model_init_seed = self.collect_the_seeds(full_experiment_details)
        full_experiment_details["model_init_seed"] = model_init_seed
        data_seed = model_init_seed
        print(f"Using model_init_seed={model_init_seed} and data_seed={data_seed}")

        dataset_str = full_experiment_details["dataset_str"]
        preproc_owner = full_experiment_details["preproc_owner"] if full_experiment_details["preproc_owner"] != "None" else None
        load_dataset_path = full_experiment_details["data_base_path"]
        save_dataset_path = load_dataset_path + dataset_str + "/"

        type_of_experiment = full_experiment_details["type_of_experiment"]

        dataset_obj, _, _, _ = self.setup_dataset(full_experiment_details)
        self.shuffle_dataset_object(dataset_obj) #NOTE: probably unnecessary - trnval not needed yet

        target_sample_sizes = full_experiment_details["target_sample_sizes"]
        target_sample_sizes = [dataset_obj.get_N() if str(target_sample_size) == "get_N" else target_sample_size for target_sample_size in target_sample_sizes]
        target_sample_sizes = [int(target_sample_size) for target_sample_size in target_sample_sizes]

        print('target_sample_sizes', target_sample_sizes)

        sian_training_args = self.get_training_configuration(full_experiment_details, exp_folder, type_of_model="SIAN")
        sian_training_args.model_config.model_init_seed = model_init_seed

        device = sian_training_args.device

        for TARGET_TRN_N in target_sample_sizes:
            print(10 * "-", f"Processing sample size: {TARGET_TRN_N}", 10 * "-")
            BASE_TRN_N = dataset_obj.get_N()
            trn_reduc_perc = TARGET_TRN_N / BASE_TRN_N
            sian_training_args.trainval_reduction_percentage = trn_reduc_perc

            dataset_obj, _, _, preproc_owner = self.setup_dataset(full_experiment_details)
            self.shuffle_dataset_object(dataset_obj, trn_reduc_perc=trn_reduc_perc)

            with open(fis_json_location, 'r') as f:
                json_data = json.load(f)

            correct_keys = []
            for thing in json_data:
                thing_dataset_str = json_data[thing]['dataset_str']
                thing_model_init_seed = json_data[thing]['model_init_seed']
                thing_TARGET_TRN_N = json_data[thing]['TARGET_TRN_N']
                if "MAX_K" in json_data[thing]:
                    thing_MAX_K = json_data[thing]['MAX_K']
                    if thing_dataset_str == dataset_str and thing_MAX_K == MAX_K and thing_model_init_seed == model_init_seed and (type_of_experiment in ["train_sian_sweep"] or thing_TARGET_TRN_N == TARGET_TRN_N):
                        correct_keys.append(thing)
            print('correct_keys', correct_keys)
            assert len(correct_keys) == 1
            correct_key = correct_keys[0]
            FIS_interactions = json_data[correct_key]["FIS_interactions"]
            print('loaded FIS_interactions', len(FIS_interactions))
            FIS_limit = 500
            FIS_interactions = FIS_interactions[:FIS_limit]
            print('limited FIS_interactions', len(FIS_interactions))

            sian_training_args.model_config.FIS_interactions = FIS_interactions

            print(10 * "=", f"Training SIAN-{MAX_K} with sample size {TARGET_TRN_N}", 10 * "=")
            sian_results = train_sian_final(dataset_obj, sian_training_args)
            print(f"SIAN Results: {sian_results}")

            gpu_details = self.get_device_details()
            sian_location = sian_training_args.saving_settings.results_save_prefix + "SIAN-K_final_net.pt"

            experiment_data = {}
            self.add_base_experiment_data(experiment_data, dataset_str, preproc_owner, model_init_seed, data_seed, gpu_details)
            self.add_training_params(experiment_data, sian_training_args)
            self.add_model_config_params(experiment_data, sian_training_args.model_config, prefix="sian")
            
            experiment_data.update({
                "TARGET_TRN_N": int(TARGET_TRN_N),
                "MAX_K": MAX_K,
                "FIS_interactions": FIS_interactions,
                "SIAN_total_training_time": round(sian_results["total_training_time"], 2),
                "sian_results.step_data": sian_results["step_data"],
                "sian_results.epoch_data": sian_results["epoch_data"],
                "fis_json_location": fis_json_location,
                "sian_location": sian_location,
            })
            
            saved_results_path = self.save_experiment_data(exp_folder, TARGET_TRN_N, experiment_data, suffix="SIAN")
            print("SIAN JSON results saved successfully")




class SIANTrainingBenchmarkMixin(ABC):
    def train_and_save_sian(self, full_experiment_details, exp_folder, fis_json_location, MAX_K):
        _, _, _, model_init_seed = self.collect_the_seeds(full_experiment_details)
        full_experiment_details["model_init_seed"] = model_init_seed
        data_seed = model_init_seed
        print(f"Using model_init_seed={model_init_seed} and data_seed={data_seed}")

        dataset_str = full_experiment_details["dataset_str"]
        preproc_owner = full_experiment_details["preproc_owner"] if full_experiment_details["preproc_owner"] != "None" else None
        load_dataset_path = full_experiment_details["data_base_path"]
        save_dataset_path = load_dataset_path + dataset_str + "/"

        type_of_experiment = full_experiment_details["type_of_experiment"]

        dataset_obj, _, _, _ = self.setup_dataset(full_experiment_details)
        self.shuffle_dataset_object(dataset_obj) #NOTE: probably unnecessary - trnval not needed yet

        sian_training_args = self.get_training_configuration(full_experiment_details, exp_folder, type_of_model="SIAN")
        sian_training_args.model_config.model_init_seed = model_init_seed

        device = sian_training_args.device

        if True:
            trn_reduc_perc = 1.0
            sian_training_args.trainval_reduction_percentage = trn_reduc_perc

            dataset_obj, _, _, preproc_owner = self.setup_dataset(full_experiment_details)
            self.shuffle_dataset_object(dataset_obj, trn_reduc_perc=trn_reduc_perc)

            with open(fis_json_location, 'r') as f:
                json_data = json.load(f)

            correct_keys = []
            for thing in json_data:
                thing_dataset_str = json_data[thing]['dataset_str']
                thing_model_init_seed = json_data[thing]['model_init_seed']
                thing_TARGET_TRN_N = json_data[thing]['TARGET_TRN_N']
                if "MAX_K" in json_data[thing]:
                    thing_MAX_K = json_data[thing]['MAX_K']
                    if thing_dataset_str == dataset_str and thing_MAX_K == MAX_K and thing_model_init_seed == model_init_seed and (type_of_experiment in ["train_sian_benchmark"] or thing_TARGET_TRN_N == TARGET_TRN_N):
                        correct_keys.append(thing)
            print('correct_keys', correct_keys)
            assert len(correct_keys) == 1
            correct_key = correct_keys[0]
            FIS_interactions = json_data[correct_key]["FIS_interactions"]
            print('loaded FIS_interactions', len(FIS_interactions))
            FIS_limit = 500
            FIS_interactions = FIS_interactions[:FIS_limit]
            print('limited FIS_interactions', len(FIS_interactions))

            sian_training_args.model_config.FIS_interactions = FIS_interactions

            print(10 * "=", f"Training SIAN-{MAX_K} with sample size {dataset_obj.get_N()}", 10 * "=")
            sian_results = train_sian_final(dataset_obj, sian_training_args)
            print(f"SIAN Results: {sian_results}")

            gpu_details = self.get_device_details()
            sian_location = sian_training_args.saving_settings.results_save_prefix + "SIAN-K_final_net.pt"

            experiment_data = {}
            self.add_base_experiment_data(experiment_data, dataset_str, preproc_owner, model_init_seed, data_seed, gpu_details)
            self.add_training_params(experiment_data, sian_training_args)
            self.add_model_config_params(experiment_data, sian_training_args.model_config, prefix="sian")
            
            experiment_data.update({
                "TARGET_TRN_N": int(dataset_obj.get_N()),
                "MAX_K": MAX_K,
                "FIS_interactions": FIS_interactions,
                "SIAN_total_training_time": round(sian_results["total_training_time"], 2),
                "sian_results.step_data": sian_results["step_data"],
                "sian_results.epoch_data": sian_results["epoch_data"],
                "fis_json_location": fis_json_location,
                "sian_location": sian_location,
            })
            
            saved_results_path = self.save_experiment_data(exp_folder, dataset_obj.get_N(), experiment_data, suffix="SIAN")
            print("SIAN JSON results saved successfully")




class LongRunMixin(ABC):
    def __init__(self):
        super().__init__()
        self.default_target_train_samples = 102400

    def run_longrun(self, full_experiment_details, exp_folder):
        _, _, _, model_init_seed = self.collect_the_seeds(full_experiment_details)

        data_base_path = full_experiment_details["data_base_path"]
        dataset_str = full_experiment_details["dataset_str"]
        preproc_owner = full_experiment_details["preproc_owner"] if full_experiment_details["preproc_owner"] != "None" else None
        is_masked_mlp = full_experiment_details["is_masked_mlp"] 
        is_masked_sian = full_experiment_details["is_masked_sian"] 
        use_mnist_scaling = full_experiment_details["use_mnist_scaling"]
        FIS_style = full_experiment_details["FIS_style"]
        MAX_K = full_experiment_details["MAX_K"]
        number_of_rounds = full_experiment_details["number_of_rounds"]
        inters_per_round = full_experiment_details["inters_per_round"]

        lambda1 = full_experiment_details["lambda1"]

        print("Dataset used:", dataset_str)
        print("Preprocessing Owner:", preproc_owner)
        if FIS_style == "batchwise":
            print("Number of rounds:", number_of_rounds)
            print("Interactions per round:", inters_per_round)

        dataset_obj, _, _, _ = self.setup_dataset(full_experiment_details)
        self.shuffle_dataset_object(dataset_obj) #NOTE: probably unnecessary - trnval not needed yet

        BS, LR, EP, STEP = self.get_basic_training_details(full_experiment_details)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp_training_args = TrainingArgs(BS, EP, LR, device, STEP)
        sian_training_args = TrainingArgs(BS, EP, LR, device, STEP)

        BASE_TRN_N = 700 * 1000 #TODO
        if "TARGET_TRN_N" in full_experiment_details:
            TARGET_TRN_N = full_experiment_details["TARGET_TRN_N"]
        else:
            raise Exception("LongRun() not supporting default \'TARGET_TRN_N\'")
            TARGET_TRN_N = self.default_target_train_samples
        print('TARGET_TRN_N', TARGET_TRN_N)

        mlp_training_args.model_config.net_name = "MLP"
        mlp_training_args.model_config.sizes = self.MLP_network_hidden_sizes
        mlp_training_args.model_config.is_masked = is_masked_mlp
        mlp_training_args.model_config.use_mnist_scaling = use_mnist_scaling
        mlp_training_args.saving_settings.exp_folder = exp_folder
        mlp_training_args.saving_settings.results_to_save = self.MLP_saving_settings_results_to_save
        if True: #TODO - clean up, part of hypers
            mlp_training_args.return_settings.epochs_to_evaluate = []
            mlp_training_args.return_settings.steps_to_evaluate = []

        sian_training_args.model_config.net_name = "SIAN-K"
        sian_training_args.model_config.sizes = self.SIAN_network_hidden_sizes
        sian_training_args.model_config.small_sizes = self.SIAN_network_hidden_small_sizes
        sian_training_args.model_config.is_masked = is_masked_sian
        sian_training_args.model_config.use_mnist_scaling = use_mnist_scaling
        sian_training_args.saving_settings.exp_folder = exp_folder
        sian_training_args.saving_settings.results_to_save = self.SIAN_saving_settings_results_to_save
        if True:
            sian_training_args.saving_settings.saving_after_evaluation = True   #TODO: special feature of LongRun()
        if "compute_shapeloss" in full_experiment_details:
            sian_training_args.compute_shapeloss = full_experiment_details['compute_shapeloss']
        if True:
            sian_training_args.lambda1 = lambda1
            
        if is_masked_sian: #TODO - clean up, part of hypers
            halfstep = STEP // 2
            halfpow = int(np.log(halfstep)/np.log(2))
            masked_steps_to_evaluate = [0] + list(np.power(2,np.arange(100)))
            masked_steps_to_evaluate = [0] + list(np.power(2,np.arange(halfpow+1))) + list(halfstep + np.power(2,np.arange(halfpow+1)))
            print('masked_steps_to_evaluate',masked_steps_to_evaluate)
            sian_training_args.return_settings.steps_to_evaluate = masked_steps_to_evaluate

        D = dataset_obj.get_D()
        readable_labels = dataset_obj.get_readable_labels()
        print(readable_labels)

        assert FIS_style=="maximal"
        if True:
            hyperparam_combinations=[None]

            print(10 * "-", f"Processing sample size: {TARGET_TRN_N}", 10 * "-")
            trn_reduc_perc = TARGET_TRN_N / BASE_TRN_N
            mlp_training_args.trainval_reduction_percentage = trn_reduc_perc
            sian_training_args.trainval_reduction_percentage = trn_reduc_perc

            are_we_traning_the_mlp = (FIS_style in ["layerwise", "batchwise"])       
            explainer_score_type = "arch" 
            jam_arch, fis_valX, mlp_results = prepare_for_FIS(are_we_traning_the_mlp, dataset_obj, mlp_training_args, explainer_score_type)

            for hyperparam_idx, hyperparam_combo in enumerate(hyperparam_combinations):
                datetimestr = gettimestamp()
                print(f"Processing hyperparameter combination {hyperparam_idx+1}/{len(hyperparam_combinations)} for sample size {TARGET_TRN_N}")
                FIS_interactions, FIS_algorithm_time_taken, FIS_other_stuff, saved_results_path = really_do_the_fis(FIS_style, MAX_K, hyperparam_combo, jam_arch, fis_valX, number_of_rounds, inters_per_round, explainer_score_type,    exp_folder, datetimestr, TARGET_TRN_N)
                print("FIS_algorithm_time_taken", FIS_algorithm_time_taken)
                print("FIS_interactions")
                print(FIS_interactions)

                sian_training_args.model_config.model_init_seed = model_init_seed
                print(f"Model init seed: {model_init_seed}")

                if True:
                    experiment_data = {
                        "exp_folder": exp_folder,
                        "dataset_str": dataset_str,
                        "batch_size": BS,
                        "epochs": EP,
                        "steps": STEP,
                        "learning_rate": LR,
                        "preproc_owner": preproc_owner,
                        "FIS_style": FIS_style,
                        "is_masked_mlp": is_masked_mlp,
                        "is_masked_sian": is_masked_sian,
                        "use_mnist_scaling": use_mnist_scaling,
                        "model_init_seed": model_init_seed,
                        "MAX_K": MAX_K,
                        "TARGET_TRN_N": int(TARGET_TRN_N),
                        "number_of_rounds": number_of_rounds if FIS_style == "batchwise" else None,
                        "inters_per_round": inters_per_round if FIS_style == "batchwise" else None,
                        "tau_thresholds": tau_thresholds if (FIS_style in ["batchwise","layerwise"]) else None,
                        "theta_thresholds": theta_thresholds if FIS_style == "layerwise" else None,
                        "FIS_algorithm_time_taken": FIS_algorithm_time_taken,
                        "FIS_interactions": FIS_interactions,
                    }


                    sian_training_args.saving_settings.saved_results_path = saved_results_path   #TODO: special feature of LongRun()
                    sian_training_args.saving_settings.experiment_data = experiment_data   #TODO: special feature of LongRun()


                print(10 * "=", f"Training SIAN-{MAX_K} with sample size {TARGET_TRN_N}", 10 * "=")

                sian_training_args.model_config.FIS_interactions = FIS_interactions
                sian_results = train_sian_final(dataset_obj, sian_training_args)
                trained_sian = sian_results["best_net"]
                val_tensor = sian_results["val_tensor"]
                print(f"SIAN Results: {sian_results}")

                gpu_details = get_device_details(device)
                if "final_test_mse" in sian_results:
                    test_metric_value = round(float(sian_results["final_test_mse"]["mse"]), 5)

                epoch_data = sian_results.get("epoch_data", {})
                if len(epoch_data) > 0:
                    last_epoch_evaluated = max(epoch_data.keys())
                    train_mse = round(float(epoch_data[last_epoch_evaluated]["trn_metric"]), 5)
                    val_mse = round(float(epoch_data[last_epoch_evaluated]["val_metric"]), 5)
                else:
                    last_epoch_evaluated = None
                    train_mse = None
                    val_mse = None

                experiment_data2 = {
                    "MLP_total_training_time": round(mlp_results["total_training_time"], 2),
                    "mlp_results.step_data" : mlp_results["step_data"],
                    "mlp_results.epoch_data" : mlp_results["epoch_data"],
                    "sian_results.step_data" : sian_results["step_data"],
                    "sian_results.epoch_data" : sian_results["epoch_data"],

                    "SIAN_total_training_time": round(sian_results["total_training_time"], 2),
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                    "final_test_mse.mse": round(float(sian_results["final_test_mse"]["mse"]), 5),
                    "GPU_name": gpu_details["gpu_name"]
                }
                for thing in experiment_data2:
                    experiment_data[thing] = experiment_data2[thing]
                print(experiment_data)
                with open(saved_results_path, "w") as f:
                    json.dump(experiment_data, f, indent=4)
                print("JSON results saved successfully")


class SyntheticSweepMixin(ABC):   
    def run_synthetic_sweep(self, full_experiment_details, exp_folder):
        _, _, _, model_init_seed = self.collect_the_seeds(full_experiment_details)
        full_experiment_details["model_init_seed"] = model_init_seed

        data_base_path = full_experiment_details["data_base_path"]
        dataset_str = full_experiment_details["dataset_str"]
        preproc_owner = full_experiment_details["preproc_owner"] if full_experiment_details["preproc_owner"] != "None" else None
        is_masked_mlp = full_experiment_details["is_masked_mlp"] 
        is_masked_sian = full_experiment_details["is_masked_sian"] 
        use_mnist_scaling = full_experiment_details["use_mnist_scaling"]
        FIS_style = full_experiment_details["FIS_style"]
        MAX_K = full_experiment_details["MAX_K"]
        number_of_rounds = full_experiment_details["number_of_rounds"]
        inters_per_round = full_experiment_details["inters_per_round"]

        print("Dataset used:", dataset_str)
        print("Preprocessing Owner:", preproc_owner)
        if FIS_style == "batchwise":
            print("Number of rounds:", number_of_rounds)
            print("Interactions per round:", inters_per_round)

        dataset_obj, _, _, _ = self.setup_dataset(full_experiment_details)
        self.shuffle_dataset_object(dataset_obj) #NOTE: probably unnecessary - trnval not needed yet

        target_sample_sizes = full_experiment_details.get("target_sample_sizes")
        target_sample_sizes = [dataset_obj.get_N() if str(target_sample_size) == "get_N" else int(target_sample_size) for target_sample_size in target_sample_sizes]
        print('target_sample_sizes', target_sample_sizes)

        BS, LR, EP, STEP = self.get_basic_training_details(full_experiment_details)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp_training_args = TrainingArgs(BS, EP, LR, device, STEP)
        sian_training_args = TrainingArgs(BS, EP, LR, device, STEP)

        if "lambda1" in full_experiment_details:
            lambda1 = full_experiment_details["lambda1"]
        else:
            lambda1 = sian_training_args.lambda1

        mlp_training_args.model_config.net_name = "MLP"
        mlp_training_args.model_config.sizes = self.MLP_network_hidden_sizes
        mlp_training_args.model_config.is_masked = is_masked_mlp
        mlp_training_args.model_config.use_mnist_scaling = use_mnist_scaling
        mlp_training_args.saving_settings.exp_folder = exp_folder
        mlp_training_args.saving_settings.results_to_save = self.MLP_saving_settings_results_to_save
        if True: #TODO - clean up, part of hypers
            mlp_training_args.return_settings.epochs_to_evaluate = []
            mlp_training_args.return_settings.steps_to_evaluate = []

        sian_training_args.model_config.net_name = "SIAN-K"
        sian_training_args.model_config.sizes = self.SIAN_network_hidden_sizes
        sian_training_args.model_config.small_sizes = self.SIAN_network_hidden_small_sizes
        sian_training_args.model_config.is_masked = is_masked_sian
        sian_training_args.model_config.use_mnist_scaling = use_mnist_scaling
        sian_training_args.saving_settings.exp_folder = exp_folder
        sian_training_args.saving_settings.results_to_save = self.SIAN_saving_settings_results_to_save
        if is_masked_sian: #TODO - clean up, part of hypers
            halfstep = STEP // 2
            halfpow = int(np.log(halfstep)/np.log(2))
            masked_steps_to_evaluate = [0] + list(np.power(2,np.arange(100)))
            masked_steps_to_evaluate = [0] + list(np.power(2,np.arange(halfpow+1))) + list(halfstep + np.power(2,np.arange(halfpow+1)))
            print('masked_steps_to_evaluate',masked_steps_to_evaluate)
            sian_training_args.return_settings.steps_to_evaluate = masked_steps_to_evaluate
        if True:
            sian_training_args.lambda1 = lambda1
        if "compute_shapeloss" in full_experiment_details:
            sian_training_args.compute_shapeloss = full_experiment_details['compute_shapeloss']

        D = dataset_obj.get_D()
        readable_labels = dataset_obj.get_readable_labels()
        print(readable_labels)


        tau_values_list = []
        theta_values_list = []
        if FIS_style == "layerwise":
            if "tau_values" in full_experiment_details and "theta_values" in full_experiment_details:
                for k in range(1, MAX_K + 1):
                    tau_vals = full_experiment_details["tau_values"]
                    theta_vals = full_experiment_details["theta_values"]
                    tau_values_list.append(tau_vals if tau_vals is not None else [self.default_tau_values.get(k, 1.0)])
                    theta_values_list.append(theta_vals if theta_vals is not None else [self.default_theta_values.get(k, 1.0)])
            else:
                # always have tau and theta values in the json - Rahil
                pass
            hyperparam_combinations = list(itertools.product(*tau_values_list, *theta_values_list))
        elif FIS_style == "batchwise":
            if "tau_values" in full_experiment_details:
                for k in range(1, MAX_K + 1):
                    tau_vals = full_experiment_details["tau_values"]
                    tau_values_list.append(tau_vals if tau_vals is not None else [self.default_tau_values.get(k, 1.0)])
            else:
                # always have tau values in the json - Rahil
                pass
            number_of_rounds_list = [full_experiment_details.get("number_of_rounds", 3)]
            inters_per_round_list = [full_experiment_details.get("inters_per_round", 1)]
            hyperparam_combinations = list(itertools.product(*tau_values_list, number_of_rounds_list, inters_per_round_list))
        else:
            hyperparam_combinations = [None]

        print(hyperparam_combinations)
        print(len(hyperparam_combinations))

        BASE_TRN_N = 700 * 1000

        for TARGET_TRN_N in target_sample_sizes:
            print(10 * "-", f"Processing sample size: {TARGET_TRN_N}", 10 * "-")
            trn_reduc_perc = TARGET_TRN_N / BASE_TRN_N
            mlp_training_args.trainval_reduction_percentage = trn_reduc_perc
            sian_training_args.trainval_reduction_percentage = trn_reduc_perc

            mlp_training_args.model_config.model_init_seed = model_init_seed
            sian_training_args.model_config.model_init_seed = model_init_seed

            are_we_traning_the_mlp = (FIS_style in ["layerwise", "batchwise"])     
            explainer_score_type = "arch"   
            jam_arch, fis_valX, mlp_results = prepare_for_FIS(are_we_traning_the_mlp, dataset_obj, mlp_training_args, explainer_score_type)


            for hyperparam_idx, hyperparam_combo in enumerate(hyperparam_combinations):
                datetimestr = gettimestamp()
                print(f"Processing hyperparameter combination {hyperparam_idx+1}/{len(hyperparam_combinations)} for sample size {TARGET_TRN_N}")
                #''' TODO: remove later, only needed for saving
                if FIS_style == "layerwise":
                    tau_values = hyperparam_combo[:MAX_K]
                    theta_values = hyperparam_combo[MAX_K:]
                    tau_thresholds = {k+1: tau_values[k] for k in range(min(MAX_K, len(tau_values)))}
                    theta_thresholds = {k+1: theta_values[k] for k in range(min(MAX_K, len(theta_values)))}
                elif FIS_style == "batchwise":
                    tau_values = hyperparam_combo
                    tau_thresholds = {k+1: tau_values[k] for k in range(min(MAX_K, len(tau_values)))}
                #'''
                FIS_interactions, FIS_algorithm_time_taken, FIS_other_stuff, saved_results_path = really_do_the_fis(FIS_style, MAX_K, hyperparam_combo, jam_arch, fis_valX, number_of_rounds, inters_per_round, explainer_score_type,    exp_folder, datetimestr, TARGET_TRN_N)
                print("FIS_algorithm_time_taken", FIS_algorithm_time_taken)
                print("FIS_interactions")
                print(FIS_interactions)
                
                sian_training_args.model_config.model_init_seed = model_init_seed
                print(f"Model init seed: {model_init_seed}")

                print(10 * "=", f"Training SIAN-{MAX_K} with sample size {TARGET_TRN_N}", 10 * "=")

                sian_training_args.model_config.FIS_interactions = FIS_interactions
                sian_results = train_sian_final(dataset_obj, sian_training_args)
                trained_sian = sian_results["best_net"]
                val_tensor = sian_results["val_tensor"]
                print(f"SIAN Results: {sian_results}")

                gpu_details = get_device_details(device)
                if "final_test_mse" in sian_results:
                    test_metric_value = round(float(sian_results["final_test_mse"]["mse"]), 5)

                epoch_data = sian_results.get("epoch_data", {})
                if len(epoch_data) > 0:
                    last_epoch_evaluated = max(epoch_data.keys())
                    train_mse = round(float(epoch_data[last_epoch_evaluated]["trn_metric"]), 5)
                    val_mse = round(float(epoch_data[last_epoch_evaluated]["val_metric"]), 5)
                else:
                    last_epoch_evaluated = None
                    train_mse = None
                    val_mse = None

                experiment_data = {
                    "exp_folder": exp_folder,
                    "dataset_str": dataset_str,
                    "batch_size": BS,
                    "epochs": EP,
                    "steps": STEP,
                    "learning_rate": LR,
                    "preproc_owner": preproc_owner,
                    "FIS_style": FIS_style,
                    "is_masked_mlp": is_masked_mlp,
                    "is_masked_sian": is_masked_sian,
                    "use_mnist_scaling": use_mnist_scaling,
                    "model_init_seed": model_init_seed,
                    "MAX_K": MAX_K,
                    "TARGET_TRN_N": int(TARGET_TRN_N),
                    "number_of_rounds": number_of_rounds if FIS_style == "batchwise" else None,
                    "inters_per_round": inters_per_round if FIS_style == "batchwise" else None,
                    "tau_thresholds": tau_thresholds if (FIS_style in ["batchwise","layerwise"]) else None,
                    "theta_thresholds": theta_thresholds if FIS_style == "layerwise" else None,
                    "FIS_algorithm_time_taken": FIS_algorithm_time_taken,
                    "FIS_interactions": FIS_interactions,

                    "MLP_total_training_time": round(mlp_results["total_training_time"], 2),
                    "mlp_results.step_data" : mlp_results["step_data"],
                    "mlp_results.epoch_data" : mlp_results["epoch_data"],
                    "sian_results.step_data" : sian_results["step_data"],
                    "sian_results.epoch_data" : sian_results["epoch_data"],

                    "SIAN_total_training_time": round(sian_results["total_training_time"], 2),
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                    "final_test_mse.mse": round(float(sian_results["final_test_mse"]["mse"]), 5),
                    "GPU_name": gpu_details["gpu_name"]
                }
                print(experiment_data)
                with open(saved_results_path, "w") as f:
                    json.dump(experiment_data, f, indent=4)
                print("JSON results saved successfully")




class RealWorldDatasetExperimentMixin(ABC):
    def run_realworld_experiment(self, full_experiment_details, exp_folder):
        _, _, _, model_init_seed = self.collect_the_seeds(full_experiment_details)
        full_experiment_details["model_init_seed"] = model_init_seed
        data_seed = model_init_seed
        print(f"Using model_init_seed={model_init_seed} and data_seed={data_seed}")

        dataset_str = full_experiment_details["dataset_str"]
        preproc_owner = full_experiment_details["preproc_owner"] if full_experiment_details["preproc_owner"] != "None" else None
        is_masked_mlp = full_experiment_details["is_masked_mlp"]
        is_masked_sian = full_experiment_details["is_masked_sian"]
        use_mnist_scaling = full_experiment_details["use_mnist_scaling"]
        hyperparameter_combo_list = full_experiment_details["hyperparameter_combo_list"]

        BS, LR, EP, STEP = self.get_basic_training_details(full_experiment_details)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp_training_args = TrainingArgs(BS, EP, LR, device, STEP)
        sian_training_args = TrainingArgs(BS, EP, LR, device, STEP)

        dataset_obj, _, _, _ = self.setup_dataset(full_experiment_details)
        self.shuffle_dataset_object(dataset_obj) #NOTE: probably unnecessary - trnval not needed yet
        BASE_TRN_N = dataset_obj.get_N()
        target_sample_sizes = full_experiment_details.get("target_sample_sizes", [BASE_TRN_N])
        target_sample_sizes = [dataset_obj.get_N() if str(target_sample_size) == "get_N" else int(target_sample_size) for target_sample_size in target_sample_sizes]
        print("target_sample_sizes", target_sample_sizes)

        mlp_training_args.model_config.net_name = "MLP"
        mlp_training_args.model_config.sizes = self.MLP_network_hidden_sizes
        mlp_training_args.model_config.is_masked = is_masked_mlp
        mlp_training_args.model_config.use_mnist_scaling = use_mnist_scaling
        mlp_training_args.saving_settings.exp_folder = exp_folder + f"model_seed{model_init_seed}_"
        mlp_training_args.saving_settings.results_to_save = self.MLP_saving_settings_results_to_save

        sian_training_args.model_config.net_name = "SIAN-K"
        sian_training_args.model_config.sizes = self.SIAN_network_hidden_sizes
        sian_training_args.model_config.small_sizes = self.SIAN_network_hidden_small_sizes
        sian_training_args.model_config.is_masked = is_masked_sian
        sian_training_args.model_config.use_mnist_scaling = use_mnist_scaling
        sian_training_args.saving_settings.exp_folder = exp_folder + f"model_seed{model_init_seed}_"
        sian_training_args.saving_settings.results_to_save = self.SIAN_saving_settings_results_to_save

        if is_masked_sian and STEP is not None:
            halfstep = STEP // 2
            halfpow = int(np.log2(halfstep))
            masked_steps = [0] + [2**i for i in range(halfpow + 1)] + [halfstep + 2**i for i in range(halfpow + 1)]
            sian_training_args.return_settings.steps_to_evaluate = masked_steps

        if "lambda1" in full_experiment_details:
            sian_training_args.lambda1 = full_experiment_details["lambda1"]
        if "compute_shapeloss" in full_experiment_details:
            sian_training_args.compute_shapeloss = full_experiment_details["compute_shapeloss"]

        for TARGET_TRN_N in target_sample_sizes:
            print(10 * "-", f"Processing sample size: {TARGET_TRN_N}", 10 * "-")
            trn_reduc_perc = TARGET_TRN_N / BASE_TRN_N
            mlp_training_args.trainval_reduction_percentage = trn_reduc_perc
            sian_training_args.trainval_reduction_percentage = trn_reduc_perc

            dataset_obj, _, _, _ = self.setup_dataset(full_experiment_details)
            self.shuffle_dataset_object(dataset_obj, trn_reduc_perc=trn_reduc_perc)

            mlp_training_args.model_config.model_init_seed = model_init_seed
            mlp_results = train_mlp_final(dataset_obj, mlp_training_args)
            mlp_best_net = mlp_results["best_net"]
            mlp_location = mlp_training_args.saving_settings.results_save_prefix + "MLP_final_net.pt"

            val_tensor = dataset_obj.pull_val_tensor(device)
            fis_valX = val_tensor if is_masked_mlp else val_tensor.detach().cpu().numpy()

            for hyperparam_idx, hyperparam_combo in enumerate(hyperparameter_combo_list):
                datetimestr = gettimestamp()
                FIS_style = hyperparam_combo["FIS_style"]
                MAX_K = hyperparam_combo["MAX_K"]
                explainer_score_type = hyperparam_combo["explainer_score_type"]

                jam_arch = prepare_for_FIS_v2(mlp_best_net, is_masked_mlp, dataset_obj, explainer_score_type)

                if FIS_style == "layerwise":
                    tau_thresholds = convert_dict_keys_to_int(hyperparam_combo["tau_thresholds"])
                    theta_thresholds = convert_dict_keys_to_int(hyperparam_combo["theta_thresholds"])
                    number_of_rounds = inters_per_round = None
                elif FIS_style == "batchwise":
                    tau_thresholds = convert_dict_keys_to_int(hyperparam_combo["tau_thresholds"])
                    number_of_rounds = hyperparam_combo["number_of_rounds"]
                    inters_per_round = hyperparam_combo["inters_per_round"]
                    theta_thresholds = None
                else:
                    raise ValueError(f"Invalid FIS_style: {FIS_style}")

                FIS_interactions, FIS_time, FIS_other, fis_path = really_do_the_fis(
                    FIS_style, MAX_K, hyperparam_combo, jam_arch, fis_valX,
                    number_of_rounds, inters_per_round, explainer_score_type,
                    exp_folder, datetimestr, TARGET_TRN_N
                )

                sian_training_args.model_config.model_init_seed = model_init_seed
                sian_training_args.model_config.FIS_interactions = FIS_interactions
                sian_results = train_sian_final(dataset_obj, sian_training_args)

                gpu_details = self.get_device_details()
                final_test_mse = sian_results.get("final_test_mse", {})
                test_mse_val = round(float(final_test_mse.get("mse", 0)), 5) if final_test_mse else None

                epoch_data = sian_results.get("epoch_data", {})
                last_epoch = max(epoch_data.keys()) if epoch_data else None
                train_mse = round(float(epoch_data[last_epoch]["trn_metric"]), 5) if last_epoch else None
                val_mse = round(float(epoch_data[last_epoch]["val_metric"]), 5) if last_epoch else None

                experiment_data = {}
                gpu_details = self.get_device_details()
                self.add_base_experiment_data(experiment_data, dataset_str, preproc_owner, model_init_seed, data_seed, gpu_details)

                self.add_training_params(experiment_data, mlp_training_args)
                self.add_model_config_params(experiment_data, mlp_training_args.model_config, prefix="mlp")
                self.add_model_config_params(experiment_data, sian_training_args.model_config, prefix="sian")
                self.add_fis_params(experiment_data, FIS_style, MAX_K, number_of_rounds, inters_per_round, tau_thresholds, theta_thresholds, explainer_score_type)

                experiment_data.update({
                    "TARGET_TRN_N": int(TARGET_TRN_N),
                    "MLP_total_training_time": round(mlp_results["total_training_time"], 5),
                    "SIAN_total_training_time": round(sian_results["total_training_time"], 5),
                    "FIS_algorithm_time_taken": FIS_time,
                    "FIS_interactions": FIS_interactions,
                    "mlp_location": mlp_location,
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                    "final_test_mse.mse": test_mse_val,
                    "mlp_results.step_data": mlp_results["step_data"],
                    "mlp_results.epoch_data": mlp_results["epoch_data"],
                    "sian_results.step_data": sian_results["step_data"],
                    "sian_results.epoch_data": sian_results["epoch_data"],
                })

                suffix = f"SIAN_K{MAX_K}_FIS{FIS_style}_hyp{hyperparam_idx}"
                saved_path = self.save_experiment_data(exp_folder, TARGET_TRN_N, experiment_data, suffix=suffix)
                print(f"Full pipeline saved: {saved_path}")




class InflationExperimentMixin(ABC):
    def run_inflation_experiment(self, full_experiment_details, exp_folder):
        _, _, _, model_init_seed = self.collect_the_seeds(full_experiment_details)
        full_experiment_details["model_init_seed"] = model_init_seed
        data_seed = model_init_seed
        print(f"Using model_init_seed={model_init_seed} and data_seed={data_seed}")

        BS, LR, EP, STEP = self.get_basic_training_details(full_experiment_details)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        mlp_training_args = TrainingArgs(BS, EP, LR, device, STEP)
        sian_training_args = TrainingArgs(BS, EP, LR, device, STEP)
        
        dataset_obj, _, _, preproc_owner = self.setup_dataset(full_experiment_details)
        self.shuffle_dataset_object(dataset_obj) #NOTE: probably unnecessary - trnval not needed yet

        is_masked_mlp = full_experiment_details["is_masked_mlp"] 
        is_masked_sian = full_experiment_details["is_masked_sian"] 
        use_mnist_scaling = full_experiment_details["use_mnist_scaling"]
        FIS_style = full_experiment_details["FIS_style"]
        MAX_K = full_experiment_details["MAX_K"]
        number_of_rounds = full_experiment_details["number_of_rounds"]
        inters_per_round = full_experiment_details["inters_per_round"]
        dataset_str = full_experiment_details["dataset_str"]

        if "lambda1" in full_experiment_details:
            lambda1 = full_experiment_details["lambda1"]
        else:
            lambda1 = sian_training_args.lambda1

        print("Dataset used:", dataset_str)
        print("Preprocessing Owner:", preproc_owner)
        if FIS_style == "batchwise":
            print("Number of rounds:", number_of_rounds)
            print("Interactions per round:", inters_per_round)

        exp_folder = self.setup_experiment_folder(dataset_str)
        print(exp_folder)

        target_sample_sizes = full_experiment_details["target_sample_sizes"]
        target_sample_sizes = [dataset_obj.get_N() if str(target_sample_size) == "get_N" else target_sample_size for target_sample_size in target_sample_sizes]
        target_sample_sizes = [int(target_sample_size) for target_sample_size in target_sample_sizes]

        print('target_sample_sizes', target_sample_sizes)

        if "max_number_of_interactions" in full_experiment_details:
            max_number_of_interactions = full_experiment_details["max_number_of_interactions"]
        else:
            max_number_of_interactions = 10*1000
        print('max_number_of_interactions', max_number_of_interactions)

        mlp_training_args.model_config.net_name = "MLP"
        mlp_training_args.model_config.sizes = self.MLP_network_hidden_sizes
        mlp_training_args.model_config.is_masked = is_masked_mlp
        mlp_training_args.model_config.use_mnist_scaling = use_mnist_scaling
        mlp_training_args.saving_settings.exp_folder = exp_folder
        mlp_training_args.saving_settings.results_to_save = self.MLP_saving_settings_results_to_save
        if True: #TODO - clean up, part of hypers
        # if False: #09/18/25 @ 5:00pm -- turning off to allow CNN
            mlp_training_args.return_settings.epochs_to_evaluate = []
            mlp_training_args.return_settings.steps_to_evaluate = []

        sian_training_args.model_config.net_name = "SIAN-K"
        sian_training_args.model_config.sizes = self.SIAN_network_hidden_sizes
        sian_training_args.model_config.small_sizes = self.SIAN_network_hidden_small_sizes
        sian_training_args.model_config.is_masked = is_masked_sian
        sian_training_args.model_config.use_mnist_scaling = use_mnist_scaling
        sian_training_args.saving_settings.exp_folder = exp_folder
        sian_training_args.saving_settings.results_to_save = self.SIAN_saving_settings_results_to_save
        if is_masked_sian: #TODO - clean up, part of hypers
            halfstep = STEP // 2
            halfpow = int(np.log(halfstep)/np.log(2))
            masked_steps_to_evaluate = [0] + list(np.power(2,np.arange(100)))
            masked_steps_to_evaluate = [0] + list(np.power(2,np.arange(halfpow+1))) + list(halfstep + np.power(2,np.arange(halfpow+1)))
            print('masked_steps_to_evaluate',masked_steps_to_evaluate)
            sian_training_args.return_settings.steps_to_evaluate = masked_steps_to_evaluate
        if True:
            sian_training_args.lambda1 = lambda1
        if "compute_shapeloss" in full_experiment_details:
            sian_training_args.compute_shapeloss = full_experiment_details['compute_shapeloss']
            
            
        if "max_part_size" in full_experiment_details:
            max_part_size = full_experiment_details['max_part_size']
            sian_training_args.model_config.max_inflation_amount = max_part_size
        else:
            raise Exception("no default max_part_size right now")
            sian_training_args.model_config.max_inflation_amount = 64
            

        if True:
            sian_training_args.verbosity_settings.VERBOSE_TRAINING = False

            # mlp_training_args.verbosity_settings.VERBOSE_TRAINING = False
            # sian_training_args.number_of_steps = 0

        if True:
            #09/18/25 @ 8:30pm -- turning off slow test evaluation
            sian_training_args.return_settings.epochs_to_evaluate = []
            sian_training_args.return_settings.steps_to_evaluate = []

        D = dataset_obj.get_D()
        C = dataset_obj.get_C() #desktop
        print('D',D,'C',C) #desktop
        readable_labels = dataset_obj.get_readable_labels()
        # print(readable_labels)

        tau_values_list = []
        theta_values_list = []
        if FIS_style == "layerwise":
            if "tau_values" in full_experiment_details and "theta_values" in full_experiment_details:
                for k in range(1, MAX_K + 1):
                    tau_vals = full_experiment_details["tau_values"].get(k, [self.default_tau_values.get(k, 1.0)])
                    theta_vals = full_experiment_details["theta_values"].get(k, [self.default_theta_perc_values.get(k, 0.5)])
                    tau_values_list.append(tau_vals if tau_vals is not None else [self.default_tau_values.get(k, 1.0)])
                    theta_values_list.append(theta_vals if theta_vals is not None else [self.default_theta_perc_values.get(k, 0.5)])
            else:
                for k in range(1, MAX_K + 1):
                    tau_values_list.append([self.default_tau_values.get(k, 1.0)])
                    theta_values_list.append([self.default_theta_perc_values.get(k, 0.5)])
            hyperparam_combinations = list(itertools.product(*tau_values_list, *theta_values_list))
        elif FIS_style == "batchwise":
            if "tau_values" in full_experiment_details:
                for k in range(1, MAX_K + 1):
                    tau_vals = full_experiment_details["tau_values"].get(k, [self.default_tau_values.get(k, 1.0)])
                    tau_values_list.append(tau_vals if tau_vals is not None else [self.default_tau_values.get(k, 1.0)])
            else:
                for k in range(1, MAX_K + 1):
                    tau_values_list.append([self.default_tau_values.get(k, 1.0)])
            number_of_rounds_list = [full_experiment_details.get("number_of_rounds", 3)]
            inters_per_round_list = [full_experiment_details.get("inters_per_round", 1)]
            hyperparam_combinations = list(itertools.product(*tau_values_list, number_of_rounds_list, inters_per_round_list))
        else:
            hyperparam_combinations = [None]

        print(hyperparam_combinations)
        print(len(hyperparam_combinations))

        for TARGET_TRN_N in target_sample_sizes:
            dataset_obj, _, _, preproc_owner = self.setup_dataset(full_experiment_details)
            self.shuffle_dataset_object(dataset_obj) #NOTE: probably unnecessary - trnval not needed yet
            print(10 * "-", f"Processing sample size: {TARGET_TRN_N}", 10 * "-")
            # BASE_TRN_N = 700 * 1000 #TODO
            BASE_TRN_N = dataset_obj.get_N()
            print('BASE_TRN_N',BASE_TRN_N)
            trn_reduc_perc = TARGET_TRN_N / BASE_TRN_N
            mlp_training_args.trainval_reduction_percentage = trn_reduc_perc
            sian_training_args.trainval_reduction_percentage = trn_reduc_perc
            
        
            if True: #desktop, mnist
                import types

                def custom_get_grouped_feature_dict(self):
                    return self.grouped_features_dict

                mnist_grouped_features_dict = {}
                patch_size = 4
                mnist_grouped_features_dict["D0"] = 49
                mnist_grouped_features_dict["D"]  = 784
                mnist_grouped_features_dict["C"]  = 10
                for ii in range(49):
                    mnist_grouped_features_dict[ii] = []
                H = 28
                W = 28
                WW = 7
                for y in range(H):
                    for x in range(W):
                        yy=y//patch_size
                        xx=x//patch_size
                        # print(y,x,'\t',yy,xx)
                        i  =  y * W + x
                        ii = yy * WW + xx
                        mnist_grouped_features_dict[ii].append(  i  )

                dataset_obj.label_stuff_dict["full_readable_labels"]["D0"] = 49
                dataset_obj.grouped_features_dict = mnist_grouped_features_dict
                dataset_obj.get_grouped_feature_dict = types.MethodType(custom_get_grouped_feature_dict, dataset_obj)
            
            # sian_training_args.model_config.model_init_seed = model_init_seed #NOTE: JAM -- isnt this also necessary to set? even if we reuse it?



            are_we_traning_the_mlp = (FIS_style in ["layerwise", "batchwise"])  
            are_we_traning_the_mlp = True #train the CNN
            are_we_traning_the_mlp = False #dont train the CNN, train gam2x2longrange
            explainer_score_type = "arch"   
            jam_arch, fis_valX, mlp_results = prepare_for_FIS(are_we_traning_the_mlp, dataset_obj, mlp_training_args, explainer_score_type)


            for hyperparam_idx, hyperparam_combo in enumerate(hyperparam_combinations):
                datetimestr = gettimestamp()
                print(f"Processing hyperparameter combination {hyperparam_idx+1}/{len(hyperparam_combinations)} for sample size {TARGET_TRN_N}")
                #''' TODO: remove later, only needed for saving
                if FIS_style == "layerwise":
                    tau_values = hyperparam_combo[:MAX_K]
                    theta_values = hyperparam_combo[MAX_K:]
                    tau_thresholds = {k+1: tau_values[k] for k in range(min(MAX_K, len(tau_values)))}
                    theta_thresholds = {k+1: theta_values[k] for k in range(min(MAX_K, len(theta_values)))}
                    number_of_rounds = None
                    inters_per_round = None
                elif FIS_style == "batchwise":
                    tau_values = hyperparam_combo
                    tau_thresholds = {k+1: tau_values[k] for k in range(min(MAX_K, len(tau_values)))}
                    number_of_rounds = hyperparam_combo["number_of_rounds"]
                    inters_per_round = hyperparam_combo["inters_per_round"]
                    theta_thresholds = None
                #'''
                else: # For FIS_style = "maximal"
                    tau_thresholds = None
                    theta_thresholds = None
                    number_of_rounds = None
                    inters_per_round = None

                FIS_interactions, FIS_algorithm_time_taken, FIS_other_stuff, saved_results_path = really_do_the_fis(FIS_style, MAX_K, hyperparam_combo, jam_arch, fis_valX, number_of_rounds, inters_per_round, explainer_score_type,    exp_folder, datetimestr, TARGET_TRN_N)
                print("FIS_algorithm_time_taken", FIS_algorithm_time_taken)
                print("FIS_interactions")
                print(FIS_interactions)
                
                sian_training_args.model_config.model_init_seed = model_init_seed
                print(f"Model init seed: {model_init_seed}")

                print(10 * "=", f"Training SIAN-{MAX_K} with sample size {TARGET_TRN_N}", 10 * "=")

                # sian_training_args.model_config.FIS_interactions = FIS_interactions
                sian_training_args.model_config.FIS_interactions = FIS_interactions[:max_number_of_interactions]
                
                sian_results = train_sian_final(dataset_obj, sian_training_args)
                trained_sian = sian_results["best_net"]
                val_tensor = sian_results["val_tensor"]
                print(f"SIAN Results: {sian_results}")

                experiment_data = {}
                gpu_details = self.get_device_details()

                self.add_base_experiment_data(experiment_data, dataset_str, preproc_owner, model_init_seed, data_seed, gpu_details)

                self.add_training_params(experiment_data, mlp_training_args)
                self.add_model_config_params(experiment_data, mlp_training_args.model_config, prefix="mlp")
                self.add_model_config_params(experiment_data, sian_training_args.model_config, prefix="sian")
                self.add_fis_params(experiment_data, FIS_style, MAX_K, number_of_rounds, inters_per_round, tau_thresholds, theta_thresholds, explainer_score_type)

                experiment_data.update({
                    "exp_folder": exp_folder,
                    "FIS_style": FIS_style,
                    "TARGET_TRN_N": int(TARGET_TRN_N),
                    "FIS_interactions": FIS_interactions,
                    "max_part_size": max_part_size,

                    "mlp_results.step_data" : mlp_results["step_data"],
                    "mlp_results.epoch_data" : mlp_results["epoch_data"],
                    "sian_results.step_data" : sian_results["step_data"],
                    "sian_results.epoch_data" : sian_results["epoch_data"],

                    "MLP_total_training_time": round(mlp_results["total_training_time"], 2),
                    "SIAN_total_training_time": round(sian_results["total_training_time"], 4),
                    "SIAN.gradient_training_time_taken": round(sian_results["gradient_training_time_taken"], 4),
                    "SIAN.metric_evaluation_time_taken": round(sian_results["metric_evaluation_time_taken"], 4),

                    "GPU_name": gpu_details["gpu_name"]
                })

                print(experiment_data)
                with open(saved_results_path, "w") as f:
                    json.dump(experiment_data, f, indent=4)
                print("JSON results saved successfully")








