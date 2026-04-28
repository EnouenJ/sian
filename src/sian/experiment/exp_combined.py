import os
import json
import torch
from .experiment import *
from sian.utils import convert_dict_keys_to_int



class CombinedExperiment(Experiment):
    def __init__(self):
        super().__init__()


        #### DEFAULT SETTINGS ###
        self.default_target_sample_sizes = [100, 800, 6400]
        self.MLP_saving_settings_results_to_save = ['best_net', 'final_net']
        self.SIAN_saving_settings_results_to_save = ['best_net', 'final_net']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    def setup_dataset(self, full_experiment_details, trn_reduc_perc=1.0):
        data_base_path = full_experiment_details["data_base_path"]
        dataset_str = full_experiment_details["dataset_str"]
        preproc_owner = full_experiment_details["preproc_owner"] if full_experiment_details["preproc_owner"] != "None" else None
        load_dataset_path = data_base_path
        save_dataset_path = data_base_path + dataset_str + "/"
        
        dataset_obj = Final_TabularDataset(
            dataset_str,
            preproc_owner=preproc_owner,
            load_dataset_path=load_dataset_path,
            save_dataset_path=save_dataset_path
        )
        dataset_obj.shuffle_and_split_trnval(trnval_shuffle_seed=0, trnval_split_percentage=0.7, trnval_reduc_percentage=trn_reduc_perc)
        return dataset_obj, load_dataset_path, save_dataset_path, preproc_owner

    def get_target_sample_sizes(self, full_experiment_details, dataset_obj=None):
        type_of_experiment = full_experiment_details.get("type_of_experiment")
        if "target_sample_sizes" in full_experiment_details:
            return full_experiment_details["target_sample_sizes"]
        elif type_of_experiment in ["real_world_benchmark", "train_sian"]:
            return [dataset_obj.get_N()] if dataset_obj else self.default_target_sample_sizes
        else:
            return self.default_target_sample_sizes





    def run_compare_fis(self, full_experiment_details):
        print(10 * "=", "Running Compare FIS Experiment", 10 * "=")
        dataset_str = full_experiment_details["dataset_str"]
        data_seed = int(dataset_str.split("_")[-1].replace("seed", ""))
        model_init_seed = data_seed
        exp_folder = self.setup_experiment_folder(dataset_str)
        full_experiment_details["model_init_seed"] = model_init_seed

        BS, LR, EP, STEP = self.get_basic_training_details(full_experiment_details)
        mlp_training_args = self.get_training_configuration(full_experiment_details, exp_folder, type_of_model="MLP")
        mlp_training_args.batch_size = BS
        mlp_training_args.number_of_epochs = EP
        mlp_training_args.learning_rate = LR
        mlp_training_args.number_of_steps = STEP
        mlp_training_args.device = self.device
        
        is_masked_mlp = full_experiment_details["is_masked_mlp"]
        use_mnist_scaling = full_experiment_details["use_mnist_scaling"]
        hyperparameter_combo_list = full_experiment_details["hyperparameter_combo_list"]
        
        
        mlp_training_args.saving_settings.exp_folder = exp_folder
        mlp_training_args.saving_settings.results_to_save = self.MLP_saving_settings_results_to_save
        mlp_training_args.model_config.net_name = "MLP"
        mlp_training_args.model_config.sizes = self.MLP_network_hidden_sizes
        mlp_training_args.model_config.is_masked = is_masked_mlp
        mlp_training_args.model_config.use_mnist_scaling = use_mnist_scaling

        if "lambda1" in full_experiment_details:
            mlp_training_args.lambda1 = full_experiment_details["lambda1"]
        if "lambda2" in full_experiment_details:
            mlp_training_args.lambda2 = full_experiment_details["lambda2"]

        dataset_obj, load_dataset_path, save_dataset_path, preproc_owner = self.setup_dataset(full_experiment_details)
        self.shuffle_dataset_object(dataset_obj) #NOTE: probably unnecessary - trnval not needed yet
        D = dataset_obj.get_D()
        readable_labels = dataset_obj.get_readable_labels()
        print(readable_labels)

        

        mlp_training_args.model_init_seed = model_init_seed
        mlp_training_args.model_config.model_init_seed = model_init_seed

        target_sample_sizes = self.get_target_sample_sizes(full_experiment_details, dataset_obj)
        print('target_sample_sizes', target_sample_sizes)

        gpu_details = self.get_device_details()

        for TARGET_TRN_N in target_sample_sizes:
            print(10 * "-", f"Processing sample size: {TARGET_TRN_N}", 10 * "-")
            BASE_TRN_N = dataset_obj.get_N()
            trn_reduc_perc = TARGET_TRN_N / BASE_TRN_N
            dataset_obj, _, _, _ = self.setup_dataset(full_experiment_details)
            self.shuffle_dataset_object(dataset_obj, trn_reduc_perc)
            mlp_training_args.trainval_reduction_percentage = trn_reduc_perc
            
            are_we_traning_the_mlp = True
            explainer_score_type = hyperparameter_combo_list[0]["explainer_score_type"]
            jam_arch, fis_valX, mlp_results = prepare_for_FIS(are_we_traning_the_mlp, dataset_obj, mlp_training_args, explainer_score_type)

            for hyperparam_idx, hyperparam_combo in enumerate(hyperparameter_combo_list):
                FIS_style = hyperparam_combo["FIS_style"]
                MAX_K = hyperparam_combo["MAX_K"]
                explainer_score_type = hyperparam_combo["explainer_score_type"]
                jam_arch.score_type_name = return_score_type_name(is_masked_mlp, explainer_score_type)

                print(10 * "=", f"Processing hyperparameter combination {hyperparam_idx+1}/{len(hyperparameter_combo_list)} for sample size {TARGET_TRN_N}", 10 * "=")
                
                if FIS_style == "layerwise":
                    tau_thresholds = hyperparam_combo["tau_thresholds"]
                    theta_thresholds = hyperparam_combo["theta_thresholds"]
                    number_of_rounds = None
                    inters_per_round = None
                elif FIS_style == "batchwise":
                    tau_thresholds = hyperparam_combo["tau_thresholds"]
                    theta_thresholds = None
                    number_of_rounds = hyperparam_combo["number_of_rounds"]
                    inters_per_round = hyperparam_combo["inters_per_round"]

                FIS_interactions, FIS_algorithm_time_taken, FIS_other_stuff, _ = really_do_the_fis(
                    FIS_style, MAX_K, hyperparam_combo, jam_arch, fis_valX, number_of_rounds, inters_per_round, explainer_score_type, exp_folder, gettimestamp(), TARGET_TRN_N
                )
                print("FIS_algorithm_time_taken", FIS_algorithm_time_taken)
                print("FIS_interactions", FIS_interactions)

                experiment_data = {}
                experiment_data['exp_folder'] = exp_folder
                self.add_base_experiment_data(experiment_data, dataset_str, preproc_owner, model_init_seed, data_seed, gpu_details)
                self.add_training_params(experiment_data, mlp_training_args)
                self.add_model_config_params(experiment_data, mlp_training_args.model_config)
                self.add_fis_params(experiment_data, FIS_style, MAX_K, number_of_rounds, inters_per_round, tau_thresholds, theta_thresholds, explainer_score_type)
                experiment_data["TARGET_TRN_N"] = int(TARGET_TRN_N)
                experiment_data["FIS_algorithm_time_taken"] = FIS_algorithm_time_taken
                experiment_data["FIS_interactions"] = FIS_interactions
                experiment_data["MLP_total_training_time"] = round(mlp_results["total_training_time"], 2)
                experiment_data["mlp_results.step_data"] = mlp_results["step_data"]
                experiment_data["mlp_results.epoch_data"] = mlp_results["epoch_data"]

                print(experiment_data)
                self.save_experiment_data(exp_folder, TARGET_TRN_N, experiment_data, suffix=f"K{MAX_K}_results")


    def run_train_mlp(self, full_experiment_details):
        print(10 * "=", "Running MLP Only Experiment", 10 * "=")
        _, _, _, model_init_seed = self.collect_the_seeds(full_experiment_details)
        full_experiment_details["model_init_seed"] = model_init_seed
        
        dataset_str = full_experiment_details["dataset_str"]
        exp_folder = self.setup_experiment_folder(dataset_str)
        mlp_training_args = self.get_training_configuration(full_experiment_details, exp_folder, type_of_model="MLP")
        mlp_training_args.model_init_seed = model_init_seed
        mlp_training_args.model_config.model_init_seed = model_init_seed
        data_seed = model_init_seed
        print(f"Using model_init_seed={model_init_seed} and data_seed={data_seed}")

        dataset_obj, load_dataset_path, save_dataset_path, preproc_owner = self.setup_dataset(full_experiment_details)
        self.shuffle_dataset_object(dataset_obj) #NOTE: probably unnecessary - trnval not needed yet
        D = dataset_obj.get_D()
        readable_labels = dataset_obj.get_readable_labels()
        print(readable_labels)

        target_sample_sizes = self.get_target_sample_sizes(full_experiment_details, dataset_obj)
        print('target_sample_sizes', target_sample_sizes)

        gpu_details = self.get_device_details()

        mlp_location_dict = {}
        for TARGET_TRN_N in target_sample_sizes:
            print(10 * "-", f"Processing sample size: {TARGET_TRN_N}", 10 * "-")
            BASE_TRN_N = dataset_obj.get_N()
            trn_reduc_perc = TARGET_TRN_N / BASE_TRN_N
            dataset_obj, _, _, _ = self.setup_dataset(full_experiment_details)
            self.shuffle_dataset_object(dataset_obj, trn_reduc_perc)
            mlp_training_args.trainval_reduction_percentage = trn_reduc_perc
            
            mlp_results = train_mlp_final(dataset_obj, mlp_training_args)

            experiment_data = {}
            experiment_data['exp_folder'] = exp_folder
            self.add_base_experiment_data(experiment_data, dataset_str, preproc_owner, model_init_seed, data_seed, gpu_details)
            self.add_training_params(experiment_data, mlp_training_args)
            self.add_model_config_params(experiment_data, mlp_training_args.model_config)
            experiment_data["TARGET_TRN_N"] = int(TARGET_TRN_N)
            experiment_data["MLP_total_training_time"] = round(mlp_results["total_training_time"], 2)
            experiment_data["mlp_results.step_data"] = mlp_results["step_data"]
            experiment_data["mlp_results.epoch_data"] = mlp_results["epoch_data"]
            mlp_location = mlp_training_args.saving_settings.results_save_prefix + "MLP_final_net.pt"
            experiment_data["mlp_location"] = mlp_location

            print(experiment_data)
            self.save_experiment_data(exp_folder, TARGET_TRN_N, experiment_data, suffix="MLP")

            mlp_location_dict[TARGET_TRN_N] = mlp_location
        return mlp_location_dict

    def run_fis_only(self, full_experiment_details):
        print(10 * "=", "Running FIS Only Experiment", 10 * "=")
        mlp_location_dict = full_experiment_details["mlp_location_dict"]
        is_masked_mlp = full_experiment_details["is_masked_mlp"]
        hyperparameter_combo_list = full_experiment_details["hyperparameter_combo_list"]
        dataset_str = full_experiment_details["dataset_str"]
        
        exp_folder = self.setup_experiment_folder(dataset_str)
        dataset_obj, load_dataset_path, save_dataset_path, preproc_owner = self.setup_dataset(full_experiment_details)
        self.shuffle_dataset_object(dataset_obj) #NOTE: probably unnecessary - trnval not needed yet
        _, _, _, model_init_seed = self.collect_the_seeds(full_experiment_details)
        data_seed = model_init_seed

        target_sample_sizes = self.get_target_sample_sizes(full_experiment_details, dataset_obj)
        print('target_sample_sizes', target_sample_sizes)

        gpu_details = self.get_device_details()
        full_interaction_dict = {}

        for TARGET_TRN_N in target_sample_sizes:
            full_interaction_dict[TARGET_TRN_N] = {}
            print(10 * "-", f"Processing sample size: {TARGET_TRN_N}", 10 * "-")
            mlp_location = mlp_location_dict[TARGET_TRN_N]
            mlp_best_net = torch.load(mlp_location, weights_only=False)
            print("loaded models successfully")

            BASE_TRN_N = dataset_obj.get_N()
            trn_reduc_perc = TARGET_TRN_N / BASE_TRN_N
            dataset_obj, _, _, _ = self.setup_dataset(full_experiment_details)
            self.shuffle_dataset_object(dataset_obj, trn_reduc_perc)
            
            val_tensor = dataset_obj.pull_val_tensor(self.device)
            fis_valX = val_tensor if is_masked_mlp else val_tensor.detach().cpu().numpy()

            for hyperparam_idx, hyperparam_combo in enumerate(hyperparameter_combo_list):
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

                FIS_interactions, FIS_algorithm_time_taken, FIS_other_stuff, _ = really_do_the_fis(
                    FIS_style, MAX_K, hyperparam_combo, jam_arch, fis_valX, number_of_rounds, inters_per_round, explainer_score_type, exp_folder, gettimestamp(), TARGET_TRN_N
                )
                print("FIS_algorithm_time_taken", FIS_algorithm_time_taken)
                print("FIS_interactions", FIS_interactions)

                batch_round_times_taken = FIS_other_stuff['times_per_round'] if FIS_style == 'batchwise' else None

                experiment_data = {}
                experiment_data['exp_folder'] = exp_folder
                self.add_base_experiment_data(experiment_data, dataset_str, preproc_owner, model_init_seed, data_seed, gpu_details)
                self.add_fis_params(experiment_data, FIS_style, MAX_K, number_of_rounds, inters_per_round, tau_thresholds, theta_thresholds, explainer_score_type)
                experiment_data["TARGET_TRN_N"] = int(TARGET_TRN_N)
                experiment_data["FIS_algorithm_time_taken"] = FIS_algorithm_time_taken
                experiment_data["FIS_interactions"] = FIS_interactions
                experiment_data["mlp_location"] = mlp_location
                experiment_data["batch_round_times_taken"] = batch_round_times_taken

                print(experiment_data)
                self.save_experiment_data(exp_folder, TARGET_TRN_N, experiment_data, suffix=f"K{MAX_K}_results")

                full_interaction_dict[TARGET_TRN_N][hyperparam_idx] = FIS_interactions
        return full_interaction_dict

    def train_and_save_sian(self, dataset_obj, sian_training_args, dataset_str, preproc_owner, load_dataset_path, save_dataset_path, fis_json_location, MAX_K, model_init_seed, type_of_experiment, TARGET_TRN_N, exp_folder):
        BASE_TRN_N = dataset_obj.get_N()
        trn_reduc_perc = TARGET_TRN_N / BASE_TRN_N
        dataset_obj, _, _, _ = self.setup_dataset({"data_base_path": load_dataset_path, "dataset_str": dataset_str, "preproc_owner": preproc_owner}, trn_reduc_perc) #TODO: why did rahil custom make the dataset details like this
        self.shuffle_dataset_object(dataset_obj) #NOTE: probably unnecessary - trnval not needed yet

        with open(fis_json_location, 'r') as f:
            json_data = json.load(f)

        correct_keys = []
        for thing in json_data:
            thing_dataset_str = json_data[thing]['dataset_str']
            thing_model_init_seed = json_data[thing]['model_init_seed']
            thing_TARGET_TRN_N = json_data[thing]['TARGET_TRN_N']
            thing_MAX_K = json_data[thing].get('MAX_K')
            if thing_dataset_str == dataset_str and thing_MAX_K == MAX_K and thing_model_init_seed == model_init_seed and (type_of_experiment in ["real_world_benchmark", "real_world_sweep"] or thing_TARGET_TRN_N == TARGET_TRN_N):
                correct_keys.append(thing)
        print('correct_keys', correct_keys)
        assert len(correct_keys) == 1
        correct_key = correct_keys[0]
        FIS_interactions = json_data[correct_key]["FIS_interactions"]
        print('loaded FIS_interactions', len(FIS_interactions))
        FIS_limit = 500
        FIS_interactions = FIS_interactions[:FIS_limit]
        print('trimmed to', len(FIS_interactions), FIS_interactions)

        sian_training_args.model_config.FIS_interactions = FIS_interactions
        print(10 * "=", f"Training SIAN-{MAX_K} with sample size {TARGET_TRN_N}", 10 * "=")

        sian_training_args.trainval_reduction_percentage = trn_reduc_perc
        sian_results = train_sian_final(dataset_obj, sian_training_args)
        print(f"SIAN Results: {sian_results}")

        gpu_details = self.get_device_details()
        experiment_data = {}
        experiment_data['exp_folder'] = exp_folder
        self.add_base_experiment_data(experiment_data, dataset_str, preproc_owner, model_init_seed, data_seed, gpu_details)
        self.add_training_params(experiment_data, sian_training_args)
        self.add_model_config_params(experiment_data, sian_training_args.model_config, prefix="sian")
        experiment_data["TARGET_TRN_N"] = int(TARGET_TRN_N)
        experiment_data["MAX_K"] = MAX_K
        experiment_data["FIS_interactions"] = FIS_interactions
        experiment_data["SIAN_total_training_time"] = round(sian_results["total_training_time"], 2)
        experiment_data["sian_results.step_data"] = sian_results["step_data"]
        experiment_data["sian_results.epoch_data"] = sian_results["epoch_data"]
        experiment_data["fis_json_location"] = fis_json_location
        sian_location = sian_training_args.saving_settings.results_save_prefix + "SIAN-K_final_net.pt"
        experiment_data["sian_location"] = sian_location

        print(experiment_data)
        self.save_experiment_data(exp_folder, TARGET_TRN_N, experiment_data, suffix="SIAN")

        return sian_location

    def run_sian(self, full_experiment_details):
        print(10 * "=", "Running SIAN Only Experiment", 10 * "=")
        _, _, _, model_init_seed = self.collect_the_seeds(full_experiment_details)
        full_experiment_details["model_init_seed"] = model_init_seed
        dataset_str = full_experiment_details["dataset_str"]
        exp_folder = self.setup_experiment_folder(dataset_str)
        
        sian_training_args = self.get_training_configuration(full_experiment_details, exp_folder, type_of_model="SIAN")
        sian_training_args.model_init_seed = model_init_seed
        sian_training_args.model_config.model_init_seed = model_init_seed
        data_seed = model_init_seed
        print(f"Using model_init_seed={model_init_seed} and data_seed={data_seed}")

        dataset_obj, load_dataset_path, save_dataset_path, preproc_owner = self.setup_dataset(full_experiment_details)
        self.shuffle_dataset_object(dataset_obj) #NOTE: probably unnecessary - trnval not needed yet        
        D = dataset_obj.get_D()
        readable_labels = dataset_obj.get_readable_labels()
        print(readable_labels)

        fis_json_location = full_experiment_details["fis_json_location"]
        MAX_K = full_experiment_details["MAX_K"]
        type_of_experiment = full_experiment_details["type_of_experiment"]
        
        target_sample_sizes = self.get_target_sample_sizes(full_experiment_details, dataset_obj)
        print('target_sample_sizes', target_sample_sizes)

        sian_locations = []
        for TARGET_TRN_N in target_sample_sizes:
            print(10 * "-", f"Processing sample size: {TARGET_TRN_N}", 10 * "-")
            sian_location = self.train_and_save_sian(
                dataset_obj, sian_training_args, dataset_str, preproc_owner, 
                load_dataset_path, save_dataset_path, fis_json_location, MAX_K, 
                model_init_seed, type_of_experiment, TARGET_TRN_N, exp_folder
            )
            sian_locations.append(sian_location)
        if len(sian_locations) == 1:
            return sian_locations[0]
        else:
            return sian_locations

    def run(self, full_experiment_details):
        # Commenting out for testing purposes right now. #TODO Raise not implemented error here - Rahil
        # raise NotImplementedError("not implemented downstream yet")

        type_of_experiment = full_experiment_details.get("type_of_experiment")
        if type_of_experiment == "compare_fis":
            self.run_compare_fis(full_experiment_details)
        elif type_of_experiment == "train_mlp_and_fis":
            mlp_location_dict = self.run_train_mlp(full_experiment_details)
            full_experiment_details["mlp_location_dict"] = mlp_location_dict
            return self.run_fis_only(full_experiment_details)
        elif type_of_experiment == "train_mlp":
            return self.run_train_mlp(full_experiment_details)
        elif type_of_experiment in ["real_world_benchmark", "real_world_sweep", "train_sian"]:
            return self.run_sian(full_experiment_details)
        else:
            raise ValueError(f"Unknown experiment_type: {type_of_experiment}. Expected one of: 'compare_fis', 'train_mlp_and_fis', 'train_mlp', 'real_world_benchmark', 'real_world_sweep', 'train_sian'")




    