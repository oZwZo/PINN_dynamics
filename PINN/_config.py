import os
import re
import numpy as np
import datetime
import json
from argparse import Namespace
from typing import Dict, Any

class ExperimentConfig:
    """Records and manages experiment configuration settings."""
    
    def __init__(self,  config: str=None,  args: Namespace = None, model = None):
        """
        Args:
            args: Parsed command-line arguments
            config: Experiment directory path
            model: Initialized model instance
        """
    

        if (args is not None) and (model is not None):
            self.run_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.experiment_config = self._get_experiment_config(args)
            self.dataset_config = self._get_dataset_config(args)
            self.model_config = self._get_model_config(model)
            self.training_config = self._get_training_config(args)
            self.raw_args = vars(args)

        elif os.path.exists(config) and config.endswith('.json'):
            try:
                abs_path = os.path.abspath(config)
                main_dir = abs_path.split("logs/")[0]
            except:
                pass
            self.from_json(config)

        elif config is None:
            pass # do nothing

        else:
            raise ValueError(f"{config} config file not Found, args and model can not be empty for new experiment")


    def _get_experiment_config(self, args: Namespace) -> Dict[str, Any]:
        return {
            'dataset': args.dataset,
            'gpu_devices': args.gpu_devices,
            'progress_bar': args.progress_bar,
        }

    def _get_dataset_config(self, args: Namespace) -> Dict[str, Any]:
        return {
            'cellstate_key': args.cellstate_key,
            'n_grid': args.n_grid,
            'n_dimension': args.n_dimension,
            "kde_kws": {"bw_method":None},
            'timepoint_idx': args.timepoint_idx,
            'deltax_key': args.deltax_key,
            'norm_time': args.norm_time,
            'knn_volume' : args.knn_volume,
        } 

    def _get_model_config(self, model) -> Dict[str, Any]:
        return {
            'model_class': model.__class__.__name__,
            'channels': getattr(model, 'channels', None),
            'activation_fn': getattr(model, 'activation_fn', None),
            'ode_tol': getattr(model, 'ode_tol', None),
            'R_weight': getattr(model, 'R_weight', None),
            'D_penalty': getattr(model, 'D_penalty', None),
            'deltax_weight': getattr(model, 'deltax_weight', None),
            'weight_intensity': getattr(model, 'weight_intensity', None),
            'time_scale_factor': getattr(model, 'time_scale_factor', None),
            'time_sensitive': getattr(model, 'time_sensitive', None),
            'v_channels': getattr(model, 'v_channels', None),
            'g_channels': getattr(model, 'g_channels', None),
            'D_channels': getattr(model, 'D_channels', None),
        }

    def _get_training_config(self, args: Namespace) -> Dict[str, Any]:
        return {
            'batch_size': args.batch_size,
            'schedule_lr': args.schedule_lr,
            'lr': args.lr,
            'max_epochs': 300,
            'optimizer': 'Adam',
        }

    def to_dict(self) -> Dict[str, Any]:
        """Returns all configurations as a dictionary."""
        return {
            'run_date': self.run_date,
            'experiment_config': self.experiment_config,
            'dataset_config': self.dataset_config,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'raw_args': self.raw_args,
        }

    def save(self, path: str):
        """Saves configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def store_attr(self, json_load):
        for k, v in json_load.items():
            self.__setattr__(k, v)
            if v is None:
                self.__setattr__(k, None)
    
    def from_json(self, file_path: str, main_dir: str = None) -> 'ExperimentConfig':
        """Load a saved experiment configuration from JSON file.
        
        Args:
            file_path: Path to the saved JSON configuration file
            
        Returns:
            ExperimentConfig instance with loaded parameters
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        self.store_attr(data)
        self.raw_args['config'] = file_path

        if main_dir is not None:
            old_main = self.experiment_config['checkpoint_dir'].split("logs/")[0]
            self.experiment_config['checkpoint_dir'] = self.experiment_config['checkpoint_dir'].replace(old_main, main_dir)
            self.experiment_config['save_dir'] = self.experiment_config['save_dir'].replace(old_main, main_dir)

    def find_lastest_ckpt(self):
        """
        find the ckpt file with the minimum loss given the config class
        """
        # look for ckpt files
        log_dir = os.path.join(self.experiment_config['checkpoint_dir'], 'checkpoints')
        ckpts = [f for f in os.listdir(log_dir) if f.endswith('.ckpt')]

        # extract loss
        loss = [float(re.match(r"epoch=\d{1,3}-\w{3,10}_loss=([-, \.,\d]{1,30}).ckpt", ckpt).group(1)) for ckpt in ckpts]
        # look for min loss
        ckpt_path = os.path.join(log_dir, ckpts[np.argmin(loss)])
        return ckpt_path
    
    def get_args(self):
        return Namespace(**self.raw_args)
            


