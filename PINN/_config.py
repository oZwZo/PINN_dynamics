import os
import datetime
import json
from argparse import Namespace
from typing import Dict, Any

class ExperimentConfig:
    """Records and manages experiment configuration settings."""
    
    def __init__(self,  save_path: str=None,  args: Namespace = None, model = None):
        """
        Args:
            args: Parsed command-line arguments
            save_path: Experiment directory path
            model: Initialized model instance
        """
        if (args is not None) and (model is not None):
            self.run_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.experiment_config = self._get_experiment_config(args, save_path)
            self.dataset_config = self._get_dataset_config(args)
            self.model_config = self._get_model_config(model)
            self.training_config = self._get_training_config(args)
            self.raw_args = vars(args)

        elif os.path.exists(save_path) and save_path.endswith('.json'):
            self.from_json(save_path)

        elif save_path is None:
            pass # do nothing

        else:
            raise ValueError(f"{save_path} config file not Found, args and model can not be empty for new experiment")


    def _get_experiment_config(self, args: Namespace, save_path: str) -> Dict[str, Any]:
        return {
            'save_dir': save_path,
            'gpu_devices': args.gpu_devices,
            'progress_bar': args.progress_bar,
        }

    def _get_dataset_config(self, args: Namespace) -> Dict[str, Any]:
        return {
            'dataset': args.dataset,
            'cellstate_key': args.cellstate_key,
            'n_grid': args.n_grid,
            'n_dimension': args.n_dimension,
            'bw': args.bw,
            'timepoint_idx': args.timepoint_idx,
            'deltax_key': args.deltax_key,
            'norm_time': args.norm_time,
        }

    def _get_model_config(self, model) -> Dict[str, Any]:
        return {
            'model_class': model.__class__.__name__,
            'channels': getattr(model, 'channels', None),
            'activation_fn': getattr(model, 'activation_fn', None),
            'ode_tol': getattr(model, 'ode_tol', None),
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
        for k, v in json_load:
            self.__setattr__(k, v)
            if v is None:
                self.__setattr__(k, None)
    
    @classmethod
    def from_json(cls, file_path: str) -> 'ExperimentConfig':
        """Load a saved experiment configuration from JSON file.
        
        Args:
            file_path: Path to the saved JSON configuration file
            
        Returns:
            ExperimentConfig instance with loaded parameters
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        cls.store_attr(data)
            
        # Create dummy Namespace for raw arguments
        args = Namespace(**data['raw_args'])

