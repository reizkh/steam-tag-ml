from abc import ABC, abstractmethod
import mlflow
from typing import Dict, Any, Optional
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import torch

class MLFlowLogger:
    def __init__(self, experiment_name: str, run_name: Optional[str] = None):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self._setup_experiment()
    
    def _setup_experiment(self):
        mlflow.set_experiment(self.experiment_name)
    
    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int]):
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_model(self, model, artifact_path: str, **kwargs):
        mlflow.pytorch.log_model(model, artifact_path, **kwargs) #type: ignore
    
    def set_tags(self, tags: Dict[str, Any]):
        mlflow.set_tags(tags)

class BaseModelTrainer(ABC):
    model: torch.nn.Module

    def __init__(self, config: DictConfig, logger: Optional[MLFlowLogger] = None):
        self.config = config
        self.logger = logger or self._create_default_logger()
    
    def _create_default_logger(self) -> MLFlowLogger:
        experiment_name = self.config.get('experiment_name', 'default_experiment')
        run_name = self.config.get('run_name', self.__class__.__name__)
        return MLFlowLogger(experiment_name, run_name)
    
    def _log_base_params(self):
        base_params = {
            'model_type': self.__class__.__name__,
            'device': self.config.get('device', 'cuda'),
            'batch_size': self.config.get('batch_size', 32),
            'learning_rate': self.config.get('learning_rate', 5e-5),
            'max_epochs': self.config.get('max_epochs', 10)
        }
        self.logger.log_params(base_params)
    
    def train(self, train_dataloader, val_dataloader=None):
        with mlflow.start_run(run_name=self.logger.run_name):
            self._log_base_params()
            self.model = self.create_model()
            
            # Логирование архитектуры модели
            self._log_model_summary()
            
            max_epochs = self.config.get("max_epochs", 10)
            pbar = tqdm(range(max_epochs))
            for epoch in pbar:
                pbar.set_description(f"Epoch {epoch+1} / {max_epochs}")
                train_metrics = self._train_epoch(train_dataloader, epoch)
                val_metrics = {}
                
                if val_dataloader:
                    val_metrics = self._validate_epoch(val_dataloader, epoch)
                
                # Объединение и логирование метрик
                all_metrics = {
                    **{f'train_{k}': v for k, v in train_metrics.items()},
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                }
                self.logger.log_metrics(all_metrics, step=epoch)
                
            # Логирование финальной модели
            self._log_final_artifacts()
    
    def _log_model_summary(self):
        model_summary = str(self.model)
        with open('model_summary.txt', 'w') as f:
            f.write(model_summary)
        self.logger.log_artifact('model_summary.txt')

    def _log_final_artifacts(self):
        self.logger.log_model(self.model, 'final_model')
        
        config_dict = OmegaConf.to_container(self.config, resolve=True)
        self.logger.log_params(config_dict) # type: ignore

    @abstractmethod
    def _train_epoch(self, dataloader, epoch) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def _validate_epoch(self, dataloader, epoch) -> Dict[str, float]:
        pass

    @abstractmethod
    def create_model(self) -> torch.nn.Module:
        pass
    
    @abstractmethod
    def preprocess_data(self, data) -> Dict[str, torch.Tensor]:
        pass