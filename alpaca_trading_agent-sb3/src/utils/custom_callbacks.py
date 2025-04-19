import numpy as np
import logging
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

class TensorboardCallback(BaseCallback):
    """
    Custom callback to log training metrics to TensorBoard.
    
    Specifically targets train/policy_gradient_loss, train/value_loss, train/entropy_loss
    which might not be logged automatically via SB3's logger.
    """
    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.tb_formatter = None

    def _on_training_start(self) -> None:
        """
        Find the TensorBoardOutputFormat in the logger's output formats.
        """
        # Find the TensorBoardOutputFormat in the logger's output formats
        for fmt in self.logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                self.tb_formatter = fmt
                #logging.info("[Callback] Found TensorBoardOutputFormat in logger's output formats")
                break
        
        if self.tb_formatter is None:
            logging.warn("[Callback] TensorBoardOutputFormat not found in logger's output formats")

    def _on_rollout_end(self) -> None:
        """
        This method is called at the end of each rollout after the policy update finishes.
        We directly log the metrics to the TensorBoard writer.
        """
        #logging.info("[Callback] ENTERING _on_rollout_end")
        
        # Skip if TensorBoard formatter wasn't found
        if self.tb_formatter is None:
            logging.warn("[Callback] TensorBoardOutputFormat not available, skipping logging")
            return
        
        #logging.info(f"[Callback] Logging metrics at timestep {self.num_timesteps}")
        
        # Access the logger's internal dictionary which maps names to current values
        values = self.logger.name_to_value
        
        # Log all available keys in the logger buffer for debugging
        #logging.info(f"[Callback] Logger keys available: {list(values.keys())}")
        
        # Check if the keys exist and log them directly to TensorBoard
        policy_loss_key = 'train/policy_gradient_loss'
        value_loss_key = 'train/value_loss'
        entropy_loss_key = 'train/entropy_loss'
        
        policy_loss_found = policy_loss_key in values
        value_loss_found = value_loss_key in values
        entropy_loss_found = entropy_loss_key in values
        
        # Log the metrics directly to TensorBoard
        if policy_loss_found:
            self.logger.record("train/policy_loss", values[policy_loss_key])
        
        if value_loss_found:
            self.logger.record("train/value_loss", values[value_loss_key])
        
        if entropy_loss_found:
            self.logger.record("train/entropy_loss", values[entropy_loss_key])
        
        # Log learning rate
        try:
            lr = self.model.policy.optimizer.param_groups[0]['lr']
            self.logger.record("train/learning_rate", lr)
        except (AttributeError, IndexError):
            logging.warn("[Callback] Could not retrieve learning rate.")
        
        # Dump the metrics to disk
        self.logger.dump(self.num_timesteps)
        
        #logging.info(f"[Callback] Found metrics: policy_loss={policy_loss_found}, value_loss={value_loss_found}, entropy_loss={entropy_loss_found}")

    def _on_step(self) -> bool:
        """
        This method is called after each step in the environment.
        We don't need to do anything here for PPO training loss logging.
        """
        return True  # Continue training
