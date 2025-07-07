from transformers.trainer_callback import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.training_args import TrainingArguments
from dataset import StatefulShardedDataset  # Assuming this is your custom dataset
import torch
import torch_xla.core.xla_model as xm
from typing import Callable

class DynamicSamplingOnEvaluationCallback(TrainerCallback):
    """
    A Hugging Face TrainerCallback that dynamically adjusts dataset sampling
    weights based on the evaluation loss.
    """
    def __init__(self, dataset: StatefulShardedDataset, weight_update_fn: Callable):
        self.dataset = dataset
        self.weight_update_fn = weight_update_fn

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics: dict[str, float], **kwargs):
        """Event called after an evaluation phase."""
        if xm.is_master_ordinal():
            eval_loss = metrics.get("eval_loss")
            if eval_loss is None:
                print("Warning: 'eval_loss' not found in metrics. Skipping weight update.")
                return

            new_weights = self.weight_update_fn(eval_loss)
            print(f"\n--- Evaluation at Step {state.global_step} ---")
            print(f"Evaluation Loss: {eval_loss:.4f}")
            print(f"Updating sampling weights to: {[f'{w:.4f}' for w in new_weights]}")
            self.dataset.update_weights(new_weights)


class DynamicSamplingCallback(TrainerCallback): # Not used
    """
    A Hugging Face TrainerCallback that dynamically adjusts dataset sampling
    weights based on the training loss every N steps.
    """
    def __init__(
        self,
        dataset: StatefulShardedDataset,
        update_every_n_steps: int = 100,
        # Your logic to map loss to weights goes here
        weight_update_fn: Callable = lambda loss: [1.0 / max(loss, 1e-6)] 
    ):
        self.dataset = dataset
        self.update_every_n_steps = update_every_n_steps
        self.weight_update_fn = weight_update_fn
        self.running_losses = []

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Event called at the end of a training step."""
        # Get the latest loss from the trainer's state
        # The log_history contains dicts like {'loss': 3.4, 'learning_rate':...}
        if state.log_history:
            latest_loss = state.log_history[-1].get("loss")
            if latest_loss is not None:
                self.running_losses.append(torch.tensor(latest_loss, device=xm.xla_device()))

        # Check if it's time to perform an update
        if state.global_step > 0 and state.global_step % self.update_every_n_steps == 0:
            if not self.running_losses:
                return # Nothing to do if we haven't collected any losses

            # This must be called on all processes to avoid deadlocks.
            # It gathers the loss tensors from all TPU cores and averages them.
            avg_loss_tensor = xm.mesh_reduce(
                'loss_reduce_tag',
                torch.mean(torch.stack(self.running_losses)),
                lambda x: torch.mean(torch.stack(x))
            )
            # Clear the buffer for the next window
            self.running_losses = []

            # The rest of the logic should only run on the master process
            if xm.is_master_ordinal():
                avg_loss_val = avg_loss_tensor #.items()
                # 1. Calculate new weights using the provided function
                new_weights = self.weight_update_fn(avg_loss_val)
                
                # 2. Log the information
                print(f"\n--- Step {state.global_step} ---")
                print(f"Avg loss over last {self.update_every_n_steps} steps: {avg_loss_val:.4f}")
                print(f"Updating sampling weights to: {[f'{w:.4f}' for w in new_weights]}")
                
                # 3. Update the dataset's weights directly
                self.dataset.update_weights(new_weights)