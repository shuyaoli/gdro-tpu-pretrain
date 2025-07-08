import os
from accelerate import Accelerator
from transformers.optimization import get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch

# Enable XLA debugging
# os.environ["XLA_IR_DEBUG"] = "1"
# os.environ["XLA_HLO_DEBUG"] = "1"
# os.environ["XLA_METRICS_ENABLED"] = 'true'

import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.parallel_loader as pl
import torch_xla.debug.profiler as xp
import torch_xla.debug.metrics as met
import os

from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.training_args import TrainingArguments
# from accelerate import Accelerator, DataLoaderConfiguration, DistributedType

from new_dataset_streaming import StatefulShardedDataset, EvaluationShardedDataset
from callback import DynamicSamplingOnEvaluationCallback, StepTimingCallback
from trainer import StreamingTrainer
from transformers.trainer import Trainer


def main():
    accelerator = Accelerator()    
    TOP_SEED = 42
    torch.manual_seed(TOP_SEED)

    base = '/home/shuyaoli/llm_data/converted_dataset'
    domain_dirs = {
        'book':        os.path.join(base, 'book'),
        'arxiv':       os.path.join(base, 'arxiv'),
        'stackexchange':os.path.join(base, 'stackexchange'),
        'wiki':        os.path.join(base, 'wiki'),
        'c4-rp':       os.path.join(base, 'c4-rp'),
        'cc':          os.path.join(base, 'cc'),
        'github':      os.path.join(base, 'github'),
    }

    EVAL_PATH = '/home/shuyaoli/llm_data/converted_dataset/eval_merge'
    eval_dataset = EvaluationShardedDataset(
        EVAL_PATH,
    )

    initial_weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    # Instantiate your master dataset
    train_dataset = StatefulShardedDataset(
        domain_dirs=domain_dirs,
        initial_weights=initial_weights,
        chunk_size=1
    )

    # Using the Sheared-LLaMA model for continued pretraining.
    model_name = "princeton-nlp/Sheared-LLaMA-1.3B-Pruned"

    # You will need to be logged into your Hugging Face account and have
    # access to meta-llama models for this to work.
    # `huggingface-cli login`
    tokenizer_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # Set pad token to EOS token for Causal LM
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    train_dataloader = DataLoader(train_dataset, batch_size=1)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1)    
    optimizer = AdamW(model.parameters(), lr=5e-5)    

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    num_train_epochs = 1
    num_training_steps = 5000

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    model.train()    
    for batch in train_dataloader:
        if completed_steps >= num_training_steps:
            break
        batch["labels"] = batch["input_ids"].clone()
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss) # Instead of loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        completed_steps += 1

