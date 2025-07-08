# Enable XLA debugging

os.environ["XLA_IR_DEBUG"] = "1"

os.environ["XLA_HLO_DEBUG"] = "1"

os.environ["XRT_TPU_CONFIG"] = "tpu_worker;0;10.0.0.2:8470;1;10.0.0.3:8470;2;10.0.0.4:8470;3;10.0.0.5:8470"

os.environ["PJRT_DEVICE"] = 'TPU'

os.environ["XLA_METRICS_ENABLED"] = 'true'


import torch

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


from dataset import StatefulShardedDataset, EvaluationShardedDataset

from callback import DynamicSamplingOnEvaluationCallback, StepTimingCallback

from trainer import StreamingTrainer



TOP_SEED = 42

torch.manual_seed(TOP_SEED)


base = '/home/shuyaoli/llm_data/converted_dataset'

domain_dirs = {

'book': os.path.join(base, 'book'),

'arxiv': os.path.join(base, 'arxiv'),

'stackexchange':os.path.join(base, 'stackexchange'),

'wiki': os.path.join(base, 'wiki'),

'c4-rp': os.path.join(base, 'c4-rp'),

'cc': os.path.join(base, 'cc'),

'github': os.path.join(base, 'github'),

}


EVAL_PATH = '/home/shuyaoli/llm_data/converted_dataset/eval_merge'

eval_dataset = EvaluationShardedDataset(

EVAL_PATH,

)


initial_weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


# Instantiate your master dataset

master_train_dataset = StatefulShardedDataset(

domain_dirs=domain_dirs,

initial_weights=initial_weights,

chunk_size=32

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


training_args = TrainingArguments(

output_dir="./tpu_eval_sampling_model",

max_steps=5000,

per_device_train_batch_size=1,

per_device_eval_batch_size=1,

gradient_accumulation_steps=2, # increase if you need >16 global batch on v4-8

gradient_checkpointing=False,

logging_steps=1,

eval_strategy="steps",

eval_steps=200, # Run evaluation every 1000 steps

bf16=True, # enable BF16

bf16_full_eval=True,

dataloader_num_workers=0,

remove_unused_columns=True,

)


eval_sampling_callback = DynamicSamplingOnEvaluationCallback(

dataset=master_train_dataset,

weight_update_fn=lambda x: 1

)


trainer = StreamingTrainer(

model=model,

args=training_args,

train_dataset=master_train_dataset,

eval_dataset=eval_dataset, # Provide the evaluation dataset

callbacks=[

# eval_sampling_callback, # for debugging

StepTimingCallback(),

], # Use the new callback

data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),

)


print(f"Starting training on device: {xm.xla_device()}, world size: {xr.world_size()}")


trainer.train() 