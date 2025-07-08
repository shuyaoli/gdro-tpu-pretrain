from torch_xla.debug import metrics
from torch_xla.distributed.parallel_loader import MpDeviceLoader
import torch
from torch.utils.data import DataLoader
from transformers.trainer import Trainer
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl 

class StreamingTrainer(Trainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        # 1) Build a standard DataLoader that uses your HF data_collator
        train_loader = DataLoader(
            self.train_dataset,#type: ignore
            batch_size=self.args.per_device_train_batch_size,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.data_collator,            # ← ensures input_ids, attention_mask, labels
            pin_memory=False,                         # no pinned memory on TPU
            generator=torch.Generator().manual_seed(self.args.seed),
            # persistent_workers=True,
            drop_last=True,
            # prefetch_factor=None,
        )
        print(f"[StreamingTrainer] Use device: {self.args.device} for dataloader")
        para_loader = pl.ParallelLoader(train_loader, [xm.xla_device()])

        # 2) Wrap it so that every batch is moved onto the TPU device
        return para_loader.per_device_loader(xm.xla_device())

    def get_eval_dataloader(self, eval_dataset=None):
        ds = eval_dataset or self.eval_dataset
        if ds is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_loader = DataLoader(
            ds,#type: ignore
            batch_size=self.args.per_device_eval_batch_size,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=self.data_collator,            # ← again, to get labels for eval loss
            pin_memory=False,
            generator=torch.Generator().manual_seed(self.args.seed + 1),
            persistent_workers=True,
            shuffle=False,
            drop_last=False,
            prefetch_factor=2,
        )
        para_loader = pl.ParallelLoader(eval_loader, [xm.xla_device()])
        return para_loader.per_device_loader(xm.xla_device())
    # def training_step(self, model, inputs, num_items_in_batch):
    #     # run one step
    #     out = super().training_step(model, inputs, num_items_in_batch)

    #     # on the very first step, dump some debug info:
    #     if self.state.global_step == 0:
    #         xm.master_print("### TPU CHECK ###")
    #         xm.master_print("  XLA devices:", xm.get_xla_supported_devices())
    #         xm.master_print("  Current device:", xm.xla_device())
    #         xm.master_print("  Batch on device:", inputs['input_ids'].device)
    #         xm.master_print("  Model on device:", next(model.parameters()).device)
    #         xm.master_print("  TPU config:", os.environ.get("XRT_TPU_CONFIG"))
    #         xm.master_print(metrics.metrics_report())
    #     return out        