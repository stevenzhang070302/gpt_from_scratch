- dataloader
- GPT Model - Transformer
- Optimize training speed
  - TF32
  - autocast mixed precision with bfloat16
  - torch.compile - kernel fusion from pytorch
  - flash attention on transformer block matric multiplication
  - vocab size - power of 2

- Gradient clipping - preventing weight shocks
- learning rate scheduler + cosine decay
- batch size scheduler - weight decay (regularization) - fusedAdamW
- gradient accumulation - effective batch size - simulation of large batch sizes
- distributed data parallel - parallel training across multiple GPUs
   - grad_accum_steps
   - train_loader
   - model = DDP(model, device_ids=[ddp_local_rank])  # In the forward pass - identitical, backward pass - each gpu has gradients - > allreduce = average the gradients and deposit them on all the GPUs on every rank
   - optimizer = model.module.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type='cuda')
   - model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)  # Only sync gradients on the last micro step - we dont need to sync gradients on every micro step
   - dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)  # Allreduce - average the gradients and deposit them on all the GPUs on every rank
   - tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size

- Pre-training on the FineWeb dataset


