import torch
import torch.nn.functional as F

from transformers import Trainer

from typing import Any, Dict, List, Optional, Tuple, Union

import random
import torch
from torch import nn
from torch.utils.data import DistributedSampler, RandomSampler
import numpy as np

from transformers import PreTrainedModel, Trainer, logging, Seq2SeqTrainer
from transformers.integrations import is_fairscale_available
from transformers.models.fsmt.configuration_fsmt import FSMTConfig
from transformers.optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)

from transformers.trainer_pt_utils import get_tpu_sampler
from transformers.training_args import ParallelMode
from transformers.utils import is_torch_tpu_available
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


if is_fairscale_available():
    from fairscale.optim import OSS


logger = logging.get_logger(__name__)

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "constant_w_warmup": get_constant_schedule_with_warmup,
}


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, config=None, data_args=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if config is None:
            assert isinstance(self.model, PreTrainedModel), (
                "If no `config` is passed the model to be trained has to be of type `PreTrainedModel`, but is"
                f" {self.model.__class__}"
            )
            self.config = self.model.config
        else:
            self.config = config

        self.data_args = data_args
        self.vocab_size = self.config.tgt_vocab_size if isinstance(self.config, FSMTConfig) else self.config.vocab_size  

        if self.args.label_smoothing_factor != 0 or (self.data_args is not None and self.data_args.ignore_pad_token_for_loss):
            assert self.config.pad_token_id is not None, (
                "Make sure that `config.pad_token_id` is correcly defined when ignoring `pad_token` for loss"
                " calculation or doing label smoothing."
            )

        if self.config.pad_token_id is None and self.config.eos_token_id is not None:
            logger.warning(
                f"The `config.pad_token_id` is `None`. Using `config.eos_token_id` = {self.config.eos_token_id} for"
                " padding.."
            )

        if self.args.label_smoothing_factor == 0:
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
        else:
            # dynamically import label_smoothed_nll_loss
            from utils import label_smoothed_nll_loss
            self.loss_fn = label_smoothed_nll_loss

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            
            optimizer_kwargs["lr"] = self.args.learning_rate
            
            if self.sharded_ddp:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if self.lr_scheduler is None:
            self.lr_scheduler = self._get_lr_scheduler(num_training_steps)
        else:  # ignoring --lr_scheduler
            logger.warning("scheduler is passed to `Seq2SeqTrainer`, `--lr_scheduler` arg is ignored.")

    def _get_lr_scheduler(self, num_training_steps):

        self.args.lr_scheduler = "constant_w_warmup" # temporary
        schedule_func = arg_to_scheduler[self.args.lr_scheduler]

        if self.args.lr_scheduler == "constant":
            scheduler = schedule_func(self.optimizer)
        elif self.args.lr_scheduler == "constant_w_warmup":
            scheduler = schedule_func(self.optimizer, num_warmup_steps=self.args.warmup_steps)
        else:
            scheduler = schedule_func(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )
        return scheduler

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.train_dataset)
        else:
            if self.args.sortish_sampler:
                self.train_dataset.make_sortish_sampler(
                    self.args.per_device_train_batch_size,
                    distributed=(self.args.parallel_mode == ParallelMode.DISTRIBUTED),
                )

            return (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

    def _compute_loss(self, model, inputs, labels):

        if self.args.label_smoothing_factor == 0:
            if self.data_args is not None and self.data_args.ignore_pad_token_for_loss:
                # force training to ignore pad token
                logits = model(**inputs, use_cache=False)[0]
                loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            else:
                # compute usual loss via models
                loss, logits = model(**inputs, labels=labels, use_cache=False)[:2]
        else:
            # compute label smoothed loss
            logits = model(**inputs, use_cache=False)[0]
            lprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss, _ = self.loss_fn(lprobs, labels, self.args.label_smoothing, ignore_index=self.config.pad_token_id)
        return loss, logits 

    def compute_loss(self, model, inputs):
        labels = inputs.pop("labels")
        loss, logits = self._compute_loss(model, inputs, labels)
        return loss, logits
        

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
            A tuple with the loss, logits and labels (each being optional).
        """
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": self.data_args.val_max_target_length
            if self.data_args is not None
            else self.config.max_length,
            "num_beams": self.data_args.eval_beams if self.data_args is not None else self.config.num_beams,
        }

        if self.args.predict_with_generate and not self.args.prediction_loss_only:
            generated_tokens = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )
            # in case the batch is shorter than max length, the output should be padded
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        labels = inputs.pop("labels")
        with torch.no_grad():
            # compute loss on predict data
            loss, logits = self._compute_loss(model, inputs, labels)

        loss = loss.mean().detach()
        if self.args.prediction_loss_only:
            return (loss, None, None)

        logits = generated_tokens if self.args.predict_with_generate else logits

        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return (loss, logits, labels)

    def _pad_tensors_to_max_len(self, tensor, max_length):
        # If PAD token is not defined at least EOS token has to be defined
        pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else self.config.eos_token_id

        if pad_token_id is None:
            raise ValueError(
                "Make sure that either `config.pad_token_id` or `config.eos_token_id` is defined if tensor has to be"
                f" padded to `max_length`={max_length}"
            )

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor


class GenerationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # noinspection PyUnresolvedReferences

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        
        logits = outputs[0]
        input_ids = inputs['input_ids']

        TGT_IDX = self.tokenizer.get_vocab()["Answer"]
        END_IDX = 50256
    
        loss_fct = nn.CrossEntropyLoss(reduce=False, ignore_index=END_IDX)

        # For validation, let's just get the loss after "Answer: " part.
        if 'mode' in inputs and inputs['mode'] == "validation":
            # Find Answer:
            target_idxs = [torch.where(input_ids[idx] == TGT_IDX)[0][0] for idx in range(len(input_ids))]
            
            # Mask the question part.
            for i in range(input_ids.shape[0]):
                input_ids[i][:target_idxs[i] + 3] = 50256 #

        shift_labels = input_ids[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()

        # 토큰당 손실값 계산
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)) # Fixed, for GPT2.

        # 샘플당 손실값을 resize하고 평균화
        loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)
        
        # Calculate and scale weighting.
        weights = 1.0
        # weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(axis=[0, 2]) # Keyword Based

        # Calculate weighted average
        weighted_loss = (loss_per_sample * weights).mean()

        if return_outputs:
            return weighted_loss, logits
        else:
           return weighted_loss

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        ## Ignore Deepspeed.

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size
        num_samples = self.num_examples(dataloader)

        print(f"***** Running {description} *****")
        print(f"  Num examples = {num_samples}")
        print(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        # if is_torch_tpu_available():
        #     dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None
        
        # losses/preds/labels on CPU (final containers)
        all_losses = 0
        all_preds = 0
        
        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0
        all_inputs = None
        
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            all_inputs = torch.cat([all_inputs, inputs['input_ids']]) if all_inputs is not None else inputs['input_ids']
            # Prediction step
            inputs['mode'] = 'validation'
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None
            
            # label도 고쳐야되는지 확인하기.
            all_losses += loss * logits.shape[0]
        
        final_loss = all_losses.item() / num_samples
        
        metrics = {}
        metrics['eval_loss'] = final_loss

        # Just dummy data.
        all_labels = []

        # Let's try to generate a random sample!
        sel_input_ids = all_inputs[random.randint(0, num_samples - 1)]

        TGT_IDX = self.tokenizer.get_vocab()["Answer"]
        SEP_IDX = torch.where(sel_input_ids == TGT_IDX)[0][0] + 3

        sel_input = self.tokenizer.decode(sel_input_ids[:SEP_IDX])
        sel_gt = self.tokenizer.decode(sel_input_ids[SEP_IDX:], skip_special_tokens=True)

        model_input = self.tokenizer.encode(sel_input, return_tensors='pt').to(self.args.device)
        with torch.no_grad():     
            result = model.module.generate(
                model_input, 
                max_length=100, 
                num_beams=3, 
                early_stopping=True
            )
            sel_output = self.tokenizer.decode(result[0], skip_special_tokens=True)
        
        print(f"***** Prediction Sample Result *****")
        print(f"*** <Sample Input> ***")
        print(sel_input)
        print(f"*** <Sample Output> ***")
        print(sel_output[len(sel_input):])
        print(f"*** <Sample GT> ***")
        print(sel_gt)
        print("***** END OF Evaluation *****")

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

# End of GenerationTrainer



def get_option_token_pred_mask_all(inputs):
    """
    Get mask for relevant tokens for PREDICTIONS (shifted by 1 compared to inputs)
    for all options
    """
    batch_size, num_options, seq_length = inputs["input_ids"].shape
    mask = torch.zeros(batch_size, num_options, (seq_length-1)).bool()
    for i in range(batch_size):
        for j in range(num_options):
            #  preds will be back-shifted by one
            mask[i, j, inputs["option_token_start_idx"][i, j]-1:] = 1
            mask[i, j, inputs["option_token_end_idx"][i, j]-1:] = 0
    mask = mask.to(inputs["input_ids"].device)
    return mask


def get_option_token_pred_mask_only_correct(inputs):
    """
    Get mask for relevant tokens for PREDICTIONS (shifted by 1 compared to inputs)
    only for correct option
    """
    batch_size, num_options, seq_length = inputs["input_ids"].shape
    mask = torch.zeros(batch_size, (seq_length-1)).bool()
    for i in range(batch_size):
        correct_idx = inputs["labels"][i]
        #  preds will be back-shifted by one
        mask[i, inputs["option_token_start_idx"][i, correct_idx]-1:] = 1
        mask[i, inputs["option_token_end_idx"][i, correct_idx]-1:] = 0
    mask = mask.to(inputs["input_ids"].device)
    return mask
