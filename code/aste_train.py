import os
import torch
import logging
import argparse


import pytorch_lightning as pl

# pl.seed_everything(42)

from transformers import AutoConfig
from pytorch_lightning.utilities import rank_zero_info
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule_with_warmup
)

arg_to_scheduler = {
    'linear': get_linear_schedule_with_warmup,
    'cosine': get_cosine_schedule_with_warmup,
    'cosine_w_restarts': get_cosine_with_hard_restarts_schedule_with_warmup,
    'polynomial': get_polynomial_decay_schedule_with_warmup,
    'constant': get_constant_schedule_with_warmup,
}



from model.model import Model
from utils.aste_datamodule import ASTEDataModule
from utils.aste_result import Result


logger = logging.getLogger(__name__)


class ASTE(pl.LightningModule):
    def __init__(self, hparams, data_module):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module
        self.config = AutoConfig.from_pretrained(self.hparams.model_name_or_path)
        self.config.span_pruning = self.hparams.span_pruning
        self.model = Model.from_pretrained(self.hparams.model_name_or_path, config=self.config)
        self.hparams.max_epochs = hparams.epoch
        hparams.max_epochs = hparams.epoch
        self.hparams.precision = hparams.p
        self.hparams.gpus = hparams.gpu
        self.hparams.gradient_clip_val = hparams.grad
        hparams.gradient_clip_val = hparams.grad
        hparams.data_dir = hparams.prefix + hparams.dataset
        self.hparams.accumulate_grad_batches = 1

    @pl.utilities.rank_zero_only
    def save_model(self):
        dir_name = os.path.join(self.hparams.output_dir + self.hparams.dataset, str(self.hparams.cuda_ids), 'model')
        print(f'## save model to {dir_name}')
        self.model.save_pretrained(dir_name)

    def load_model(self):
        dir_name = os.path.join(self.hparams.output_dir + self.hparams.dataset, str(self.hparams.cuda_ids), 'model')
        print(f'## load model to {dir_name}')
        self.model = Model.from_pretrained(dir_name)

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        outputs = self(**batch)
        loss = outputs['table_loss_S'] + outputs['table_loss_E'] + outputs['pair_loss'] + outputs['table_loss_iaS'] + outputs['table_loss_iaE']
        return loss

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        outputs = self(**batch)
        loss = outputs['table_loss_S'] + outputs['table_loss_E'] + outputs['pair_loss']  + outputs['table_loss_iaS'] + outputs['table_loss_iaE']
        self.log('valid_loss', loss)

        return {
            'ids': outputs['ids'],
            'table_predict_S': outputs['table_predict_S'],
            'table_predict_E': outputs['table_predict_E'],
            'table_labels_S': outputs['table_labels_S'],
            'table_labels_E': outputs['table_labels_E'],
            'pair_preds': outputs['pairs_preds']
        }

    def validation_epoch_end(self, outputs):
        examples = self.data_module.raw_datasets['dev']

        self.current_val_result = Result.parse_from(outputs, examples)
        self.current_val_result.cal_metric()

        if not hasattr(self, 'best_val_result'):
            self.best_val_result = self.current_val_result

        elif self.best_val_result < self.current_val_result:
            self.best_val_result = self.current_val_result
            self.save_model()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch,batch_idx)

    def test_epoch_end(self, outputs):
        examples = self.data_module.raw_datasets['test']

        self.test_result = Result.parse_from(outputs, examples)
        self.test_result.cal_metric()

    def save_test_result(self):
        dir_name = os.path.join(self.hparams.output_dir + self.hparams.dataset, 'result')
        self.test_result.save(dir_name, self.hparams)

    def setup(self, stage):
        if stage == 'fit':
            self.train_loader = self.train_dataloader()
            ngpus = 1
            effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches * ngpus
            dataset_size = len(self.train_loader.dataset)
            self.total_steps = (dataset_size / effective_batch_size) * self.hparams.max_epochs

    def get_lr_scheduler(self):
        get_scheduler_func = arg_to_scheduler[self.hparams.lr_scheduler]
        if self.hparams.lr_scheduler == 'constant':
            scheduler = get_scheduler_func(self.opt, num_warmup_steps=self.hparams.warmup_steps)
        else:
            scheduler = get_scheduler_func(self.opt, num_warmup_steps=self.hparams.warmup_steps,
                                           num_training_steps=self.total_steps)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return scheduler

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']

        def has_keywords(n, keywords):
            return any(nd in n for nd in keywords)

        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not has_keywords(n, no_decay)],
                'lr': self.hparams.learning_rate,
                'weight_decay': 0
            },
            {
                'params': [p for n, p in self.model.named_parameters() if has_keywords(n, no_decay)],
                'lr': self.hparams.learning_rate,
                'weight_decay': self.hparams.weight_decay
            }
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]
    def add_model_specific_args(parser):
        parser.add_argument("--learning_rate", default=4e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=100, type=int)
        parser.add_argument("--num_workers", default=1, type=int)
        parser.add_argument("--cuda_ids", default=0, type=int)
        parser.add_argument("--model_name_or_path", default='bert-base-uncased', type=str)
        parser.add_argument('--prefix', type=str, default="../data/aste_data_bert/",)
        parser.add_argument('--dataset', type=str, default="fashion", help='dataset')
        parser.add_argument("--p", default=16, type=int)
        parser.add_argument("--output_dir", default="..\output\ASTE\V2/", type=str)
        parser.add_argument("--train_batch_size", default=1, type=int)
        parser.add_argument("--eval_batch_size", default=1, type=int)
        parser.add_argument("--seed", default=42, type=int)
        parser.add_argument("--lr_scheduler", default='linear', type=str)
        parser.add_argument("--weight_decay", default=0.1, type=float)
        parser.add_argument("--max_seq_length", default=-1, type=int)
        parser.add_argument("--epoch", default=15, type=int)
        parser.add_argument("--gpu", default=1, type=int)
        parser.add_argument("--grad", default=1, type=int)
        parser.add_argument("--do_train", action='store_true')
        parser.add_argument("--span_pruning", type=float, default=0.2)
        return parser


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        metrics = {k:(v.detach() if type(v) is torch.Tensor else v) for k,v in metrics.items()}
        rank_zero_info(metrics)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        print('-------------------------------------------------------------------------------------------------------------------\n[current]\t', end='')
        pl_module.current_val_result.report()

        print('[best]\t\t', end='')
        pl_module.best_val_result.report()
        print('-------------------------------------------------------------------------------------------------------------------\n')

    def on_test_end(self, trainer, pl_module):
        pl_module.test_result.report()
        pl_module.save_test_result()


def main():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = ASTE.add_model_specific_args(parser)
    parser = ASTEDataModule.add_argparse_args(parser)
    args = parser.parse_args()

    # pl.seed_everything(args.seed)

    if args.learning_rate >= 1:
        args.learning_rate /= 1e5


    data_module = ASTEDataModule(args)
    data_module.load_dataset()
    model = ASTE(args, data_module)

    logging_callback = LoggingCallback()

    kwargs = {
        'callbacks': [logging_callback],
        'logger': True,
        'checkpoint_callback': False,
        'num_sanity_val_steps': 5 if args.do_train else 0,
    }

    trainer = pl.Trainer.from_argparse_args(args, **kwargs, gpus=1)
    trainer.fit(model, datamodule=data_module)
    model.load_model()
    trainer.test(model, datamodule=data_module)

if __name__ == '__main__':
    main()
