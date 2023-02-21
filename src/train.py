import os
import sys
import jieba
import torch
from tqdm.auto import tqdm
# from bert4torch.model import *
from src.config import init_argument
from src.features import prepare_data
from src.evaluate import evaluate
from src.utils import EarlyStopping, set_logger, set_seeds
from transformers import MT5ForConditionalGeneration, BertTokenizer


class T5PegasusTokenizer(BertTokenizer):
    """结合中文特点完善的Tokenizer
    基于词颗粒度的分词，如词表中未出现，再调用BERT原生Tokenizer
    """
    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


def train_model(model, optimizer, train_data, dev_data, tokenizer, args):
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)

    early_stopping = EarlyStopping(
        patience=2, verbose=True, trace_func=logger.info,
        save_dir=args.model_dir)
    # best = 0
    for epoch in range(args.num_epoch):
        model.train()
        for i, cur in enumerate(tqdm(train_data, desc='Epoch {}:'.format(epoch))):
            cur = {k: v.to(args.device) for k, v in cur.items()}
            prob = model(**cur)[0]
            mask = cur['decoder_attention_mask'][:, 1:].reshape(-1).bool()
            prob = prob[:, :-1]
            prob = prob.reshape((-1, prob.size(-1)))[mask]
            labels = cur['decoder_input_ids'][:, 1:].reshape(-1)[mask]
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(prob, labels)
            if i % 100 == 0:
                print("Iter {}:  Training Loss: {}".format(i, loss.item()))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # 验证
        scores = evaluate(model, tokenizer, dev_data, args)
        logger.info("Validation Loss: {}".format(scores))
        rouge_l = scores['rouge-l']
        early_stopping(rouge_l, model)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            # tokenizer.save_pretrained(early_stopping_save_dir)
            sys.exit(0)


def main(args):
    # prepare training data and validation data
    tokenizer = T5PegasusTokenizer.from_pretrained(args.pretrain_model)
    train_data = prepare_data(args, tokenizer, term='train')
    dev_data = prepare_data(args, tokenizer, term='dev')
    # load pretrain model
    args.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model = MT5ForConditionalGeneration.from_pretrained(args.pretrain_model).to(args.device)
    if args.data_parallel and torch.cuda.is_available():
        device_ids = range(torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    # finetune
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_model(model, optimizer, train_data, dev_data, tokenizer, args)


if __name__ == '__main__':

    args = init_argument()
    set_seeds(args.seed)
    logger = set_logger(log_path=os.path.join(args.log_dir, 'train.log'))
    main(args)
