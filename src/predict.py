import torch
import os
import csv
from tqdm.auto import tqdm
from multiprocessing import Pool, Process
import pandas as pd
from src.config import init_argument
from src.features import prepare_data
from src.evaluate import compute_rouges
from src.train import T5PegasusTokenizer


def generate(test_data, model, tokenizer, args):
    gens, summaries = [], []
    with open(args.result_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        model.eval()
        for feature in tqdm(test_data):
            raw_data = feature['raw_data']
            content = {k: v.to(args.device) for k, v in feature.items() if k not in ['raw_data', 'title']} 
            gen = model.generate(max_length=args.max_len_generate,
                                eos_token_id=tokenizer.sep_token_id,
                                decoder_start_token_id=tokenizer.cls_token_id,
                                **content)
            gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
            gen = [item.replace(' ', '') for item in gen]
            writer.writerows(zip(gen, raw_data))
            gens.extend(gen)
            if 'title' in feature:
                summaries.extend(feature['title'])
    if summaries:
        scores = compute_rouges(gens, summaries)
        print(scores)
    print('Done!')


def generate_multiprocess(feature):
    """多进程
    """
    model.eval()
    raw_data = feature['raw_data']
    content = {k: v for k, v in feature.items() if k != 'raw_data'}
    gen = model.generate(max_length=args.max_len_generate,
                             eos_token_id=tokenizer.sep_token_id,
                             decoder_start_token_id=tokenizer.cls_token_id,
                             **content)
    gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
    results = ["{}\t{}".format(x.replace(' ', ''), y) for x, y in zip(gen, raw_data)]
    return results


if __name__ == '__main__':
    
    # step 1. init argument
    args = init_argument()
    args.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # step 2. prepare test data
    tokenizer = T5PegasusTokenizer.from_pretrained(args.pretrain_model)
    test_data = prepare_data(args, tokenizer, 'test')
    
    # step 3. load finetuned model
    model_path = os.path.join(args.model_dir, 'model.pth')
    model = torch.load(model_path, map_location=args.device)

    # step 4. predict
    res = []
    if args.use_multiprocess and args.device == 'cpu':
        print('Parent process %s.' % os.getpid())
        p = Pool(2)
        res = p.map_async(generate_multiprocess, test_data, chunksize=2).get()
        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        res = pd.DataFrame([item for batch in res for item in batch])
        res.to_csv(args.result_file, index=False, header=False, encoding='utf-8')
        print('Done!')
    else:
        generate(test_data, model, tokenizer, args)