import torch
import rouge
from tqdm import tqdm


def compute_rouge(source, target):
    """计算rouge-1、rouge-2、rouge-l
    """
    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.Rouge().get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }


def compute_rouges(sources, targets):
    scores = {
        'rouge-1': 0.0,
        'rouge-2': 0.0,
        'rouge-l': 0.0,
    }
    for source, target in zip(sources, targets):
        score = compute_rouge(source, target)
        for k, v in scores.items():
            scores[k] = v + score[k]

    return {k: v / len(targets) for k, v in scores.items()}


def evaluate(model, tokenizer, dev_data, args):
    model.eval()
    gens = []
    summaries = []
    for feature in tqdm(dev_data):
        title = feature['title']
        content = {k: v.to(args.device) for k, v in feature.items() if k != 'title'}
        if args.data_parallel and torch.cuda.is_available():
            gen = model.module.generate(max_length=args.max_len_generate,
                         eos_token_id=tokenizer.sep_token_id,
                         decoder_start_token_id=tokenizer.cls_token_id,
                         **content)
        else:
            gen = model.generate(max_length=args.max_len_generate,
                         eos_token_id=tokenizer.sep_token_id,
                         decoder_start_token_id=tokenizer.cls_token_id,
                         **content)
        gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
        gen = [item.replace(' ', '') for item in gen]
        # print(title)
        # print(gen)
        gens.extend(gen)
        summaries.extend(title)
    scores = compute_rouges(gens, summaries)
    return scores