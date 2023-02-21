import argparse


def init_argument():
    parser = argparse.ArgumentParser(description='t5-pegasus-chinese')
    parser.add_argument('--train_data', default='data/train.tsv')
    parser.add_argument('--dev_data', default='data/dev.tsv')
    parser.add_argument('--test_data', default='data/predict.tsv')
    parser.add_argument('--pretrain_model', default='pretrained_model/')
    parser.add_argument('--model_dir', default='models/')
    parser.add_argument('--log_dir', default='log/')
    parser.add_argument('--result_file', default='models/predict_result.tsv')
    parser.add_argument('--num_epoch', default=1, help='number of epoch')
    parser.add_argument('--batch_size', default=16, help='batch size')
    parser.add_argument('--lr', default=2e-4, help='learning rate')
    parser.add_argument('--data_parallel', default=False)
    parser.add_argument('--max_len', default=512, help='max length of inputs')
    parser.add_argument('--max_len_generate', default=40, help='max length of outputs')
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument('--use_multiprocess', default=False, action='store_true')

    args = parser.parse_args()
    return args