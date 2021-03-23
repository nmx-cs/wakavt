# -*- coding: utf-8 -*-


import torch
import os
import time
import argparse
import json

from Opts import Opts
from DataHelper import DataHelper
from trainer import Trainer
from generator import Generator
from utils import ensure_deterministic


assert torch.__version__ >= "1.6.0"


def check_gen_args(params):
    allowed_gen_args = {"gen_size", "batch_size", "gen_mode", "beam_width", "length_norm", "n_best"}
    assert len(set(params.keys()) - allowed_gen_args) == 0


def check_test_args(params):
    allowed_test_args = {"batch_size", "test_sample_n", "loss_mean"}
    assert len(set(params.keys()) - allowed_test_args) == 0


def set_cuda_device(device_id):
    try:
        torch.cuda.set_device(device_id)
        assert torch.cuda.is_available()
        Opts.device = torch.device("cuda:{}".format(device_id))
    except:
        print("cuda device {} not available, will only use cpu".format(device_id))
        Opts.device = torch.device("cpu")


def get_train_data(opts, reform_data, get_word2sylnum):
    if opts is None:
        opts = Opts()
    word2id_path = os.path.join(data_dir, "word2id.json")
    data_helper = DataHelper(opts, data_path, None, emb_path, vocab_path, word2id_path,
                             get_word2sylnum, reform_data, data_dir, emb_data_dir)
    return data_helper.prepare_train_data(load_pretrained_embs=args.load_pretrained_embs)


def train(train_mode, reform_data, get_word2sylnum, pretrained_model_path):
    opts = Opts()
    if train_mode != "restart":
        opts.load_opts(hyp_params_path)
        print("Opts after loading:\n{}".format(opts))
    data_dict = get_train_data(opts, reform_data, get_word2sylnum)
    if train_mode == "restart":
        opts.dump_opts(hyp_params_path)
        print("opts for training:\n{}".format(opts))
    trainer_ = Trainer(opts, train_mode, data_dict, states_path, metrics_path, pretrained_model_path)
    trainer_.train()


def test(get_word2sylnum, **kwargs):
    opts = Opts()
    opts.load_opts(hyp_params_path)
    opts.set_opts(kwargs)
    print("Opts after loading:\n{}".format(opts))

    word2id_path = os.path.join(data_dir, "word2id.json")
    data_helper = DataHelper(opts, None, None, None, None, word2id_path,
                             get_word2sylnum, False, data_dir, None)
    data_dict = data_helper.prepare_test_data()

    trainer_ = Trainer(opts, "test", data_dict, states_path, metrics_path)
    trainer_.test()
    trainer_.save_metrics()


def generate(remove_EOS, remove_UNK, reform_gendata, only_top_beam, get_word2sylnum, latent_file_name, offset, **kwargs):
    opts = Opts()
    opts.load_opts(hyp_params_path)
    opts.set_opts(kwargs)
    print("Opts after loading:\n{}".format(opts))

    word2id_path = os.path.join(data_dir, "word2id.json")
    data_helper = DataHelper(opts, None, keywords_path, None, None, word2id_path,
                             get_word2sylnum, False, data_dir, None)
    data_dict = data_helper.prepare_gen_data(reform_gendata, kwargs.get("gen_size", None), latent_file_name, offset)
    generator = Generator(opts, data_dict, states_path)

    stime = time.time()
    gen_ids = generator.generate()
    res = generator.process_gen_ids(gen_ids, remove_EOS, remove_UNK, only_top_beam)
    generator.save_results(res, gen_res_path, remove_EOS, remove_UNK, only_top_beam)
    print("Time consumed :", time.time() - stime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="train", choices=["train", "test", "gen", "reform_data"])
    parser.add_argument("-dd", "--data_dir", type=str, default="./data")
    parser.add_argument("-edd", "--emb_data_dir", type=str, default="./data/pretrained_embs")
    parser.add_argument("-dp", "--data_path", type=str, default=None)
    parser.add_argument("-pp", "--pretrained_word_emb_path", type=str, default=None, help="pretrained txt embeddings")
    parser.add_argument("-kp", "--keywords_path", type=str, default=None, help="keywords for generations")
    parser.add_argument("-vp", "--vocab_path", type=str, default=None, help="vocab file")
    parser.add_argument("-rd", "--results_dir", type=str, default="./results")
    parser.add_argument("-grd", "--gen_res_dirname", type=str, default="gen_results")
    parser.add_argument("-f", "--state_file", type=str, default="step_0000.pt")
    parser.add_argument("-lf", "--latent_file_name", type=str, default=None)
    parser.add_argument("-lo", "--latent_offset", type=int, default=0)
    parser.add_argument("-pm", "--pretrained_model_path", type=str, default=None,
                        help="pretrained weights used to initialize model")
    parser.add_argument("-tm", "--train_mode", type=str, default="restart", choices=["restart", "continue", "test"])
    parser.add_argument("-lpe", "--load_pretrained_embs", action="store_true")
    parser.add_argument("-rf", "--reform_data", action="store_true")
    parser.add_argument("-tp", "--test_params", type=json.loads, default='{"batch_size": 32, "test_sample_n": 3, "loss_mean": "steps"}',
                        help="params to be set into the opts object before constructing the trainer and testing")
    parser.add_argument("-rg", "--new_gen_data", action="store_true")
    parser.add_argument("-re", "--remove_EOS", action="store_true")
    parser.add_argument("-ru", "--remove_UNK", action="store_true")
    parser.add_argument("-ot", "--only_top_beam", action="store_true", help="only keep top beams in the generation file")
    parser.add_argument("-gp", "--gen_params", type=json.loads,
                        default='{"gen_size": 1000, "batch_size": 32, "gen_mode": "beam", "beam_width": 20, "length_norm": 0, "n_best": 20}',
                        help="params to be set into the opts object before constructing the generator and generating")
    parser.add_argument("-dt", "--ensure_deterministic", action="store_true")
    parser.add_argument("-gw", "--get_word2sylnum", action="store_true")
    parser.add_argument("-rs", "--random_seed", default=0, type=int)
    parser.add_argument("-cd", "--cuda_device", type=int, default=0, help="-1 to disable cuda and only use cpu")
    args = parser.parse_args()

    check_gen_args(args.gen_params)
    check_test_args(args.test_params)

    if args.cuda_device >= 0:
        set_cuda_device(args.cuda_device)

    if args.ensure_deterministic:
        ensure_deterministic(args.random_seed)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    data_dir = args.data_dir
    emb_data_dir = args.emb_data_dir
    data_path = args.data_path if args.data_path else os.path.join(data_dir, "texts.json")
    emb_path = args.pretrained_word_emb_path if args.pretrained_word_emb_path else os.path.join(data_dir, "embeddings.txt")
    keywords_path = args.keywords_path if args.keywords_path else os.path.join(data_dir, "keywords.txt")
    vocab_path = args.vocab_path if args.vocab_path else os.path.join(data_dir, "vocab.txt")
    results_dir = args.results_dir
    states_path = os.path.join(results_dir, args.state_file)
    metrics_path = os.path.join(results_dir, "metrics.json")
    gen_res_path = os.path.join(results_dir, args.gen_res_dirname)
    hyp_params_path = os.path.join(results_dir, "hyp_params.json")

    if args.mode == "train":
        train(args.train_mode, args.reform_data, args.get_word2sylnum, args.pretrained_model_path)
    elif args.mode == "gen":
        generate(args.remove_EOS, args.remove_UNK, args.new_gen_data, args.only_top_beam,
                 args.get_word2sylnum, args.latent_file_name, args.latent_offset, **args.gen_params)
    elif args.mode == "test":
        test(args.get_word2sylnum, **args.test_params)
    elif args.mode == "reform_data":
        get_train_data(None, True, args.get_word2sylnum)
