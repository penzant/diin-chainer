import argparse
import os
import six
import math
import gzip
import pickle

import chainer
import chainer.functions as F
import numpy as np
from chainer import cuda, reporter, serializers, training
from chainer.dataset import iterator as iterator_module
from chainer.iterators import SerialIterator, MultiprocessIterator
from chainer.training import extensions
from chainer import reporter as reporter_module
from chainer import link, function

from net import DIIN
from data_processing import *


def load_embedding(config, load_params, word_indices):
    embedding_dir = os.path.join(config.data_path, "embeddings")
    if not os.path.exists(embedding_dir):
        os.makedirs(embedding_dir)
    embedding_path = os.path.join(embedding_dir, "mnli_emb_snli_embedding.pkl.gz")

    if os.path.exists(embedding_path):
        print("embedding path exist")
        f = gzip.open(embedding_path, 'rb')
        loaded_embeddings = pickle.load(f)
        f.close()
    else:
        loaded_embeddings = loadEmbedding_rand(load_params["embedding_data_path"], word_indices)
        f = gzip.open(embedding_path, 'wb')
        pickle.dump(loaded_embeddings, f)
        f.close()

    print("embedding loaded")
    return loaded_embeddings


def load_dataset(config):

    load_params = {}
    load_params['training_mnli'] = '{}/multinli_0.9/multinli_0.9_train.jsonl'.format(config.data_path)
    load_params['dev_matched'] = '{}/multinli_0.9/multinli_0.9_dev_matched.jsonl'.format(config.data_path)
    load_params['dev_mismatched'] = '{}/multinli_0.9/multinli_0.9_dev_mismatched.jsonl'.format(config.data_path)
    load_params['test_matched'] = '{}/multinli_0.9/multinli_0.9_test_matched_unlabeled.jsonl'.format(config.data_path)
    load_params['test_mismatched'] = '{}/multinli_0.9/multinli_0.9_test_mismatched_unlabeled.jsonl'.format(config.data_path)
    load_params['training_snli'] = '{}/snli_1.0/snli_1.0_train.jsonl'.format(config.data_path)
    load_params['dev_snli'] = '{}/snli_1.0/snli_1.0_dev.jsonl'.format(config.data_path)
    load_params['test_snli'] = '{}/snli_1.0/snli_1.0_test.jsonl'.format(config.data_path)
    load_params['embedding_data_path'] = '{}/glove.840B.300d.txt'.format(config.data_path)

    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)

    if config.debug_mode:
        test_matched = load_nli_data(load_params['dev_matched'], shuffle = False)[:150]
        training_snli, dev_snli, test_snli, training_mnli, dev_matched, dev_mismatched, test_mismatched = test_matched, test_matched,test_matched,test_matched,test_matched,test_matched,test_matched
        indices_to_words, word_indices, char_indices, indices_to_chars = sentences_to_padded_index_sequences([test_matched], config)
        shared_content_path = config.data_path + '/shared_debug.pkl.gz'
        if os.path.exists(shared_content_path):
            f = gzip.open(shared_content_path, 'rb')
            shared_content = pickle.load(f)
            f.close()
        else:
            shared_content = load_mnli_shared_content(config)
            f = gzip.open(shared_content_path, 'wb')
            pickle.dump(shared_content, f)
            f.close()
    
    else:
        training_snli = load_nli_data(load_params["training_snli"], snli=True)
        dev_snli = load_nli_data(load_params["dev_snli"], snli=True)
        test_snli = load_nli_data(load_params["test_snli"], snli=True)
        training_mnli = load_nli_data(load_params["training_mnli"])
        dev_matched = load_nli_data(load_params["dev_matched"])
        dev_mismatched = load_nli_data(load_params["dev_mismatched"])

        test_matched = load_nli_data(load_params["test_matched"], shuffle = False)
        test_mismatched = load_nli_data(load_params["test_mismatched"], shuffle = False)

        shared_content_path = config.data_path + '/shared.pkl.gz'
        if os.path.exists(shared_content_path):
            f = gzip.open(shared_content_path, 'rb')
            shared_content = pickle.load(f)
            f.close()
        else:
            shared_content = load_mnli_shared_content(config)
            f = gzip.open(shared_content_path, 'wb')
            pickle.dump(shared_content, f)
            f.close()
        indices_to_words, word_indices, char_indices, indices_to_chars = sentences_to_padded_index_sequences([training_mnli, training_snli, dev_matched, dev_mismatched, test_matched, test_mismatched, dev_snli, test_snli], config)

    loaded_embeddings = load_embedding(config, load_params, word_indices)
    config.word_vocab_size = loaded_embeddings.shape[0]
    config.char_vocab_size = len(char_indices.keys())

    if config.snli:
        train_data = training_snli
        test_data = test_snli
    else:
        train_data = training_mnli + random.sample(training_snli, int(config.snli_ratio * len(training_snli)))
        test_data = dev_matched + dev_mismatched
        
    return train_data, test_data, shared_content, loaded_embeddings, config


def nli_converter(batch, device):
    if device >= 0:
        xp = cuda.cupy
        xp.cuda.Device(device).use()
    else:
        xp = np
    return {'x_prem': xp.asarray([b[0] for b in batch], dtype=xp.int32),
            'x_hypo': xp.asarray([b[1] for b in batch], dtype=xp.int32),
            'x_prem_pos': xp.asarray([b[2] for b in batch], dtype=xp.float32),
            'x_hypo_pos': xp.asarray([b[3] for b in batch], dtype=xp.float32),
            'x_prem_char': xp.asarray([b[4] for b in batch], dtype=xp.int32),
            'x_hypo_char': xp.asarray([b[5] for b in batch], dtype=xp.int32),
            'x_prem_match': xp.asarray([b[6] for b in batch], dtype=xp.float32),
            'x_hypo_match': xp.asarray([b[7] for b in batch], dtype=xp.float32),
            'y': xp.asarray([b[8] for b in batch], dtype=xp.int32)}


class NLIDataset(chainer.dataset.DatasetMixin):

    def __init__(self, data, shared_data, config):
        self.data = data
        self.shared_data = shared_data
        self.seq_length = config.seq_length
        self.char_column_size = config.char_in_word_size

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        data_item = self.data[i]
        genre = data_item['genre']
        labels = data_item['label']
        pairID = data_item['pairID']

        prem_vec = data_item['sentence1_binary_parse_index_sequence']
        hypo_vec = data_item['sentence2_binary_parse_index_sequence']

        prem_char = data_item['sentence1_binary_parse_char_index']
        hypo_char = data_item['sentence2_binary_parse_char_index']

        prem_pos = generate_pos_feature_vector(data_item['sentence1_parse'], self.seq_length)        
        hypo_pos = generate_pos_feature_vector(data_item['sentence2_parse'], self.seq_length)

        prem_exact_match = construct_one_hot_feature_vector(
            self.shared_data[pairID]['sentence1_token_exact_match_with_s2'], 1, self.seq_length)
        hypo_exact_match = construct_one_hot_feature_vector(
            self.shared_data[pairID]['sentence2_token_exact_match_with_s1'], 1, self.seq_length)

        prem_exact_match = np.expand_dims(prem_exact_match, 2)
        hypo_exact_match = np.expand_dims(hypo_exact_match, 2)

        return (prem_vec, hypo_vec, prem_pos, hypo_pos, prem_char, hypo_char,
                prem_exact_match, hypo_exact_match, labels)


class DIINEvaluator(training.extensions.Evaluator):

    def __init__(self, iterator, target, converter=nli_converter, device=None,
                 eval_hook=None, eval_func=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if isinstance(target, link.Link):
            target = {'main': target}
        self._targets = target

        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook
        self.eval_func = eval_func

    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                with function.no_backprop_mode():
                    loss, acc = eval_func(**in_arrays)
                    reporter_module.report({'val/loss': loss.data, 'val/acc': acc.data})

            summary.add(observation)

        return summary.compute_mean()
            

class DIINUpdater(training.StandardUpdater):

    def __init__(self, iterator, model, optimizer, config, converter=nli_converter, device=None):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self.model = model

        if not isinstance(optimizer, dict):
            optimizer = {'main': optimizer}
        self._optimizers = optimizer

        if device is not None and device >= 0:
            for optimizer in six.itervalues(self._optimizers):
                optimizer.target.to_gpu(device)

        self.converter = converter
        
        self.device = device
        self.iteration = 0

        self.decay_step = config.dropout_decay_step
        self.decay_rate = config.dropout_decay_rate
        self.weight_l2loss_step = config.weight_l2loss_step
        self.weight_l2loss_ratio = config.weight_l2loss_ratio
        self.diff_penalty_loss_ratio = config.diff_penalty_loss_ratio

    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        optimizer = self._optimizers['main']
        
        loss, acc = self.model(**in_arrays)

        # loss append
        weights_added = self.add_n([self.l2_loss(param) for param in optimizer.target.params() if param.name == 'W'])
        half_l2_step = self.weight_l2loss_step / 2
        global_step = optimizer.t
        full_l2_ratio = self.weight_l2loss_ratio
        l2loss_ratio = self.sigmoid(((global_step - half_l2_step) * 8) / half_l2_step) * full_l2_ratio
        l2loss = weights_added * l2loss_ratio

        # penaly update of loss
        diff_links = [(self.model.self_att_layers_p.logits_linear.W, self.model.self_att_layers_h.logits_linear.W),
                      (self.model.self_att_layers_p.fg_rhs_1.W, self.model.self_att_layers_h.fg_rhs_1.W),
                      (self.model.self_att_layers_p.fg_lhs_1.W, self.model.self_att_layers_h.fg_lhs_1.W),
                      (self.model.self_att_layers_p.fg_rhs_2.W, self.model.self_att_layers_h.fg_rhs_2.W),
                      (self.model.self_att_layers_p.fg_lhs_2.W, self.model.self_att_layers_h.fg_lhs_2.W),
                      (self.model.self_att_layers_p.fg_rhs_3.W, self.model.self_att_layers_h.fg_rhs_3.W),
                      (self.model.self_att_layers_p.fg_lhs_3.W, self.model.self_att_layers_h.fg_lhs_3.W)]
        diffs = [p[0] - p[1] for p in diff_links]
        diff_loss = self.add_n([self.l2_loss(diff) for diff in diffs])
        diff_loss *= self.diff_penalty_loss_ratio
        
        loss += l2loss
        loss += diff_loss

        reporter.report({'loss_all': loss.data, 'loss_l2': l2loss, 'loss_diff': diff_loss, 'acc': acc.data})

        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()

        self.model.drop_rate = 1 - self.exponential_decay(self.model.init_rate, self.decay_rate,
                                                          optimizer.t, self.decay_step)

        reporter.report({'drop_rate': self.model.drop_rate})


    def exponential_decay(self, keep_rate, decay_rate, global_step, decay_steps):
        decayed_rate = keep_rate * decay_rate ** float(global_step / decay_steps)
        return decayed_rate

    def add_n(self, param_list):
        return F.sum(F.stack(param_list))

    def l2_loss(self, param):
        return F.sum(param ** 2) / 2

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))


def main():
    parser = argparse.ArgumentParser()

    pa = parser.add_argument

    pa('--gpu', type=str, default='0')
    pa('--epoch', type=int, default=100)
    pa('--debug_mode', action='store_true')

    pa('--data_path', type=str, default='../Densely-Interactive-Inference-Network/data')
    pa('--ckpt_path', type=str, default='logs')
    pa('--log_path', type=str, default='logs')

    pa('--batch_size', type=int, default=70)
    pa('--display_step', type=int, default=50)
    pa('--eval_step', type=int, default=1000)

    pa('--seq_length', type=int, default=48, help='Max sequence length')
    pa('--num_process_prepro', type=int, default=24, help='num process prepro')
    pa('--embedding_replacing_rare_word_with_UNK', action='store_true',
       help='embedding_replacing_rare_word_with_UNK')
    pa('--UNK_threshold', type=int, default=5, help='UNK threshold')

    pa('--snli', action='store_true', help='train and test on snli')
    pa('--snli_ratio', type=float, default=0.15, help='train on mnli and x% snli')

    # pa('--word_vocab_size', type=int, default=122650)
    pa('--word_emb_dim', type=int, default=300)
    # pa('--char_vocab_size', type=int, default=75)
    pa('--char_emb_dim', type=int, default=8)
    pa('--char_in_word_size', type=int, default=16)
    pa('--char_conv_n_kernel', type=int, default=100)
    pa('--char_conv_height', type=int, default=5)
    pa('--pos_size', type=int, default=47) # data_processing.py
    pa('--match_size', type=int, default=1) # data_processing.py

    pa('--learning_rate', type=float, default=0.5)
    pa('--adadelta_rho', type=float, default=0.95)
    pa('--adadelta_eps', type=float, default=1e-8)
    # pa('--weight_decay', type=float, default=0.0)
    pa('--gradient_clip', type=float, default=1.0)
    pa('--keep_rate', type=float, default=1.0)

    pa('--dropout_decay_step', type=int, default=10000)
    pa('--dropout_decay_rate', type=float, default=0.977)
    pa('--weight_l2loss_step', type=int, default=100000)
    pa('--weight_l2loss_ratio', type=float, default=9e-5)
    pa('--diff_penalty_loss_ratio', type=float, default=1e-3)

    pa('--highway_n_layer', type=int, default=2)
    pa('--dense_n_block', type=int, default=3)
    pa('--dense_n_layer', type=int, default=8)
    pa('--dense_feat_dim', type=int, default=448)
    pa('--dense_kernel_size', type=int, default=3)
    pa('--dense_growth_rate', type=int, default=20)
    pa('--dense_first_scale_down_rate', type=float, default=0.3)
    pa('--dense_trans_scale_down_rate', type=float, default=0.5)

    pa('--pred_size', type=int, default=3)
    
    config = parser.parse_args()
    print(json.dumps(config.__dict__, indent=4))

    # dataset
    train_data, test_data, shared_data, embeddings, config = load_dataset(config)
    train_data = NLIDataset(train_data, shared_data, config)
    test_data = NLIDataset(test_data, shared_data, config)

    # config update
    config.enc_dim = config.word_emb_dim + config.char_conv_n_kernel + config.pos_size + config.match_size
    config.gpu = [int(g) for g in config.gpu.split(',')]
    config.embeddings = embeddings

    # model
    model = DIIN(config)
    if config.gpu[0] >= 0:
        model.to_gpu(config.gpu[0])

    # Optimizer
    optimizer = chainer.optimizers.Adam() #Delta(config.adadelta_rho, eps=config.adadelta_eps) # TODO: lr 0.5
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(config.gradient_clip))
    # optimizer.add_hook(chainer.optimizer.WeightDecay(config.weight_decay))

    # Iterator
    train_iter = MultiprocessIterator(train_data, config.batch_size, repeat=True, shuffle=True)
    test_iter = MultiprocessIterator(test_data, config.batch_size, repeat=False, shuffle=False)
    
    # Updater, trainer
    updater = DIINUpdater(train_iter, model, optimizer, config, device=config.gpu[0])
    trainer = training.Trainer(updater, (config.epoch, 'epoch'), out=config.log_path)
    
    evaluator = DIINEvaluator(test_iter, model,
                              converter=nli_converter, device=config.gpu[0])
    evaluator.name = 'val'

    iter_per_epoch = len(train_data) // config.batch_size
    print('Iter/epoch =', iter_per_epoch)

    log_trigger = (min(10, iter_per_epoch // 2), 'iteration')
    eval_trigger = (500, 'iteration')
    record_trigger = training.triggers.MaxValueTrigger(
        'val/acc', eval_trigger)

    trainer.extend(extensions.snapshot_object(
        model, 'model_epoch_{.updater.epoch}.npz'),
        trigger=record_trigger)
    trainer.extend(evaluator, trigger=eval_trigger)
    trainer.extend(extensions.LogReport(trigger=log_trigger, log_name='iteration.log'))
    trainer.extend(extensions.LogReport(trigger=eval_trigger, log_name='epoch.log'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'loss_all', 'loss_l2', 'loss_diff', 'acc',
         'val/loss', 'val/acc', 'drop_rate', 'elapsed_time']))

    trainer.run()


if __name__ == '__main__':
    main()

