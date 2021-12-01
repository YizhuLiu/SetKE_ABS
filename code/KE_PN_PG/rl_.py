""" RL training utilities"""
import math
from time import time
from datetime import timedelta

from toolz.sandbox.core import unzip
from cytoolz import concat

import numpy as np
import torch
from torch.nn import functional as F
from torch import autograd
from torch.nn.utils import clip_grad_norm_

from metric import compute_rouge_l, compute_rouge_n, compute_rouge_l_summ
from training import BasicPipeline
from nltk import sent_tokenize


def get_grad_fn(agent, clip_grad, max_grad=1e2):
    """ monitor gradient for each sub-component"""
    params = [p for p in agent.parameters()]
    def f():
        grad_log = {}
        for n, m in agent.named_children():
            tot_grad = 0
            for p in m.parameters():
                if p.grad is not None:
                    tot_grad += p.grad.norm(2) ** 2
            tot_grad = tot_grad ** (1/2)
            grad_log['grad_norm'+n] = tot_grad.item()
        grad_norm = clip_grad_norm_(
            [p for p in params if p.requires_grad], clip_grad)
        try:
            grad_norm = grad_norm.item()
        except AttributeError:
            grad_norm = grad_norm
        if max_grad is not None and grad_norm >= max_grad:
            print('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            grad_norm = max_grad
        grad_log['grad_norm'] = grad_norm
        return grad_log
    return f


class A2CPipeline(BasicPipeline):
    def __init__(self, name,
                 net, abstractor,
                 train_batcher, val_batcher,
                 optim, grad_fn,
                 reward_fn, gamma,
                 stop_reward_fn, stop_coeff):
        self.name = name
        self._net = net
        self._train_batcher = train_batcher
        self._val_batcher = val_batcher
        self._opt = optim
        self._grad_fn = grad_fn

        self._abstractor = abstractor
        self._gamma = gamma
        self._reward_fn = reward_fn
        self._stop_reward_fn = stop_reward_fn
        self._stop_coeff = stop_coeff

        self._n_epoch = 0  # epoch not very useful?

    def batches(self):
        raise NotImplementedError('A2C does not use batcher')

    def train_step(self):
        # forward pass of model
        self._net.train()
        log_dict = a2c_train_step(
            self._net, self._abstractor,
            self._train_batcher,
            self._opt, self._grad_fn,
            self._gamma, self._reward_fn,
            self._stop_reward_fn, self._stop_coeff
        )
        return log_dict

    def validate(self):
        return a2c_validate(self._net, self._abstractor, self._val_batcher)

    def checkpoint(self, *args, **kwargs):
        # explicitly use inherited function in case I forgot :)
        return super().checkpoint(*args, **kwargs)

    def terminate(self):
        pass  # No extra processs so do nothing

def compute_rouge_with_marginal_increase(reward_fn, targets, abss, step, gamma=0.95):
    reward = 0
    if reward_fn.__name__ != 'compute_rouge_l_summ':
        if step == len(targets): # this time step greedy generates "stop"
            reward += reward_fn(list(concat(targets[:])), list(concat(abss)))
        else:
            sent_reward = [reward_fn(list(concat(targets[:i+1+step])), list(concat(abss))) for i in range(len(targets) - step)]
            for ind in range(len(sent_reward)):
                if ind != 0:
                    reward += math.pow(gamma, ind) * (sent_reward[ind] - sent_reward[ind-1])
                else:
                    reward += sent_reward[ind]
    else:
        if step == len(targets): # this time step greedy generates "stop"
            reward += reward_fn(list(concat(targets[:])), abss)
        else:
            sent_reward = [reward_fn(targets[:i+1+step], abss) for i in range(len(targets) - step)]
            for ind in range(len(sent_reward)):
                if ind != 0:
                    reward += math.pow(gamma, ind) * (sent_reward[ind] - sent_reward[ind-1])
                else:
                    reward += sent_reward[ind]

    return reward


#def a2c_train_step(agent, abstractor, loader, opt, grad_fn, reward_fn=compute_rouge_l):
def a2c_train_step(agent, abstractor, loader, opt, grad_fn,
                   gamma=0.99, reward_fn=compute_rouge_l,
                   stop_reward_fn=compute_rouge_n(n=1), stop_coeff=1.0):
    sample_time = 1
    time_variant = True
    gamma = 0.95
    opt.zero_grad()
    art_batch, abs_batch, extracts = next(loader)
    all_loss = []
    reward = 0
    advantage = 0
    i = 0
    greedy_inputs = []
    sample_inputs = []
    sample_log_probs = []
    for idx, raw_arts in enumerate(art_batch):
        #print(len(raw_arts))
        #print(len(raw_arts[0]))
        greedy, samples, all_log_probs = agent(raw_arts, sample_time=sample_time)
        #print(greedy, samples, all_log_probs)
        #print(len(greedy), len(samples), len(all_log_probs))
        #exit()
        if time_variant:
            bs = []
            abss = abs_batch[idx]
            for _ind, gd in enumerate(greedy):
                #print('1',greedy_sents)
                greedy_sents = []
                ext_sent = []
                for ids in gd:
                    if ids < len(raw_arts):
                       if ids == 0:
                          if ext_sent:
                             greedy_sents.append(ext_sent)
                          ext_sent = []
                       else:
                          ext_sent += raw_arts[ids]
                if gd[-1] != 0 and ext_sent:
                   greedy_sents.append(ext_sent)
                #print('2',greedy_sents)
                #greedy_sents = [raw_arts[ind] for ind in gd]
                
                baseline = 0
                for i, sent in enumerate(greedy_sents):
                    #greedy_sents = [[word for sent in greedy_sents for word in sent]]
                    #greedy_sents = abstractor(greedy_sents)
                    #print('3',greedy_sents)
                    #exit()
                    with torch.no_grad():
                        #print(sent)
                        greedy_sent = abstractor([sent])
                        #print(greedy_sent)
                        #print('section')
                        #exit()
                        greedy_sent = sent_tokenize(' '.join(greedy_sent[0]))
                        greedy_sent = [s.strip().split(' ') for s in greedy_sent]
                    #print('1', list(concat(greedy_sent)))
                    #exit()
                    if reward_fn.__name__ != 'compute_rouge_l_summ':
                        if _ind != len(greedy)-1:
                            if i < len(abss):
                               baseline += compute_rouge_with_marginal_increase(reward_fn, greedy_sent, abss[i], _ind, gamma=gamma)
                        else:
                            if i < len(abss):
                               baseline += reward_fn(list(concat(greedy_sent)), list(concat(abss[i])))
                        #print('1', baseline)
                    else:
                        if _ind != len(greedy)-1:
                            if i < len(abss):
                               baseline += compute_rouge_with_marginal_increase(reward_fn, greedy_sent, abss[i], _ind, gamma=gamma)
                        else:
                            if i < len(abss):
                               baseline += reward_fn(greedy_sent, abss[i])
                        #print('2', baseline)
                #print(baseline)
                bs.append(baseline)
                #exit()
            #print(greedy, len(greedy), len(bs))
            #print(len(greedy), len(bs))
            #print(samples)
            #exit()
            #sample_sents = [raw_arts[ind] for ind in samples[0]]
            sample_sents = []
            ext_sent = []
            for ids in samples[0]:
                    if ids < len(raw_arts):
                       if ids == 0:
                          if ext_sent:
                             sample_sents.append(ext_sent)
                          ext_sent = []
                       else:
                          ext_sent += raw_arts[ids]
            if gd[-1] != 0 and ext_sent:
                   sample_sents.append(ext_sent)

            all_rewards = []
            for j, sent in enumerate(sample_sents):
                with torch.no_grad():
                     #sample_sents = [[word for sent in sample_sents for word in sent]]
                     sample_sent = abstractor([sent])
                     sample_sent = sent_tokenize(' '.join(sample_sent[0]))
                     sample_sent = [s.strip().split(' ') for s in sample_sent]

                #print('2', sample_sent)
                if reward_fn.__name__ != 'compute_rouge_l_summ':
                   #print(sample_sent, abss[j])
                   #exit()
                   #rewards = [reward_fn(list(concat(sample_sent[:i+1])), list(concat(abss[j]))) for i in range(len(sample_sent))]
                   rewards = []
                   for i in range(len(sample_sent)):
                       if j < len(abss):
                          #print('3', sample_sent[:+1])
                          #exit()
                          rewards.append(reward_fn(list(concat(sample_sent[:i+1])), list(concat(abss[j]))))
                       else:
                          rewards.append(0)
                   #print(rewards,len(rewards))
                   #exit()
                   for index in range(len(rewards)):
                       rwd = 0
                       for _index in range(len(rewards)-index):
                           if _index != 0:
                               rwd += (rewards[_index+index] - rewards[_index+index-1]) * math.pow(gamma, _index)
                           else:
                               rwd += rewards[_index+index]
                       all_rewards.append(rwd)
                   if j < len(abss):
                      all_rewards.append(
                             compute_rouge_n(list(concat(sample_sent[:j])), list(concat(abss[:j])))
                      )
                   else:
                      all_rewards.append(
                             compute_rouge_n(list(concat(sample_sent[:j])), list(concat(abss)))
                      )
                else:
                   #rewards = [reward_fn(sample_sent[:i + 1], abss[j]) for i in
                   #           range(len(sample_sent))]
                   rewards = []
                   for i in range(len(sample_sent)):
                       if j < len(abss):
                          rewards.append(reward_fn(list(concat(sample_sent[:i+1])), list(concat(abss[j]))))
                       else:
                          rewards.append(0)
                   for index in range(len(rewards)):
                       rwd = 0
                       for _index in range(len(rewards) - index):
                           if _index != 0:
                               rwd += (rewards[_index + index] - rewards[_index + index - 1]) * math.pow(gamma, _index)
                           else:
                               rwd += rewards[_index + index]
                       all_rewards.append(rwd)
                   all_rewards.append(
                       compute_rouge_n(list(concat(sample_sent)), list(concat(abss[j])))
                   )
            # print('greedy:', greedy)
            # print('sample:', samples[0])
            # print('baseline:', bs)
            # print('rewars:', all_rewards)
            reward += bs[-1]
            advantage += (all_rewards[-1] - bs[-1])
            i += 1
            advs = [torch.tensor([_bs - rwd], dtype=torch.float).to(all_log_probs[0][0].device) for _bs, rwd in zip(bs, all_rewards)]
            for log_prob, adv in zip(all_log_probs[0], advs):
                all_loss.append(log_prob * adv)
        else:
            greedy_sents = [raw_arts[ind] for ind in greedy]
            greedy_sents = [word for sent in greedy_sents for word in sent]
            greedy_inputs.append(greedy_sents)
            sample_sents = [raw_arts[ind] for ind in samples[0]]
            sample_sents = [word for sent in sample_sents for word in sent]
            sample_inputs.append(sample_sents)
            sample_log_probs.append(all_log_probs[0])
    if not time_variant:
        with torch.no_grad():
            greedy_outs = abstractor(greedy_inputs)
            sample_outs = abstractor(sample_inputs)
        for greedy_sents, sample_sents, log_probs, abss in zip(greedy_outs, sample_outs, sample_log_probs, abs_batch):
            greedy_sents = sent_tokenize(' '.join(greedy_sents))
            greedy_sents = [sent.strip().split(' ') for sent in greedy_sents]
            if reward_fn.__name__ != 'compute_rouge_l_summ':
                bs = reward_fn(list(concat(greedy_sents)), list(concat(abss)))
            else:
                bs = reward_fn(greedy_sents, abss)
            sample_sents = sent_tokenize(' '.join(sample_sents))
            sample_sents = [sent.strip().split(' ') for sent in sample_sents]
            if reward_fn.__name__ != 'compute_rouge_l_summ':
                rwd = reward_fn(list(concat(sample_sents)), list(concat(abss)))
            else:
                rwd = reward_fn(sample_sents, abss)
            reward += bs
            advantage += (rwd - bs)
            i += 1
            adv = torch.tensor([bs - rwd], dtype=torch.float).to(log_probs[0].device)
            for log_prob in log_probs:
                all_loss.append(log_prob * adv)
    reward = reward / i
    advantage = advantage / i
    

    # backprop and update
    loss = torch.cat(all_loss, dim=0).mean()
    loss.backward()
    grad_log = grad_fn()
    opt.step()
    log_dict = {}
    log_dict.update(grad_log)
    log_dict['reward'] = reward
    log_dict['advantage'] = advantage
    log_dict['mse'] = 0
    assert not math.isnan(log_dict['grad_norm'])
    return log_dict


def a2c_validate(agent, abstractor, loader):
    agent.eval()
    start = time()
    print('start running validation...', end='')
    avg_reward = 0
    i = 0
    with torch.no_grad():
        for art_batch, abs_batch, extract in loader:
            greedy_inputs = []
            for idx, raw_arts in enumerate(art_batch):
                greedy, sample, log_probs = agent(raw_arts, sample_time=1, validate=True)
                sample = sample[0]
                log_probs = log_probs[0]
                greedy_sents = [raw_arts[ind] for ind in greedy]
                greedy_sents = [word for sent in greedy_sents for word in sent]
                #print(greedy_sents)
                #greedy_sents = list(concat(greedy_sents))
                greedy_sents = []
                ext_sent = []
                for ids in greedy:
                    if ids < len(raw_arts):
                       if ids == 0:
                          if ext_sent:
                             greedy_sents.append(ext_sent)
                          ext_sent = []
                       else:
                          ext_sent += raw_arts[ids]
                if greedy[-1] != 0 and ext_sent:
                   greedy_sents.append(ext_sent)
                #print(greedy_sents)
                #exit()
                greedy_inputs.append(greedy_sents)
            greedy_abstracts = []
            for abs_src in greedy_inputs:
                with torch.no_grad():
                     greedy_outputs = abstractor(abs_src)
                #greedy_abstract = []
                #for greedy_sents in greedy_outputs:
                #    greedy_sents = sent_tokenize(' '.join(greedy_sents))
                #    greedy_sents = [sent.strip().split(' ') for sent in greedy_sents]
                #    greedy_abstract += greedy_sents
                greedy_abstract = list(concat(greedy_outputs))
                greedy_abstracts.append(greedy_abstract)
            for idx, greedy_sents in enumerate(greedy_abstracts):
                abss = abs_batch[idx]
                bs = compute_rouge_n(greedy_sents, list(concat(abss)))
                avg_reward += bs
                i += 1
                #print(i)
                #print(avg_reward)
                #exit()
    avg_reward /= (i/100)
    print('finished in {}! avg reward: {:.2f}'.format(
        timedelta(seconds=int(time()-start)), avg_reward))
    return {'reward': avg_reward}
