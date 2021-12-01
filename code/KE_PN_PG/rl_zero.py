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

from metric import compute_rouge_l, compute_rouge_n
from training import BasicPipeline
from MAP_MRR import metrics


def a2c_validate(agent, abstractor, loader):
    agent.eval()
    start = time()
    print('start running validation...', end='')
    avg_reward = 0
    i = 0
    with torch.no_grad():
        for art_batch, abs_batch, _ in loader:
            #print(art_batch, abs_batch, _)
            ext_sents = []
            ext_inds = []
            new_indices = []
            for raw_arts in art_batch:
                indices = agent(raw_arts)
                extL = len(ext_sents)
                #ext_sents += [raw_arts[idx.item()]
                #              for idx in indices if idx.item() < len(raw_arts)]
                inds_ = []
                ext_sent = []
                for idx in indices:
                    if idx.item() < len(raw_arts):
                       if idx.item() == 0:
                          if ext_sent:
                             ext_sents.append(ext_sent)
                          ext_sent = []
                          inds_.append(1)
                       else:
                          ext_sent += raw_arts[idx.item()]
                if indices[-1].item() != 0 and ext_sent:
                   ext_sents.append(ext_sent)
                   inds_.append(1)
                new_indices.append(inds_)
                indxL = len(new_indices)-1
                ext_inds += [(extL, indxL)]
            all_summs = abstractor(ext_sents)
            for (j, n), abs_sents in zip(ext_inds, abs_batch):
                summs = all_summs[j:j+n]
                # python ROUGE-1 (not official evaluation)
                avg_reward += compute_rouge_n(list(concat(summs)),
                                              list(concat(abs_sents)), n=1)
                i += 1
    avg_reward /= (i/100)
    print('finished in {}! avg reward: {:.2f}'.format(
        timedelta(seconds=int(time()-start)), avg_reward))
    return {'reward': avg_reward}


def a2c_train_step(agent, abstractor, loader, opt, grad_fn,
                   gamma=0.99, reward_fn=compute_rouge_l,
                   stop_reward_fn=compute_rouge_n(n=1), stop_coeff=1.0):
    opt.zero_grad()
    indices = []
    indicesL = []
    new_indices = []
    probs = []
    new_probs = []
    baselines = []
    ext_sents = []
    art_batch, abs_batch, ext_batch = next(loader)
    raw_adv = []
    r = 0 
    a = 0 
    f = 0
    zeros = []
    for raw_arts in art_batch:
        (inds, ms), bs = agent(raw_arts)
        baselines.append(bs) 
        indices.append(inds)
        indicesL.append(len(inds))
        probs.append(ms)
        #ext_sents += [raw_arts[idx.item()]
        #              for idx in inds if idx.item() < len(raw_arts)] 
        #print(ext_sents)
        #exit(0)
        inds_ = []
        ms_ = []
        ext_sent = []
        zero = []

        count = 0 
        q = 0 
        f = 0
        #print(inds)
        for idx in inds:
            #print('bx',bx)
            if idx.item() < len(raw_arts):
               if idx.item() == 0:
                  if ext_sent:
                     count += 1
                     ext_sents.append(ext_sent)
                     zero.append(f)
                     inds_.append(1)
                     raw_adv.append((r,a))
                     a += 1
                     q = 1
                     #print('1',a)					 
                  else:
                     if q==1:				  
                        raw_adv.append((r,a-1))
                     else:
                        raw_adv.append((r,a))
                     #print('a')					 
                  ext_sent = []
                  count = 0
               else:
                  ext_sent += raw_arts[idx.item()] 
                  #print('b')
                  count += 1
                  raw_adv.append((r,a))
                  q = 0
            else:
                  #print('c')
                  if q==1:				  
                        raw_adv.append((r,a-1))
                  else:
                        raw_adv.append((r,a))
            f += 1
        if inds[-1].item() != 0:
           if ext_sent:
              ext_sents.append(ext_sent)
              zero.append(f)
              inds_.append(1)
              a += 1
              #print('2',a)					 
           #ext_sent = []
        if not inds_:
           ext_sents.append(raw_arts[0])
           inds_.append(1)
           zero.append(f)
           a += 1
           #print('3',a)					 
        r+=1
        zeros.append(zero)
        new_indices.append(inds_)
        #print(len(inds),len(bs),len(inds_))
        #print(len(zero), len(inds_))
        #print(ms[0].probs)
        #exit()
    with torch.no_grad():
        summaries = abstractor(ext_sents)
    i = 0
    rewards = []
    avg_reward = 0
    set_reward, summ_reward = [],[]
    for inds, abss in zip(new_indices, abs_batch):
        set_reward.append([reward_fn(summaries[i+j], abss[j])
              for j in range(min(len(inds)-1, len(abss)))])
        summ_reward.append(stop_coeff*stop_reward_fn(
                  list(concat(summaries[i:i+len(inds)-1])),
                  list(concat(abss))))
        '''
        rs = ([reward_fn(summaries[i+j], abss[j])
              for j in range(min(len(inds)-1, len(abss)))]
              + [0 for _ in range(max(0, len(inds)-1-len(abss)))]
              + [stop_coeff*stop_reward_fn(
                  list(concat(summaries[i:i+len(inds)-1])),
                  list(concat(abss)))])
        if len(rs) != len(inds):				  
           print(len(rs),len(inds))
           print(indices)
           print(new_indices)
        assert len(rs) == len(inds)
        avg_reward += rs[-1]/stop_coeff
        i += len(inds)-1
        # compute discounted rewards
        R = 0
        disc_rs = []
        for r in rs[::-1]:
            R = r + gamma * R
            disc_rs.insert(0, R)
        rewards += disc_rs
        '''

    ext_rewards = []
    avg_ext_reward = 0  
    for inds, exts, arts, setR, summR, zero in zip(indices, ext_batch, art_batch, set_reward, summ_reward, zeros):
        k = 0
        rs1 = [] 
        ext_zero = []
        for e in range(len(exts)):
            if exts[e] == 0:
               ext_zero.append(e)
        e0 = 0               
        maxlen = 0
        for j in range(min(len(inds)-1, len(exts))):
            #print(set(inds) & set(exts))
            maxlen+=1
            if j in zero and k < len(setR) and k < len(ext_zero)-1:
               rs1.append(setR[k])
               k+=1
               e0 = ext_zero[k] + 1
            else:
               try:
                  if inds[j].item() < len(arts):
                     rs1.append(compute_rouge_n(arts[inds[j].item()], arts[exts[e0]],n=1))
                  else:
                     rs1.append(0.0)
                  e0 += 1
               except:
                  maxlen -= 1
                  break
        rs2 = [0 for _ in range(len(inds)-1-maxlen)]  
        stop = metrics(gt=exts, pred=inds, metrics_map=['MAP'])[0]
        rs3 = [stop_coeff*summR]
        rs = rs1 + rs2 + rs3
        assert len(rs) == len(inds)
        avg_ext_reward += rs[-1]/stop_coeff
        # compute discounted rewards
        R = 0
        disc_rs = []
        #print(rs)
        #print(rs[::-1])
        #exit()
        for r in rs[::-1]:
            R = r + gamma * R
            disc_rs.insert(0, R)
        ext_rewards += disc_rs
    #print(len(ext_rewards))
    #exit()
    indices = list(concat(indices))
    #print(len(indices),len(raw_adv))
    new_indices = list(concat(new_indices))
    #print(len(new_indices))
    probs = list(concat(probs))
    baselines = list(concat(baselines))
    #exit()
    # standardize rewards
    ext_reward = torch.Tensor(ext_rewards).to(baselines[0].device)
    ext_reward = (ext_reward - ext_reward.mean()) / (
        ext_reward.std() + float(np.finfo(np.float32).eps))
    baseline = torch.cat(baselines).squeeze()
    avg_advantage = 0
    losses = []
    for r, b, p, action in zip(ext_reward, baseline, probs, indices):
        advantage = r - b
        avg_advantage += advantage
        losses.append((-p.log_prob(action)
                       * (advantage/len(indices)))) # divide by T*B
    #exit()
    critic_loss = F.mse_loss(baseline, ext_reward)
    # backprop and update
    #print("[DEBUG]")
    #print(critic_loss)
    critic_loss = critic_loss.view([1])
    #print(critic_loss)
    autograd.backward(
        tensors=[critic_loss] + losses,
        grad_tensors=[torch.ones(1).to(critic_loss.device)]*(1+len(losses))
    )
    #exit()
    grad_log = grad_fn()
    opt.step()
    log_dict = {}
    log_dict.update(grad_log)
    log_dict['reward'] = avg_ext_reward/len(art_batch)
    log_dict['advantage'] = avg_advantage.item()/len(indices)
    log_dict['mse'] = critic_loss.item()
    assert not math.isnan(log_dict['grad_norm'])
    return log_dict


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
        #print("[DEBUG]")
        #print(grad_norm)
        #grad_norm = grad_norm.item()
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
