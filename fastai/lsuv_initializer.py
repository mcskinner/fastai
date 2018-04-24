"""
From https://github.com/ducha-aiki/LSUV-pytorch

Copyright (C) 2017, Dmytro Mishkin
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the
   distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import torch
import torch.nn.init
import torch.nn as nn


class LSUVInitializer(object):
    def __init__(self):
        self.gg = {
            'hook_position': 0,
            'total_fc_conv_layers': 0,
            'done_counter': -1,
            'hook': None,
            'act_dict': {},
            'counter_to_apply_correction': 0,
            'correction_needed': False,
            'current_coef': 1.0,
        }

    # Orthonorm init code is taked from Lasagne
    # https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
    def svd_orthonormal(self, w):
        shape = w.shape
        if len(shape) < 2:
            raise RuntimeError("Only shapes of length 2 or more are supported.")
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)#w;
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
        return q.astype(np.float32)

    def store_activations(self, input, output):
        self.gg['act_dict'] = output.data.cpu().numpy();
        return

    def add_current_hook(self, m):
        if self.gg['hook'] is not None:
            return
        if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
            if self.gg['hook_position'] > self.gg['done_counter']:
                self.gg['hook'] = m.register_forward_hook(self.store_activations)
            else:
                self.gg['hook_position'] += 1
        return

    def count_conv_fc_layers(self, m):
        if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
            self.gg['total_fc_conv_layers'] +=1
        return

    def remove_hooks(self, hooks):
        for h in hooks:
            h.remove()
        return

    def orthogonal_weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if hasattr(m, 'weight_v'):
                w_ortho = self.svd_orthonormal(m.weight_v.data.cpu().numpy())
                m.weight_v.data = torch.from_numpy(w_ortho)
                try:
                    nn.init.constant(m.bias, 0)
                except:
                    pass
            else:
                w_ortho = self.svd_orthonormal(m.weight.data.cpu().numpy())
                m.weight.data = torch.from_numpy(w_ortho)
                try:
                    nn.init.constant(m.bias, 0)
                except:
                    pass
        return

    def apply_weights_correction(self, m):
        if self.gg['hook'] is None:
            return
        if not self.gg['correction_needed']:
            return
        if (isinstance(m, nn.Conv2d)) or (isinstance(m, nn.Linear)):
            if self.gg['counter_to_apply_correction'] < self.gg['hook_position']:
                self.gg['counter_to_apply_correction'] += 1
            else:
                if hasattr(m, 'weight_g'):
                    m.weight_g.data *= float(self.gg['current_coef'])
                    self.gg['correction_needed'] = False
                else:
                    m.weight.data *= self.gg['current_coef']
                    self.gg['correction_needed'] = False
                return
        return

    def apply_lsuv_init(self, model, data, needed_std=1.0, std_tol=0.1, max_attempts=10, do_orthonorm=True, cuda=True):
        model.eval()
        if cuda:
            model,data = model.cuda(), data.cuda()
        else:
            model,data = model.cpu(),data.cpu()
            
        model.apply(self.count_conv_fc_layers)
        if do_orthonorm:
            model.apply(self.orthogonal_weights_init)
            if cuda:
                model=model.cuda()
        for layer_idx in range(self.gg['total_fc_conv_layers']):
            model.apply(self.add_current_hook)
            out = model(data)
            current_std = self.gg['act_dict'].std()
            attempts = 0
            while (np.abs(current_std - needed_std) > std_tol):
                self.gg['current_coef'] =  needed_std / (current_std  + 1e-8);
                self.gg['correction_needed'] = True
                model.apply(self.apply_weights_correction)
                if cuda:
                    model=model.cuda()
                out = model(data)
                current_std = self.gg['act_dict'].std()
                attempts+=1
                if attempts > max_attempts:
                    print(f'Cannot converge in {max_attempts} iterations')
                    break
            if self.gg['hook'] is not None:
               self.gg['hook'].remove()
            self.gg['done_counter']+=1
            self.gg['counter_to_apply_correction'] = 0
            self.gg['hook_position'] = 0
            self.gg['hook']  = None
        return model if cuda else model.cpu()


def apply_lsuv_init(model, data, needed_std=1.0, std_tol=0.1, max_attempts=10, do_orthonorm=True, cuda=True):
    return LSUVInitializer().apply_lsuv_init(model, data, needed_std, std_tol, max_attempts, do_orthonorm, cuda)
