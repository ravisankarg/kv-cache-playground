#!/usr/bin/env python3
"""Train a two-layer associative-recall decoder; export weights and audited labs.

CUDA is required. Random key/value assignments make context necessary. This is
a synthetic task, not a pretrained language model or a production benchmark.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

VOCAB = ['<pad>', '<bos>', '<eos>', '<unk>', 'context', ':', '=', ';',
         'user', '?', 'assistant', 'france', 'japan', 'india', 'italy', 'egypt',
         'peru', 'paris', 'tokyo', 'delhi', 'rome', 'cairo', 'lima']
IDS = {w: i for i, w in enumerate(VOCAB)}
D, H, HD, L = 32, 4, 8, 2


def norm(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)


def rope(x, positions, inverse=False):
    angle = positions.to(x.dtype)[:, None] * 10000 ** (-torch.arange(0, HD, 2, device=x.device) / HD)
    if inverse:
        angle = -angle
    c, s = angle.cos()[None, :, None], angle.sin()[None, :, None]
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a*c-b*s, a*s+b*c), -1).flatten(-2)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        for name, a, b in [('q', D, D), ('k', D, D), ('v', D, D), ('o', D, D),
                           ('gate', D, 64), ('up', D, 64), ('down', 64, D)]:
            setattr(self, name, nn.Linear(a, b, bias=False))

    def forward(self, x, cache=None, capture=False):
        b, t, _ = x.shape
        start = 0 if cache is None else cache['k'].shape[1]
        positions = torch.arange(start, start+t, device=x.device)
        z = norm(x)
        q = rope(self.q(z).view(b, t, H, HD), positions)
        kc = self.k(z).view(b, t, H, HD)
        k = rope(kc, positions)
        v = self.v(z).view(b, t, H, HD)
        if cache is not None:
            k = torch.cat((cache['k'], k), 1)
            v = torch.cat((cache['v'], v), 1)
        scores = torch.einsum('bthd,bshd->bhts', q, k) / math.sqrt(HD)
        mask = torch.arange(k.shape[1], device=x.device)[None, :] > positions[:, None]
        a = scores.masked_fill(mask[None, None], -torch.inf).softmax(-1)
        attended = torch.einsum('bhts,bshd->bthd', a, v).reshape(b, t, D)
        r = x + self.o(attended)
        z = norm(r)
        y = r + self.down(F.silu(self.gate(z)) * self.up(z))
        new = {'k': k, 'v': v}
        trace = {'input': x, 'q': q, 'k_content': kc, 'k': k, 'v': v, 'scores': scores,
                 'attention': a, 'weighted_values': attended, 'after_attention': r, 'output': y} if capture else None
        return y, new, trace


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(VOCAB), D)
        self.position = nn.Embedding(32, D)
        self.blocks = nn.ModuleList([Block() for _ in range(L)])
        self.head = nn.Linear(D, len(VOCAB), bias=False)
        self.mtp = nn.Linear(D, len(VOCAB), bias=False)

    def forward(self, ids, caches=None, capture=False):
        start = 0 if caches is None else caches[0]['k'].shape[1]
        pos = torch.arange(start,start+ids.shape[1],device=ids.device)
        x = self.embedding(ids) + self.position(pos)
        updated, traces = [], []
        for i, block in enumerate(self.blocks):
            x, c, tr = block(x, None if caches is None else caches[i], capture)
            updated.append(c)
            traces.append(tr)
        return self.head(norm(x)), self.mtp(norm(x)), updated, traces


def example(rng, split='train'):
    # Fixed key slots keep the small teaching task learnable with two layers;
    # values and both queried keys vary. This restriction is shown in the UI.
    keys = VOCAB[11:14]
    while True:
        vals = rng.sample(VOCAB[17:23], 3)
        heldout = sum((7,3,1)[i] * VOCAB[17:23].index(v) for i,v in enumerate(vals)) % 5 == 0
        if heldout == (split=='test'):
            break
    query = rng.sample(range(3), 2)
    words = ['<bos>', 'context', ':']
    for key, val in zip(keys, vals):
        words += [key, '=', val, ';']
    words += ['user', ':', keys[query[0]], keys[query[1]], '?', 'assistant', ':']
    prompt = len(words)
    words += [vals[query[0]], vals[query[1]], '<eos>']
    return [IDS[w] for w in words], prompt


def aslist(x):
    return x.detach().cpu().tolist()


def clone(caches):
    return [{k: v.clone() for k, v in c.items()} for c in caches]


@torch.no_grad()
def audit(model, device):
    rng = random.Random(1904)
    items = [example(rng, 'test') for _ in range(256)]
    ids = torch.tensor([x[0] for x in items], device=device)
    logits, mtp, _, _ = model(ids[:, :22])
    answers = ids[:, 22:24]
    first = logits[:, -1].argmax(-1)
    second = mtp[:, -1].argmax(-1)
    l2, _, _, _ = model(ids[:, :23])
    generated_second = l2[:, -1].argmax(-1)
    # Full-prefix and token-by-token produce the same causal decoder states.
    prompt = ids[:1, :22]
    full, _, full_caches, _ = model(prompt)
    caches = None
    for j in range(prompt.shape[1]):
        inc, _, caches, _ = model(prompt[:, j:j+1], caches)
    err = (full[:, -1]-inc[:, -1]).abs().max().item()
    cache_err = max((a[k]-b[k]).abs().max().item() for a,b in zip(full_caches,caches) for k in ('k','v'))
    assert err < 2e-4 and cache_err < 2e-4
    # Batched candidate verification has the same logits as serial execution.
    candidate = ids[:1, 22:24]
    batched, _, speculative_cache, _ = model(candidate, clone(full_caches))
    c = clone(full_caches)
    serial = []
    for j in range(2):
        o, _, c, _ = model(candidate[:, j:j+1], c)
        serial.append(o)
    verify_err = (batched-torch.cat(serial,1)).abs().max().item()
    assert verify_err < 2e-4
    # Rollback is truncation of both K and V in every layer.
    rolled = [{k:v[:,:22].clone() for k,v in c.items()} for c in speculative_cache]
    assert all(torch.equal(a[k], b[k]) for a,b in zip(rolled,full_caches) for k in ('k','v'))
    return {'heldout_examples':256, 'next_token_accuracy':(first==answers[:,0]).float().mean().item(),
            'autoregressive_second_accuracy':(generated_second==answers[:,1]).float().mean().item(),
            'mtp_second_accuracy':(second==answers[:,1]).float().mean().item(),
            'prefill_vs_cached_max_logit_error':err, 'prefill_vs_cached_max_cache_error':cache_err,
            'batch_verification_max_logit_error':verify_err, 'rollback_exact':True}


@torch.no_grad()
def transfer_lab(target, device):
    # A controlled invertible basis change and an independently initialized
    # source decoder deliberately test different levels of compatibility.
    torch.manual_seed(941)
    source = Decoder().to(device).eval()
    mats = [torch.linalg.qr(torch.randn(HD, HD, device=device)).Q for _ in range(2)]
    rng = random.Random(2410)
    train = torch.tensor([example(rng)[0][:21] for _ in range(96)], device=device)
    test = torch.tensor([example(rng, 'test')[0][:21] for _ in range(32)], device=device)
    _, _, tc, _ = target(train)
    _, _, sc, _ = source(train)
    _, _, te, _ = target(test)
    _, _, se, _ = source(test)
    pos = torch.arange(21,device=device)
    colon = torch.full((32,1), IDS[':'], device=device)
    true_logits, _, _, _ = target(colon, te)
    p = true_logits[:,-1].softmax(-1)
    results = []
    for mode in ('basis', 'independent'):
        for lam in (0.0001, 0.01, 1.0, 100.0):
            mapped, wrong, residuals = [], [], []
            for li in range(L):
                out, raw = {}, {}
                for ki, key in enumerate(('k','v')):
                    y = rope(tc[li][key],pos,True) if key=='k' else tc[li][key]
                    yt = rope(te[li][key],pos,True) if key=='k' else te[li][key]
                    if mode=='basis':
                        x, xt = y @ mats[ki], yt @ mats[ki]
                    else:
                        x = rope(sc[li][key],pos,True) if key=='k' else sc[li][key]
                        xt = rope(se[li][key],pos,True) if key=='k' else se[li][key]
                    preds = []
                    for h in range(H):
                        a,b = x[:,:,h].reshape(-1,HD), y[:,:,h].reshape(-1,HD)
                        w = torch.linalg.solve(a.T@a+lam*torch.eye(HD,device=device), a.T@b)
                        pred = xt[:,:,h] @ w
                        preds.append(pred)
                    pred = torch.stack(preds,2)
                    residuals.append((pred-yt).square().mean().item())
                    out[key] = rope(pred,pos) if key=='k' else pred
                    raw[key] = rope(xt,pos) if key=='k' else xt
                mapped.append(out)
                wrong.append(raw)
            lm, _, _, _ = target(colon,mapped)
            lw, _, _, _ = target(colon,wrong)
            pm, pw = lm[:,-1].softmax(-1), lw[:,-1].softmax(-1)
            results.append({'mode':mode,'lambda':lam,'cache_mse':sum(residuals)/len(residuals),
                            'mapped_tv':((pm-p).abs().sum(-1)/2).mean().item(),
                            'raw_tv':((pw-p).abs().sum(-1)/2).mean().item(),
                            'next_token_agreement':(pm.argmax(-1)==p.argmax(-1)).float().mean().item(),
                            'target_p':aslist(p[0]),'mapped_p':aslist(pm[0]),'raw_p':aslist(pw[0])})
    return {'calibration_prompts':96,'test_prompts':32,'results':results,
            'scope':'basis = exact information-preserving reparameterization; independent = another randomly initialized decoder. Neither is a pretrained LLM transfer benchmark.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=1600)
    parser.add_argument('--output',type=Path,default=Path('docs/data/course_model.json'))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required; no CPU fallback.')
    torch.manual_seed(991)
    torch.set_num_threads(2)
    device = torch.device('cuda')
    model = Decoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.003,weight_decay=0.01)
    rng = random.Random(331)
    history, start = [], time.time()
    for step in range(args.steps):
        ids = torch.tensor([example(rng)[0] for _ in range(64)], device=device)
        logits, mtp, _, _ = model(ids[:,:-1])
        loss1 = F.cross_entropy(logits[:,21:24].reshape(-1,len(VOCAB)),ids[:,22:25].reshape(-1))
        loss2 = F.cross_entropy(mtp[:,21:23].reshape(-1,len(VOCAB)),ids[:,23:25].reshape(-1))
        grammar = F.cross_entropy(logits.reshape(-1,len(VOCAB)),ids[:,1:].reshape(-1))
        loss = loss1 + 0.3*loss2 + 0.2*grammar
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if step % 200 == 0 or step==args.steps-1:
            entry = {'step':step+1,'next_loss':loss1.item(),'mtp_loss':loss2.item()}
            history.append(entry)
            print(entry,flush=True)
    model.eval()
    checks = audit(model,device)
    # Linear weights are transposed on export: browser uses row vectors x W.
    weights = {'embedding':aslist(model.embedding.weight),'position':aslist(model.position.weight),
               'head':aslist(model.head.weight.T),'mtp':aslist(model.mtp.weight.T),
               'blocks':[{name:aslist(getattr(b,name).weight.T) for name in ('q','k','v','o','gate','up','down')} for b in model.blocks]}
    preset = ['<bos>','context',':','france','=','paris',';','japan','=','tokyo',';',
              'india','=','delhi',';','user',':','france','japan','?','assistant',':']
    ids = torch.tensor([[IDS[w] for w in preset]],device=device)
    logits, _, _, _ = model(ids)
    payload = {'schema':1,'vocab':VOCAB,'config':{'d_model':D,'heads':H,'head_dim':HD,'layers':L,'max_context':32,'mlp_dim':64},
               'weights':weights,'training':{'seed':991,'steps':args.steps,'batch_size':64,'examples':args.steps*64,
                 'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,
                 'seconds':time.time()-start,'history':history,'task':'fixed keys france/japan/india in that order; random distinct values; recall two requested values in order',
                 'split':'weighted value-index sum modulo 5 equals 0: test only; all other assignments: train only'},
               'checks':checks,'reference':{'prompt':preset,'last_logits':aslist(logits[0,-1])},
               'transfer':transfer_lab(model,device)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,separators=(',',':'))+'\n')
    print(json.dumps(checks,indent=2),flush=True)
    print(f'Wrote {args.output}',flush=True)


if __name__=='__main__':
    main()
