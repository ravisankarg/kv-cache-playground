import assert from 'node:assert/strict';
import fs from 'node:fs';
import {dot,mv,softmax,tv,rotate,argmax,run,prefill,generate,step,cloneCache,speculativeMass,quantize,onlineAttention,tokenize} from '../docs/engine.js';
const model=JSON.parse(fs.readFileSync(new URL('../docs/data/course_model.json',import.meta.url)));
const ids=model.reference.prompt.map(w=>model.vocab.indexOf(w));
const inc=run(model,ids),full=prefill(model,ids),error=(a,b)=>Math.max(...a.map((v,i)=>Math.abs(v-b[i])));
assert(error(inc.last.logits,model.reference.last_logits)<5e-4,'JavaScript decoder must match independently computed CUDA logits');
assert(error(inc.last.logits,full.logits.at(-1))<1e-9,'matrix prefill must match cached decoder');
for(let i=0;i<ids.length;i++)for(const tr of inc.steps[i].traces){
  assert.equal(tr.cache.length,i+1);
  for(const a of tr.attention)assert(Math.abs(a.reduce((s,v)=>s+v,0)-1)<1e-10);
}
// Future-token changes must not influence earlier logits or cached rows.
const changed=[...ids];changed[18]=model.vocab.indexOf('india');
const other=prefill(model,changed);
for(let t=0;t<18;t++)assert(error(full.logits[t],other.logits[t])<1e-10);
const c=cloneCache(inc.caches),before=JSON.stringify(c),y=argmax(inc.last.p);
step(model,y,c,ids.length);step(model,y,c,ids.length+1);
const rolled=c.map(rows=>rows.slice(0,ids.length));assert.equal(JSON.stringify(rolled),before);
assert.throws(()=>step(model,y,c,32),/cap/);
assert(tokenize('moon',model).unknown.includes('moon'));assert.deepEqual(tokenize('moon',model).ids,[3]);
for(let n=0;n<50;n++){
  const p=softmax([n/10,0,-.2,.8]),q=softmax([.2,n/23,-.3,1]);
  const r=speculativeMass(p,q);assert(error(p,r.output)<1e-12);
}
assert.equal(speculativeMass([.5,.5],[.5,.5]).alpha,1);
assert(error(speculativeMass([1,0],[0,1]).output,[1,0])<1e-12);
const tr=inc.last.traces[1];
for(const B of [1,2,4,8,32])assert(error(onlineAttention(tr.scores[0],tr.cache.map(r=>r.v[0]),B).output,tr.weightedValues[0])<1e-10);
const u=[.2,3,-1,.8],v=[1,.3,1.5,-2];assert(Math.abs(dot(u,v)-dot(rotate(u,11),rotate(v,11)))<1e-10);
const constant=quantize([2,2,2],2);assert.deepEqual(constant.values,[2,2,2]);
const out=generate(model,ids,3).generated.map(g=>g.token);
console.log('Preset generated:',out.join(' '));
assert.deepEqual(out,['paris','tokyo','<eos>'],'default context/question should be answered');
const swap=[...ids];swap[5]=model.vocab.indexOf('tokyo');swap[9]=model.vocab.indexOf('paris');
assert.deepEqual(generate(model,swap,3).generated.map(g=>g.token),['tokyo','paris','<eos>']);
const reverse=[...ids];reverse[17]=model.vocab.indexOf('japan');reverse[18]=model.vocab.indexOf('france');
assert.deepEqual(generate(model,reverse,3).generated.map(g=>g.token),['tokyo','paris','<eos>']);
assert(model.checks.next_token_accuracy>.95,'held-out recall must be reliable for teaching');
assert(model.checks.rollback_exact);
assert(model.transfer.results.some(r=>r.mode==='basis'&&r.mapped_tv<.01));
console.log('PASS: CUDA/browser parity, independent matrix prefill, causal invariance, rollback, context cap, tokenizer, exact speculative mass, online softmax, RoPE, quantization, recall presets, held-out recall and controlled transfer.');
