import assert from 'node:assert/strict';
import {sentences,posterior,diffusionTrace,corrupt,signals,encodeHierarchy,reconstructHierarchy,generationWork,dependencyTrace} from '../docs/parallel-engine.js';
const near=(a,b)=>assert(Math.abs(a-b)<1e-10,`${a} != ${b}`);
const empty=Array(8).fill(null),p=posterior(empty);
near(p.mass,1);near(p.probabilities[1].find(([w])=>w==='red')[1],.4);near(p.probabilities[2][0][1],.6);
const square=[...empty];square[2]='square';const conditioned=posterior(square);
near(conditioned.probabilities[1].find(([w])=>w==='blue')[1],7/12);
assert(!conditioned.probabilities[1].some(([w])=>w==='red'));
for(let i=0;i<3;i++){
  assert.deepEqual(corrupt(0,i),sentences[i].words);
  assert.deepEqual(corrupt(1,i),empty);
  let previous=0;
  for(let t=0;t<=1.001;t+=.05){const masks=corrupt(t,i).filter(w=>w===null).length;assert(masks>=previous);previous=masks;}
}
for(const rounds of [1,2,4,8]){
  const trace=diffusionTrace(rounds);assert.equal(trace.length,rounds+1);
  for(let s=1;s<trace.length;s++){
    assert.equal(trace[s].visible.filter(w=>w!==null).length,8*s/rounds);
    trace[s-1].visible.forEach((w,i)=>{if(w!==null)assert.equal(trace[s].visible[i],w,'committed tokens never change');});
    for(const distribution of trace[s].probabilities)if(distribution.length)near(distribution.reduce((sum,[,p])=>sum+p,0),1);
  }
  assert(trace.at(-1).visible.every(w=>w!==null));
}
assert.equal(diffusionTrace(1).at(-1).mass,0,'parallel greedy marginals expose unsupported red square');
assert.deepEqual(diffusionTrace(8).at(-1).visible,sentences[1].words,'sequential conditioning yields blue square');
for(const z of Object.values(signals)){
  const encoded=encodeHierarchy(z);assert.deepEqual(encoded.map(h=>h.codes.length),[1,2,4,8]);
  assert.deepEqual(reconstructHierarchy(encoded,4),z);
  for(let k=1;k<4;k++)assert(encoded[k].mse<=encoded[k-1].mse);
  const baseline=reconstructHierarchy(encoded,4);
  // Independent expected influence: +2 to all positions vs +0.5 at only position 0.
  const coarse=reconstructHierarchy(encoded,4,{level:0,position:0,code:encoded[0].codes[0]+1});
  coarse.forEach((v,i)=>near(v,baseline[i]+2));
  const fine=reconstructHierarchy(encoded,4,{level:3,position:0,code:encoded[3].codes[0]+1});
  fine.forEach((v,i)=>near(v,baseline[i]+(i===0?.5:0)));
  assert.deepEqual(reconstructHierarchy(encoded,1,{level:3,position:0,code:0}),reconstructHierarchy(encoded,1),'unrevealed edit has no effect');
}
assert.deepEqual(encodeHierarchy(signals.rising).map(h=>h.mse),[4.5,.5,.25,0]);
const work=generationWork(32,8,2);
assert.deepEqual([work.ar.calls,work.ar.positions,work.ar.pairs],[32,32,528]);
assert.deepEqual([work.diffusion.calls,work.diffusion.positions,work.diffusion.pairs],[8,256,8192]);
assert.deepEqual([work.ncm.calls,work.ncm.positions,work.ncm.pairs],[6,63,1365]);
for(const n of [16,32,64,128])for(const s of [2,4,8,16,32,64])for(const g of [2,4]){
  const w=generationWork(n,s,g);assert.equal(w.ncm.widths.at(-1),n);assert(w.ncm.widths.every((v,i,a)=>i===0||v>a[i-1]));
}
for(let j=0;j<8;j++)for(const delta of [-4,0,4])for(const causal of [false,true]){
  const trace=dependencyTrace(j,delta,causal);
  trace.differences.forEach((v,i)=>near(v,(i===j?delta:0)+(causal?(j<=i?delta/(i+1):0):delta/8)));
}
near(dependencyTrace(6,4,false).differences[2],.5);
near(dependencyTrace(6,4,true).differences[2],0);
console.log('PASS: diffusion conditionals, corruption endpoints, simultaneous dependency failure, sequential recovery, residual reconstruction, code influence, work counts and causal/bidirectional KV dependencies.');
