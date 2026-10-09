import {dot,softmax,mv,add,tv,argmax,run,generate,prefill,step,cloneCache,speculativeMass,quantize,onlineAttention,tokenize} from './engine.js';
const $=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const f=(v,n=3)=>Number(v).toFixed(n), pct=v=>`${f(v*100,1)}%`, sci=v=>Number(v).toExponential(2);
const colors=['#13745d','#426cb5','#ba8738','#9b669d'];
let model,gpu,prompt,generated,selected=21;
const set=(id,html)=>{$(id).innerHTML=html;};
const metric=(v,label,small='')=>`<div class="metric"><b>${esc(v)}</b><span>${esc(label)}</span>${small?`<br><small>${esc(small)}</small>`:''}</div>`;
const vector=v=>`[${v.map(x=>f(x,3)).join(', ')}]`;
function barHTML(labels,values,color=colors[0],max=1){return labels.map((label,i)=>`<div class="bar-row"><span class="bar-label" title="${esc(label)}">${esc(label)}</span><div class="track"><span class="fill" style="width:${Math.max(0,Math.min(100,values[i]/max*100))}%;background:${color}"></span></div><span class="number">${f(values[i],3)}</span></div>`).join('');}
function probBars(id,p){const order=p.map((_,i)=>i).sort((a,b)=>p[b]-p[a]).slice(0,6);set(id,barHTML(order.map(i=>model.vocab[i]),order.map(i=>p[i])));}
function multiBars(id,labels,series,max=1){set(id,`<div class="bar-key">${series.map((s,i)=>`<span><i style="background:${colors[i]}"></i>${esc(s.name)}</span>`).join('')}</div>`+labels.map((label,j)=>`<div class="group-bars"><strong>${esc(label)}</strong>${series.map((s,i)=>barHTML([s.name],[s.values[j]],colors[i],max)).join('')}</div>`).join(''));}
function topMulti(id,series){const order=model.vocab.map((_,i)=>i).sort((a,b)=>Math.max(...series.map(s=>s.values[b]))-Math.max(...series.map(s=>s.values[a]))).slice(0,5);multiBars(id,order.map(i=>model.vocab[i]),series.map(s=>({...s,values:order.map(i=>s.values[i])})));}
function matrix(id,rows){set(id,`<div class="matrix">${rows.map(row=>`<div class="matrix-row">${row.map(v=>`<span class="matrix-cell" style="background:${v<0?'#dbe5f3':'#d9ecdc'}" title="${f(v,6)}">${f(v,2)}</span>`).join('')}</div>`).join('')}</div>`);}
function heat(id,rows,labels){const max=Math.max(...rows.flat(),1e-6);set(id,`<div class="heat-scroll"><div class="heat" style="grid-template-columns:30px repeat(${labels.length},minmax(23px,1fr))">${rows.map((row,h)=>`<span class="heat-cell">H${h+1}</span>${row.map((v,j)=>`<span class="heat-cell" style="background:rgba(19,116,93,${.08+.7*v/max});color:${v/max>.65?'white':'#193e30'}" title="Head ${h+1} · ${esc(labels[j])} · ${f(v,6)}">${f(v,2)}</span>`).join('')}`).join('')}<span></span>${labels.map(l=>`<span class="heat-label">${esc(l)}</span>`).join('')}</div></div>`);}
function slots(id,positions,retained,newpos=null,blocked=null){set(id,positions.map(p=>`<span class="slot ${retained.includes(p)?'':'dropped'} ${p===newpos?'new':''} ${p===blocked?'blocked':''}" title="Position ${p}: ${retained.includes(p)?'retained':'not retained'}">${p}${p===newpos?'<small>new</small>':''}</span>`).join(''));}
function bind(id,fn,event='input'){$(id).addEventListener(event,fn);}
function buildPrompt(){
  const tok=tokenize(`<bos> context : ${$('context-text').value} user : ${$('ask-text').value} ? assistant :`,model);
  if(tok.ids.length>model.config.max_context-3){$('prompt-warning').textContent=`Prompt is ${tok.ids.length} tokens. Leave at least 3 slots inside the 32-token cap; shorten the context.`;return;}
  let note=tok.unknown.length?`Unknown tokens ${[...new Set(tok.unknown)].join(', ')} become <unk>; this is outside the toy's training task.`:'';
  const format=/^(france|japan|india|italy|egypt|peru)\s*=\s*(paris|tokyo|delhi|rome|cairo|lima)\s*;\s*(france|japan|india|italy|egypt|peru)\s*=\s*(paris|tokyo|delhi|rome|cairo|lima)\s*;\s*(france|japan|india|italy|egypt|peru)\s*=\s*(paris|tokyo|delhi|rome|cairo|lima)\s*;\s*$/i;
  const matches=$('context-text').value.trim().toLowerCase().match(format),ask=$('ask-text').value.trim().toLowerCase().split(/\s+/);
  if(!matches || [matches[1],matches[3],matches[5]].join(' ')!=='france japan india' || new Set([matches[2],matches[4],matches[6]]).size!==3 || ask.length!==2 || new Set(ask).size!==2 || !ask.every(w=>['france','japan','india'].includes(w)))note+=' Use the fixed-order three-fact, two-distinct-key training format for interpretable recall.';
  $('prompt-warning').textContent=note;
  prompt={...tok,...run(model,tok.ids)};
  generated=generate(model,tok.ids,3);selected=tok.ids.length-1;
  $('token-step').max=generated.steps.length-1;$('token-step').value=selected;
  set('generation',generated.generated.map((g,i)=>metric(g.token,`Predicted output ${i+1}`,`p = ${f(g.p,4)}`)).join(''));
  renderDecoder();renderDependents();
}
function renderDecoder(){
  const st=generated.steps[selected],li=+$('decoder-layer').value,h=+$('decoder-head').value,tr=st.traces[li];
  $('token-position').textContent=`position ${selected} · ${selected<prompt.ids.length?'prefill input':'fed-back output'} · ${st.token}`;
  set('prompt-tokens',generated.steps.map((s,i)=>`<span class="token ${i===selected?'current':''} ${i>selected?'future':''} ${i<15?'context':i<21?'ask':'new'}" data-position="${i}" title="Token ID ${s.id}" tabindex="0" role="button" aria-label="Inspect token ${esc(s.token)} at position ${i}">${esc(s.token)}<small>${i} · id ${s.id}</small></span>`).join(''));
  $('prompt-tokens').querySelectorAll('[data-position]').forEach(el=>{const choose=()=>{selected=+el.dataset.position;$('token-step').value=selected;renderDecoder();};el.onclick=choose;el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}};});
  set('layer-flow',st.traces.map((t,i)=>`<div class="flow-node ${i===li?'active':''}"><b>Layer ${i+1}</b>RMSNorm → attention → MLP<br><small>h′ first 3: ${vector(t.output.slice(0,3))}</small></div>`).join('<span class="flow-arrow">→</span>')+'<span class="flow-arrow">→</span><div class="flow-node"><b>Vocabulary head</b>RMSNorm → 23 logits</div>');
  probBars('vocab-bars',st.p);
  set('decoder-cache',st.traces.map((t,i)=>`<div class="flow-node"><b>Layer ${i+1}: ${t.cache.length} rows</b>K [${t.cache.length}, 4, 8] · V [${t.cache.length}, 4, 8]<br>${t.cache.length*2*4*8*2} hypothetical FP16 bytes</div>`).join(''));
  $('decoder-explain').textContent=`Processed “${st.token}” at position ${selected}. Both layers now have its K/V row. The largest next-token probability is “${model.vocab[argmax(st.p)]}”. This prediction has no new cache row until it is fed through both layers. Q is temporary; K/V remain reusable.`;
  const labels=tr.cache.map(r=>`${r.position}:${r.token}`);
  const scores=tr.scores[h],a=tr.attention[h],m=Math.max(...scores),sum=scores.reduce((s,x)=>s+Math.exp(x-m),0);
  const best=argmax(a),row=tr.cache[best];
  set('attention-equation',`q · k(position ${row.position}) = ${f(dot(tr.q[h],row.k[h]),5)}\nscore = dot / √8 = ${f(scores[best],5)}\nweight = exp(${f(scores[best],3)} − ${f(m,3)}) / ${f(sum,5)} = ${f(a[best],5)}\nΣ weights = ${f(a.reduce((s,v)=>s+v,0),8)} · output coordinate 0 = Σ aⱼ Vⱼ[0] = ${f(tr.weightedValues[h][0],5)}`);
  set('attention-table','<thead><tr><th>Position · token</th><th>Q · K</th><th>Scaled score</th><th>Softmax weight</th><th>V[0]</th><th>a × V[0]</th></tr></thead><tbody>'+tr.cache.map((r,j)=>`<tr><td>${esc(labels[j])}</td><td>${f(dot(tr.q[h],r.k[h]),4)}</td><td>${f(scores[j],4)}</td><td class="weight">${f(a[j],5)}</td><td>${f(r.v[h][0],4)}</td><td>${f(a[j]*r.v[h][0],5)}</td></tr>`).join('')+'</tbody>');
  heat('attention-heatmap',tr.attention,labels);matrix('query-vector',[tr.q[h]]);matrix('weighted-vector',[tr.weightedValues[h]]);
}
function renderOverflow(){const policy=$('overflow-policy').value,p=+$('overflow-step').value,blocked=policy==='strict'&&p>=8,positions=blocked?Array.from({length:8},(_,i)=>i):gpu.variants.mha[policy].steps[p].layers[0].cache.positions;
  $('overflow-label').textContent=`position ${p} · ${blocked?'blocked':'processed'}`;
  slots('overflow-slots',Array.from({length:10},(_,i)=>i),positions,blocked?null:p,blocked?p:null);
  $('overflow-result').textContent=blocked?`The strict runtime stops before position ${p}. Cache stays at positions 0–7. Position 7's logits may select the next token, but processing it is blocked.`:policy==='sliding'&&p>=8?`Processing position ${p} evicts position ${p-8}; each layer retains [${positions.join(', ')}]. Its query attends to those 8 rows. Absolute positions continue increasing.`:p===7?'The final legal position writes row 7 and attends to all 8 rows. Selecting a next token and storing that token’s K/V are separate events.':`Position ${p} writes one new row in both layers; ${p+1} rows are now available.`;
}
function renderArchitectures(){const v=$('arch').value,s=+$('arch-step').value,l=+$('arch-layer').value,b=+$('arch-bytes').value,tr=gpu.variants[v].strict.steps[s].layers[l],c=tr.cache;
  const G={mha:4,gqa:2,mqa:1}[v];
  set('head-sharing',Array.from({length:4},(_,h)=>`<div class="flow-node"><b>Query head ${h+1}</b>${v==='mla'?`own content projection<br>shared cKV + kRoPE`:`reads KV head ${Math.floor(h/(4/G))+1}`}</div>`).join(''));
  const names=['mha','gqa','mqa','mla'],bytes=names.map(n=>gpu.variants[n].strict.scalars_per_token_all_layers*b*(s+1));
  set('arch-memory',barHTML(names.map(n=>n.toUpperCase()),bytes,colors[0],Math.max(...bytes))+'<p class="hint">Bytes for '+(s+1)+' cached tokens across BOTH layers. Unallocated memory and temporary expanded tensors are excluded.</p>');
  set('arch-shapes',Object.entries(c.shapes).map(([name,shape])=>`${name}: [${shape.join(', ')}]`).join(' · ')+`\n${c.scalars_per_token} stored scalars / token / layer · ${c.scalars_per_token*b} bytes / token / layer`);
  heat('arch-attention',tr.weights_by_head,c.positions.map(p=>`${p}:${gpu.token_sequence[p]}`));matrix('arch-k',c.k);matrix('arch-v',c.v);
  $('arch-note').textContent=v==='mla'?'Stored A = shared content latent (first 4 of 8 dimensions). Stored B = positional key (all 2 dimensions). Four heads reconstruct their own content K/V in this teaching trace. Efficient MLA can instead use the absorption identity below.':`Stored A = K, stored B = V: head 1, first 4 of 8 dimensions. This ${v.toUpperCase()} trace has ${G} KV heads and 4 query heads. All head weights are shown above. Separate random projections make this a layout experiment, not a quality comparison.`;
}
function renderLatent(){const q=[+$('latent-q').value,-.5],C=[[1,0],[0,1],[1,1]],UK=[[1,2],[-1,1]],UV=[[2,0],[1,3]];
  const K=C.map(c=>mv(c,UK)),V=C.map(c=>mv(c,UV)),absorbed=UK.map(row=>dot(q,row));
  const se=K.map(k=>dot(q,k)/Math.sqrt(2)),sa=C.map(c=>dot(absorbed,c)/Math.sqrt(2)),a=softmax(se);
  const expanded=[0,1].map(d=>a.reduce((s,w,i)=>s+w*V[i][d],0)),latent=[0,1].map(d=>a.reduce((s,w,i)=>s+w*C[i][d],0)),out=mv(latent,UV);
  $('latent-q-label').textContent=`q = ${vector(q)}`;
  set('latent-data',`C = [[1,0], [0,1], [1,1]]\nUᴷ = [[1,2], [-1,1]] · Uⱽ = [[2,0], [1,3]]\nExpanded K = ${JSON.stringify(K)} · V = ${JSON.stringify(V)}\nAbsorbed query q(Uᴷ)ᵀ = ${vector(absorbed)}\nWeighted latent Σaⱼcⱼ = ${vector(latent)}`);
  multiBars('latent-bars',['row 0','row 1','row 2'],[{name:'expanded',values:softmax(se)},{name:'absorbed',values:softmax(sa)}]);
  set('latent-result',metric(vector(expanded),'expanded weighted values')+metric(vector(out),'aggregate latent → Uⱽ')+metric(sci(Math.max(...out.map((x,i)=>Math.abs(x-expanded[i])))),'max output error'));
}
function renderSpec(){const ids=prompt.ids,base=run(model,ids),c=cloneCache(base.caches),p1=base.last.p,y1=argmax(p1),after1=step(model,y1,c,ids.length),truth2=argmax(after1.p);
  let y2=truth2;if($('draft-mode').value==='wrong')y2=model.vocab.indexOf('paris')===truth2?model.vocab.indexOf('tokyo'):model.vocab.indexOf('paris');
  const after2=step(model,y2,c,ids.length+1),accepted=y2===truth2?2:1,bonus=y2===truth2?argmax(after2.p):truth2;
  const rolled=c.map(rows=>rows.slice(0,ids.length+accepted));
  set('spec-tokens',`<div class="flow-node"><b>Candidate 1 · ${esc(model.vocab[y1])}</b>prefix logits → accepted</div><span class="flow-arrow">→</span><div class="flow-node ${accepted===1?'rejected':''}"><b>Candidate 2 · ${esc(model.vocab[y2])}</b>after candidate 1 → ${accepted===2?'accepted':'reject'}</div><span class="flow-arrow">→</span><div class="flow-node"><b>${accepted===2?'Bonus':'Correction'} · ${esc(model.vocab[bonus])}</b>selected; not cached yet</div>`);
  slots('spec-cache',Array.from({length:ids.length+2},(_,i)=>i),rolled[0].map(r=>r.position),ids.length+accepted-1);
  $('spec-result').textContent=`Verified two candidates. Accepted ${accepted}; tentative cache had ${ids.length+2} rows in each layer. After rollback it has ${rolled[0].length}, preserving exactly the prefix + accepted candidates. ${accepted+1} tokens are selected this cycle (including ${accepted===2?'bonus':'correction'}), without claiming a runtime speedup.`;
}
function renderSampling(){const mix=+$('draft-mix').value,p=[.55,.25,.15,.05],bad=[.1,.1,.2,.6],q=p.map((v,i)=>v*(1-mix)+bad[i]*mix),r=speculativeMass(p,q);
  $('draft-mix-label').textContent=`${pct(mix)} mixture with a mismatched draft`;
  let seed=8149,counts=[0,0,0,0];const rand=()=>{seed=(1664525*seed+1013904223)>>>0;return seed/4294967296;},sample=ps=>{let u=rand(),s=0;for(let i=0;i<ps.length;i++){s+=ps[i];if(u<s)return i;}return ps.length-1;};
  for(let i=0;i<20000;i++){let y=sample(q);if(rand()>Math.min(1,p[y]/q[y]))y=sample(r.residual);counts[y]++;}
  multiBars('sampling-bars',['A','B','C','D'],[{name:'target p',values:p},{name:'draft q',values:q},{name:'corrected',values:r.output},{name:'empirical',values:counts.map(n=>n/20000)}]);
  set('sampling-result',metric(pct(r.alpha),'expected acceptance')+metric(sci(Math.max(...p.map((v,i)=>Math.abs(v-r.output[i])))),'analytic max error')+metric(f(tv(p,counts.map(n=>n/20000)),4),'empirical TV · 20k trials'));
}
function renderMTP(){const p=prompt.last.p,m=prompt.last.mtp,y1=argmax(p),c=cloneCache(prompt.caches),second=step(model,y1,c,prompt.ids.length).p;
  probBars('mtp-next',p);probBars('mtp-second',m);
  $('mtp-result').textContent=`From the same final prompt hidden state: head 1 predicts “${model.vocab[y1]}”; head 2 predicts “${model.vocab[argmax(m)]}”. After actually feeding head 1’s token, the base decoder predicts “${model.vocab[argmax(second)]}” with a conditional distribution ${f(tv(m,second),4)} TV away from the auxiliary marginal. Auxiliary tokens create no K/V until processed.`;
  set('mtp-audit',metric(pct(model.checks.next_token_accuracy),'held-out next token')+metric(pct(model.checks.mtp_second_accuracy),'held-out MTP second token')+metric(pct(model.checks.autoregressive_second_accuracy),'teacher-forced base second token'));
}
function renderTransfer(){const mode=$('transfer-mode').value,lam=+$('ridge').value,r=model.transfer.results.find(r=>r.mode===mode&&r.lambda===lam);
  set('transfer-metrics',metric(sci(r.cache_mse),'mean cache MSE')+metric(f(r.raw_tv,4),'raw copy · mean output TV')+metric(f(r.mapped_tv,4),'mapped · mean output TV')+metric(pct(r.next_token_agreement),'next-token agreement'));
  topMulti('transfer-bars',[{name:'target',values:r.target_p},{name:'raw copy',values:r.raw_p},{name:'ridge map',values:r.mapped_p}]);
  $('transfer-result').textContent=mode==='basis'?`An orthogonal basis change preserves all cache information. With a sufficiently small ridge penalty, the learned inverse should recover the target nearly exactly. Larger λ shrinks the solution and can harm continuation despite compatible dimensions.`:`Identical dimensions and token IDs do not make independently initialized hidden spaces interchangeable. Per-head ridge calibration may lower cache MSE, yet cannot guarantee attention or continuation equivalence. The held-out metrics expose that limitation.`;
}
function renderPrefix(){const a=tokenize('<bos> context : france = paris ; japan = tokyo ; india = delhi ; user : france japan ? assistant :',model).words,b=[...a];if($('prefix-mode').value==='question')b[17]='india';else b[5]='rome';
  let p=0;while(p<a.length&&a[p]===b[p])p++;const B=+$('prefix-block').value,reuse=Math.floor(p/B)*B;
  set('prefix-tokens',`<h5>Request A</h5><div class="tokens">${a.map((w,i)=>`<span class="token ${i<reuse?'context':'future'}">${esc(w)}<small>${i}</small></span>`).join('')}</div><h5>Request B · green = reusable full-block rows</h5><div class="tokens">${b.map((w,i)=>`<span class="token ${i<reuse?'current':''}">${esc(w)}<small>${i}</small></span>`).join('')}</div>`);
  $('prefix-result').textContent=`Exact shared prefix: ${p} tokens. With ${B}-token full blocks, reuse ${reuse} rows and process ${b.length-reuse} suffix rows. The first changed position is ${p}; matching text after it cannot restore cached equality.`;
}
function eviction(){const n=prompt.ids.length,budget=Math.min(+$('evict-budget').value,n),p=n-1,previous=run(model,prompt.ids.slice(0,-1)),all=previous.caches[0].map(r=>r.position),policy=$('evict-policy').value;
  let keep;
  if(policy==='recent')keep=all.slice(-(budget-1));
  else if(policy==='sink')keep=[0,...all.slice(-(budget-2))];
  else{const importance=all.map(()=>0);for(const st of previous.steps)for(const tr of st.traces)for(const a of tr.attention)for(let j=0;j<a.length;j++)importance[j]+=a[j];
    const recent=all.slice(-2),rank=all.filter(i=>!recent.includes(i)).sort((a,b)=>importance[b]-importance[a]);keep=[...recent,...rank.slice(0,budget-3)].sort((a,b)=>a-b);}
  const result=step(model,prompt.ids[p],previous.caches,p,{keep:{at:p,positions:keep}}),positions=[...keep,p].sort((a,b)=>a-b);
  $('evict-label').textContent=`${budget} total rows / ${n} original`;
  slots('evict-slots',Array.from({length:n},(_,i)=>i),positions,p);
  topMulti('evict-bars',[{name:'full cache',values:prompt.last.p},{name:'evicted',values:result.p}]);
  $('evict-result').textContent=`Retained positions [${positions.join(', ')}]. Next-token TV changes by ${f(tv(prompt.last.p,result.p),4)}; full-cache choice “${model.vocab[argmax(prompt.last.p)]}”, evicted-cache choice “${model.vocab[argmax(result.p)]}”. Other surviving rows still encode their original causal histories.`;
}
function quantLab(){const tr=prompt.last.traces[1],q=tr.q[0],rows=tr.cache,b=+$('quant-bits').value,part=$('quant-part').value,k=rows.map(r=>r.k[0]),v=rows.map(r=>r.v[0]);
  const khat=part==='v'?k:k.map(x=>quantize(x,b).values),vhat=part==='k'?v:v.map(x=>quantize(x,b).values),scores=khat.map(x=>dot(q,x)/Math.sqrt(8)),a=softmax(scores),full=tr.attention[0];
  const out=Array.from({length:8},(_,d)=>a.reduce((s,w,i)=>s+w*vhat[i][d],0)),original=tr.weightedValues[0],parts=part==='both'?2:1;
  const payload=rows.length*8*parts*b/8+rows.length*8*(2-parts)*4,metadata=rows.length*parts*8;
  const maxscore=Math.max(...scores.map((s,i)=>Math.abs(s-tr.scores[0][i]))),relative=Math.sqrt(out.reduce((s,v,i)=>s+(v-original[i])**2,0))/(Math.sqrt(dot(original,original))||1);
  const order=full.map((_,i)=>i).sort((a,b)=>full[b]-full[a]).slice(0,6);
  multiBars('quant-bars',order.map(i=>`${i}:${rows[i].token}`),[{name:'exact weight',values:order.map(i=>full[i])},{name:'quantized',values:order.map(i=>a[i])}]);
  set('quant-metrics',metric(f(tv(full,a),4),'attention TV')+metric(f(maxscore,4),'max score error')+metric(pct(relative),'relative output L2 error')+metric(`${payload+metadata} B`,'payload + metadata',`payload ${payload} B · metadata ${metadata} B`));
  set('quant-values',`K[0] exact = ${vector(k[0])}\nK[0] approx = ${vector(khat[0])}\nV[0] exact = ${vector(v[0])}\nV[0] approx = ${vector(vhat[0])}\nBaseline: ${rows.length*2*8*4} B FP32 or ${rows.length*2*8*2} B FP16, for this single head / single layer only.`);
}
function renderPaging(){const T=+$('page-tokens').value,B=+$('page-block').value,N=Math.ceil(T/B),physical=[5,1,9,2,11,0,8,3,10,4,7,6];
  set('page-map',Array.from({length:N},(_,j)=>`<div class="page-block"><strong>logical ${j} → physical ${physical[j]}</strong>${Array.from({length:B},(_,k)=>{const t=j*B+k;return `<span class="slot ${t<T?'':'dropped'}" title="physical row ${physical[j]*B+k}">${t<T?t:'free'}</span>`;}).join('')}<span>base address = ${physical[j]*B}</span></div>`).join(''));
  $('page-result').textContent=`${T} live token rows occupy ${N} blocks × ${B} slots = ${N*B} allocated slots. ${N*B-T} tail slots are unused. Token ${T-1} is addressed at physical row ${physical[N-1]*B+(T-1)%B}; attention still reads tokens in logical order.`;
}
function renderFlash(){const tr=prompt.last.traces[1],result=onlineAttention(tr.scores[0],tr.cache.map(r=>r.v[0]),+$('flash-block').value);
  set('flash-table','<thead><tr><th>Tile starts at</th><th>Running max m</th><th>Exp sum ℓ</th><th>Output numerator o[0]</th><th>Normalized o[0]/ℓ</th></tr></thead><tbody>'+result.states.map(r=>`<tr><td>${r.start}</td><td>${f(r.m,5)}</td><td>${f(r.l,5)}</td><td>${f(r.o[0],5)}</td><td>${f(r.o[0]/r.l,5)}</td></tr>`).join('')+'</tbody>');
  set('flash-result',metric(sci(Math.max(...result.output.map((v,i)=>Math.abs(v-tr.weightedValues[0][i])))),'max error versus dense')+metric(`${result.states.length}`,'tiles processed')+metric(`${tr.cache.length}`,'K/V rows still read'));
}
function renderRotation(){const x=[8,.4,-.2,.1],R=[[.5,.5,.5,.5],[.5,-.5,.5,-.5],[.5,.5,-.5,-.5],[.5,-.5,-.5,.5]],y=mv(x,R),b=+$('rotation-bits').value,plain=quantize(x,b).values,rot=mv(quantize(y,b).values,R),q=[.2,1,.5,-.3],qr=mv(q,R);
  const err=a=>Math.sqrt(a.reduce((s,v,i)=>s+(v-x[i])**2,0));
  set('rotation-data',`Original x = ${vector(x)}\nOrthogonal Hadamard-4 rotation xR = ${vector(y)}\nWithout rotation, rounded x = ${vector(plain)}\nRotate → round → inverse rotate = ${vector(rot)}`);
  multiBars('rotation-bars',['L2 error','|dot error|'],[{name:'plain',values:[err(plain),Math.abs(dot(q,plain)-dot(q,x))]},{name:'rotated',values:[err(rot),Math.abs(dot(q,rot)-dot(q,x))]}],Math.max(err(plain),err(rot),1e-6));
  $('rotation-result').textContent=`Before rounding, dot products agree with error ${sci(Math.abs(dot(q,x)-dot(qr,y)))}. At ${b} bits: plain L2 error ${f(err(plain),4)}, rotated L2 error ${f(err(rot),4)}. Both store ${4*b} payload bits plus vector metadata; this toy does not implement TurboQuant’s full algorithm.`;
}
function renderDependents(){renderSpec();renderMTP();eviction();quantLab();renderFlash();}
function audits(){const reference=model.reference.prompt.map(w=>model.vocab.indexOf(w)),serial=run(model,reference),parallel=prefill(model,reference),max=Math.max(...parallel.logits.at(-1).map((v,i)=>Math.abs(v-serial.last.logits[i]))),py=Math.max(...serial.last.logits.map((v,i)=>Math.abs(v-model.reference.last_logits[i])));
  set('prefill-check',metric(sci(max),'browser matrix prefill ↔ cached decode')+metric(sci(py),'browser ↔ CUDA reference logits')+metric(sci(model.checks.prefill_vs_cached_max_logit_error),'CUDA prefill ↔ cached decode'));
  const checks=[['CUDA next-token accuracy',pct(model.checks.next_token_accuracy)],['Held-out prompts',model.checks.heldout_examples],['CUDA prefill/cached max logit error',sci(model.checks.prefill_vs_cached_max_logit_error)],['CUDA batch/serial verification max logit error',sci(model.checks.batch_verification_max_logit_error)],['Rejected cache rollback',model.checks.rollback_exact?'exact, both K and V':'failed'],['Browser/CUDA reference max logit error',sci(py)],['Training examples',model.training.examples],['GPU used',model.training.device]];
  set('audit-data','<div class="table-scroll"><table class="audit-table"><tbody>'+checks.map(([k,v])=>`<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join('')+'</tbody></table></div>');
  if(max>1e-7 || py>5e-4)throw Error('Decoder parity audit failed.');
}
async function init(){try{
  [model,gpu]=await Promise.all(['data/course_model.json','data/gpu_demo.json'].map(async u=>{const r=await fetch(u);if(!r.ok)throw Error(`${u}: HTTP ${r.status}`);return r.json();}));
  $('model-proof').textContent=`TWO LAYERS · 23 TOKENS · ${model.training.examples.toLocaleString()} SYNTHETIC TRAINING EXAMPLES · ${model.training.device.toUpperCase()} · LOCAL BROWSER INFERENCE`;
  bind('run-prompt',buildPrompt,'click');bind('preset',()=>{const v=$('preset').value;$('context-text').value=v==='swap'?'france = tokyo ; japan = paris ; india = delhi ;':'france = paris ; japan = tokyo ; india = delhi ;';$('ask-text').value=v==='reverse'?'japan france':'france japan';buildPrompt();},'change');
  bind('token-step',()=>{selected=+$('token-step').value;renderDecoder();});for(const id of ['decoder-layer','decoder-head'])bind(id,renderDecoder,'change');
  for(const id of ['overflow-policy','overflow-step'])bind(id,renderOverflow);
  for(const id of ['arch','arch-step','arch-layer','arch-bytes'])bind(id,renderArchitectures);
  bind('latent-q',renderLatent);bind('draft-mode',renderSpec);bind('draft-mix',renderSampling);
  for(const id of ['transfer-mode','ridge'])bind(id,renderTransfer);
  for(const id of ['prefix-mode','prefix-block'])bind(id,renderPrefix);
  for(const id of ['evict-policy','evict-budget'])bind(id,eviction);
  for(const id of ['quant-bits','quant-part'])bind(id,quantLab);
  for(const id of ['page-tokens','page-block'])bind(id,renderPaging);
  bind('flash-block',renderFlash);bind('rotation-bits',renderRotation);
  buildPrompt();renderOverflow();renderArchitectures();renderLatent();renderSampling();renderTransfer();renderPrefix();renderPaging();renderRotation();audits();
  const observer=new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting){document.querySelectorAll('.contents a').forEach(a=>a.classList.toggle('active',a.hash===`#${e.target.id}`));}},{rootMargin:'-10% 0px -70% 0px'});document.querySelectorAll('.chapter').forEach(el=>observer.observe(el));
  document.body.dataset.ready='true';window.dispatchEvent(new Event('course-ready'));
}catch(error){$('load-error').hidden=false;$('load-error').textContent=`The labs could not load: ${error.message}. Serve the site over HTTP so the checked-in JSON files can be fetched.`;document.body.dataset.error=error.message;console.error(error);}}
init();
