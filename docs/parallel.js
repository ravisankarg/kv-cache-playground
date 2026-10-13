import {sentences,posterior,diffusionTrace,corrupt,codebooks,signals,sizes,encodeHierarchy,reconstructHierarchy,generationWork,dependencyTrace} from './parallel-engine.js';
const $=id=>document.getElementById(id);
const fixed=(x,d=3)=>x.toFixed(d),number=x=>x.toLocaleString('en-US');
const metric=(value,label)=>`<div class="metric"><b>${value}</b><span>${label}</span></div>`;
const table=(headers,rows)=>`<table><thead><tr>${headers.map(h=>`<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${rows.map(row=>`<tr>${row.map(c=>`<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
const tokens=(words,active=[])=>words.map((w,i)=>`<span class="token ${w===null?'masked':''} ${active.includes(i)?'current':''}"><small>position ${i}</small>${w??'MASK'}</span>`).join('');
function renderWork(){
  const n=+$('parallel-length').value,s=+$('parallel-rounds').value,g=+$('parallel-growth').value;
  const work=generationWork(n,s,g),names={ar:'Autoregressive',diffusion:'Masked diffusion',ncm:'NCM hierarchy'};
  $('parallel-schedules').innerHTML=Object.entries(work).map(([key,w])=>{
    // Show at most 16 calls; retain exact totals and expose the full NCM schedule.
    const shown=w.widths.slice(0,16);
    return `<div class="schedule-row"><div><b>${names[key]}</b><small>${w.calls} sequential calls</small></div><div class="schedule-steps" aria-label="${names[key]} positions per call">${shown.map((width,i)=>`<span class="schedule-step ${key}" style="--extent:${20+50*width/n}px" title="Call ${i+1}: ${width} new positions">${width}</span>`).join('')}${w.widths.length>16?`<span class="schedule-more">+${w.widths.length-16} calls</span>`:''}</div></div>`;
  }).join('');
  $('parallel-work').innerHTML=table(['Generator','Calls (NFE)','Positions processed','Attention pairs'],Object.entries(work).map(([key,w])=>[names[key],number(w.calls),number(w.positions),number(w.pairs)]));
  $('parallel-work-result').textContent=`For N = ${n}: the hierarchy visits ${work.ncm.widths.join(' → ')} positions in successive calls (${work.ncm.positions} total). Diffusion visits all ${n} positions on each of ${s} calls (${work.diffusion.positions} total). The hierarchy still needs a final text decoder; these counts do not measure latency.`;
}
function renderCorruption(){
  const t=+$('diff-noise').value,example=+$('diff-example').value,x=corrupt(t,example),p=posterior(x);
  const terms=x.flatMap((w,i)=>w===null?[{i,p:p.probabilities[i].find(([word])=>word===sentences[example].words[i])[1]}]:[]);
  const loss=terms.reduce((s,v)=>s-Math.log(v.p),0);
  $('diff-noise-label').textContent=`t = ${fixed(t,2)} · ${terms.length}/8 masks`;
  $('diff-corrupted').innerHTML=tokens(x);
  $('diff-loss').textContent=t===0?'t = 0: no corruption. The 1/t training expression is not evaluated at this endpoint.':`Masked-token negative log likelihood = ${terms.length?terms.map(v=>`−ln(${fixed(v.p)})`).join(' + '):'0'} = ${fixed(loss)} nats.\nWeighted loss contribution = ${fixed(loss)} / ${fixed(t,2)} = ${fixed(loss/t)}. Only masked positions contribute.`;
}
let diffTrace=[],diffStep=0,ncmLevel=1;
let stopDiff=()=>{},stopNcm=()=>{};
function resetDiff(){stopDiff();diffTrace=diffusionTrace(+$('diff-rounds').value);diffStep=0;$('diff-step').max=diffTrace.length-1;renderDiff();}
function renderDiff(){
  const state=diffTrace[diffStep],count=state.visible.filter(w=>w!==null).length,done=count===8;
  $('diff-step').value=diffStep;$('diff-step-label').textContent=`Round ${diffStep} / ${diffTrace.length-1}`;
  $('diff-tokens').innerHTML=tokens(state.visible,state.newPositions);
  $('diff-metrics').innerHTML=metric(`${count} / 8`,'positions committed')+metric(diffStep,'denoiser calls so far')+metric(8*diffStep,'sequence positions processed');
  $('diff-proposals').innerHTML=table(['Position','Visible token','Distribution for next round'],state.visible.map((w,i)=>[i,w??'MASK',w!==null?'Already committed':state.probabilities[i].map(([word,p])=>`${word}: ${fixed(p*100,1)}%`).join(' · ')]));
  $('diff-result').textContent=done?(state.mass>0?`Supported output: “${state.visible.join(' ')}”. This sentence has ${fixed(state.mass*100,0)}% probability in the original toy language.`:`Unsupported output: “${state.visible.join(' ')}”. “red square” has zero probability in this toy language. Simultaneous marginal choices missed the dependency. Reset with eight rounds to condition on each new token.`):`The next round uses the currently visible tokens to recompute every masked position’s distribution. ${diffStep===0?'Both color and shape are still uncertain.':`Newly committed positions: ${state.newPositions.join(', ')}.`}`;
  $('diff-next').disabled=done;$('diff-play').disabled=done;
  if(done)stopDiff();
}
let hierarchy;
function selectedEdit(){const level=+$('ncm-edit-level').value;return level<0?null:{level,position:+$('ncm-edit-position').value,code:+$('ncm-edit-code').value};}
function editOptions(reset=true){
  const level=+$('ncm-edit-level').value,disabled=level<0;
  $('ncm-edit-position').disabled=disabled;$('ncm-edit-code').disabled=disabled;
  if(disabled){$('ncm-edit-position').innerHTML='<option value="0">1</option>';$('ncm-edit-code').innerHTML='<option value="0">—</option>';return;}
  if(reset)$('ncm-edit-position').innerHTML=Array.from({length:sizes[level]},(_,j)=>`<option value="${j}">${j+1}</option>`).join('');
  const position=+$('ncm-edit-position').value;
  $('ncm-edit-code').innerHTML=codebooks[level].map((v,j)=>`<option value="${j}">c${j} = ${v}</option>`).join('');
  $('ncm-edit-code').value=hierarchy[level].codes[position];
}
function resetNcm(){stopNcm();ncmLevel=1;hierarchy=encodeHierarchy(signals[$('ncm-signal').value]);$('ncm-edit-level').value=-1;editOptions();renderNcm();}
function renderNcm(){
  const z=signals[$('ncm-signal').value],edit=selectedEdit(),r=reconstructHierarchy(hierarchy,ncmLevel,edit),base=reconstructHierarchy(hierarchy,ncmLevel);
  $('ncm-level').value=ncmLevel;$('ncm-level-label').textContent=`${ncmLevel} / 4 levels revealed`;
  $('ncm-hierarchy').innerHTML=hierarchy.map((h,k)=>`<div class="hierarchy-row ${k<ncmLevel?'revealed':'unrevealed'}"><b>Level ${k+1}<small>${sizes[k]} ${sizes[k]===1?'code':'codes'}</small></b><div class="hierarchy-codes" style="grid-template-columns:repeat(${sizes[k]},minmax(0,1fr))">${h.codes.map((code,j)=>{const c=edit?.level===k&&edit.position===j?edit.code:code;return `<span class="hierarchy-code ${c!==code?'edited':''}">${k<ncmLevel?`c${c}<small>${codebooks[k][c]>=0?'+':''}${codebooks[k][c]}</small>`:'<span aria-label="hidden code">?</span>'}</span>`;}).join('')}</div></div>`).join('');
  const all=[...z,...r,0],lo=Math.min(...all)-.5,hi=Math.max(...all)+.5;
  const x=i=>45+i*65,y=v=>155-(v-lo)/(hi-lo)*125,points=v=>v.map((a,i)=>`${x(i)},${y(a)}`).join(' ');
  $('ncm-plot').innerHTML=`<svg viewBox="0 0 550 190" role="img" aria-label="Target and reconstructed eight-position scalar signals; exact values are in the following table"><line x1="35" x2="515" y1="160" y2="160" stroke="#c8d4c8"/><polyline points="${points(z)}" fill="none" stroke="#426cb5" stroke-width="2" stroke-dasharray="5 4"/><polyline points="${points(r)}" fill="none" stroke="#13745d" stroke-width="3"/>${z.map((v,i)=>`<circle cx="${x(i)}" cy="${y(v)}" r="4" fill="#426cb5"/><circle cx="${x(i)}" cy="${y(r[i])}" r="4" fill="#13745d"/><text x="${x(i)}" y="180" text-anchor="middle">${i}</text>`).join('')}</svg>`;
  const mse=z.reduce((s,v,i)=>s+(v-r[i])**2,0)/8;
  $('ncm-metrics').innerHTML=metric(sizes.slice(0,ncmLevel).reduce((s,v)=>s+v,0),'codes revealed')+metric(fixed(mse,4),'reconstruction MSE')+metric(8,'reconstructed positions');
  $('ncm-values').innerHTML=table(['Position',...z.map((_,i)=>i)],[['Target z',...z],['Reconstruction',...r.map(v=>fixed(v,2))],['Remaining residual',...z.map((v,i)=>fixed(v-r[i],2))]]);
  if(ncmLevel===0)$('ncm-arithmetic').textContent='Start: reconstruction = [0, 0, 0, 0, 0, 0, 0, 0]. No codes revealed yet.';
  else {const k=ncmLevel-1,h=hierarchy[k];$('ncm-arithmetic').textContent=`Original encoding at level ${ncmLevel}:\nAverage-pooled residual = [${h.pooled.join(', ')}]\nCodebook = [${codebooks[k].join(', ')}]\nNearest code IDs = [${h.codes.map(v=>'c'+v).join(', ')}]\nUpsampled correction = [${h.contribution.join(', ')}]\nAdd this correction to the preceding reconstruction.${edit?' The chart applies your replacement; this trace retains the original encoding.':''}`;}
  const changed=r.filter((v,i)=>Math.abs(v-base[i])>1e-10).length;
  $('ncm-edit-result').textContent=!edit?'No edit. Select a level to inspect how widely one code contributes.':edit.level>=ncmLevel?`Level ${edit.level+1} is not revealed yet. Reveal it to apply this edit.`:`Replacement changes ${changed} / 8 reconstructed positions. This level’s code spans ${8/sizes[edit.level]} positions under repeated upsampling. Later stored codes are unchanged.`;
  $('ncm-next').disabled=ncmLevel===4;$('ncm-play').disabled=ncmLevel===4;if(ncmLevel===4)stopNcm();
}
function renderDependency(){
  const position=+$('dependency-position').value,delta=+$('dependency-delta').value,causal=$('dependency-mask').value==='causal';
  const t=dependencyTrace(position,delta,causal);
  $('dependency-label').textContent=`position ${position} · Δ = ${delta}`;
  $('dependency-table').innerHTML=table(['Position','Input before → after','Layer-2 K/V before','Layer-2 K/V after','Δ K/V'],t.before.map((v,i)=>[i,`${v} → ${t.after[i]}${i===position?' (edited)':''}`,fixed(t.oldK[i]),fixed(t.newK[i]),`<span class="${Math.abs(t.differences[i])>1e-10?'state-changed':'state-stable'}">${fixed(t.differences[i])}</span>`]));
  const changed=t.differences.filter((d,i)=>i!==position&&Math.abs(d)>1e-10).length;
  $('dependency-result').textContent=`${changed} unchanged-token positions have changed layer-2 K/V. At position 2: input ${t.before[2]} → ${t.after[2]}, K/V ${fixed(t.oldK[2])} → ${fixed(t.newK[2])}. ${causal?'Positions before the edited input cannot see it; their cached rows remain identical.':'Every position attends to the edited input, so a nonzero change propagates to every later-layer row in this toy.'}`;
}
// No autoplay; user-controlled timers pause on tab changes and manual input.
function player(id,advance){let timer=null;const button=$(id);const stop=()=>{clearInterval(timer);timer=null;button.textContent='Play';};button.addEventListener('click',()=>{if(timer){stop();return;}button.textContent='Pause';timer=setInterval(advance,1000);advance();});document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});return stop;}
export function initParallelLabs(){
  for(const id of ['parallel-length','parallel-rounds','parallel-growth'])$(id).addEventListener('input',renderWork);
  for(const id of ['diff-noise','diff-example'])$(id).addEventListener('input',renderCorruption);
  const nextDiff=()=>{diffStep=Math.min(diffStep+1,diffTrace.length-1);renderDiff();};
  stopDiff=player('diff-play',nextDiff);$('diff-next').addEventListener('click',()=>{stopDiff();nextDiff();});$('diff-reset').addEventListener('click',resetDiff);$('diff-rounds').addEventListener('input',resetDiff);$('diff-step').addEventListener('input',()=>{stopDiff();diffStep=+$('diff-step').value;renderDiff();});
  const nextNcm=()=>{ncmLevel=Math.min(ncmLevel+1,4);renderNcm();};
  stopNcm=player('ncm-play',nextNcm);$('ncm-next').addEventListener('click',()=>{stopNcm();nextNcm();});$('ncm-reset').addEventListener('click',resetNcm);$('ncm-signal').addEventListener('input',resetNcm);$('ncm-level').addEventListener('input',()=>{stopNcm();ncmLevel=+$('ncm-level').value;renderNcm();});
  $('ncm-edit-level').addEventListener('input',()=>{stopNcm();editOptions();renderNcm();});$('ncm-edit-position').addEventListener('input',()=>{stopNcm();editOptions(false);renderNcm();});$('ncm-edit-code').addEventListener('input',()=>{stopNcm();renderNcm();});
  for(const id of ['dependency-mask','dependency-position','dependency-delta'])$(id).addEventListener('input',renderDependency);
  renderWork();renderCorruption();resetDiff();resetNcm();renderDependency();
}
