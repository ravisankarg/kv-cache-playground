import {losses,defaults} from './loss-catalog.js?v=2';
import {evaluateLoss} from './loss-engine.js?v=2';
const canonical=id=>id==='calibrated'?'rlcd':id;
const $=id=>document.getElementById(id),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const f=x=>x===null?'undefined':typeof x==='number'?(Math.abs(x)<1e-10?'0.0000':x.toFixed(4)):esc(x);
let selected=losses[0],values=defaults(selected);
function renderNumbers(){
 const result=evaluateLoss(selected.id,values);
 for(const control of selected.controls){const el=$('parameter-'+control.key);el.value=values[control.key];const out=$('parameter-output-'+control.key);if(out)out.textContent=values[control.key];}
 $('loss-value').innerHTML=`<div class="metric"><b>${f(result.value)}</b><span>Selected toy objective</span></div><div class="metric"><b>${selected.controls.length}</b><span>adjustable inputs</span></div>`;
 $('loss-steps').innerHTML=result.steps.map(s=>`<li>${esc(s)}</li>`).join('');
 const third=selected.id==='infonce'?'∂L / ∂candidate score':selected.id==='mse'?'∂L / ∂third prediction':'∂L / ∂logit';
 const columns=result.rows.some(r=>r.length===3)?['Quantity',selected.id==='infonce'?'Probability':'Value',third]:['Quantity','Value'];
 $('loss-table').innerHTML=`<table><thead><tr>${columns.map(x=>`<th scope="col">${esc(x)}</th>`).join('')}</tr></thead><tbody>${result.rows.map(row=>`<tr>${columns.map((_,i)=>`<td>${row[i]===undefined?'—':f(row[i])}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
 renderPlot(result.value);
}
function renderPlot(current){
 const control=selected.controls.find(c=>!c.options),key=control.key;
 const samples=Array.from({length:81},(_,i)=>{const x=control.min+(control.max-control.min)*i/80;return {x,y:evaluateLoss(selected.id,{...values,[key]:x}).value};});
 const finite=samples.filter(p=>p.y!==null&&Number.isFinite(p.y));
 if(!finite.length){$('loss-plot').textContent='This slice is undefined.';return;}
 let lo=Math.min(0,...finite.map(p=>p.y)),hi=Math.max(0,...finite.map(p=>p.y));if(hi-lo<1e-7)hi=lo+1;
 const sx=x=>52+(x-control.min)/(control.max-control.min)*450,sy=y=>205-(y-lo)/(hi-lo)*170;
 let drawing=false;const path=samples.map(p=>{if(p.y===null||!Number.isFinite(p.y)){drawing=false;return '';}const s=`${drawing?'L':'M'}${sx(p.x)},${sy(p.y)}`;drawing=true;return s;}).join(' ');
 const h=1e-4,left=Math.max(control.min,values[key]-h),right=Math.min(control.max,values[key]+h);
 const a=evaluateLoss(selected.id,{...values,[key]:left}).value,b=evaluateLoss(selected.id,{...values,[key]:right}).value;
 const derivative=a===null||b===null?null:(b-a)/(right-left);
 $('loss-plot').innerHTML=`<svg viewBox="0 0 540 265" role="img" aria-label="Loss slice as ${esc(control.label)} varies; selected value ${f(current)}"><line x1="52" y1="205" x2="505" y2="205" stroke="#b4c5b5"/><line x1="52" y1="30" x2="52" y2="205" stroke="#b4c5b5"/><text x="46" y="39" text-anchor="end">${f(hi)}</text><text x="46" y="205" text-anchor="end">${f(lo)}</text><path d="${path}" stroke="#13745d" fill="none" stroke-width="2.5"/>${current!==null&&Number.isFinite(current)?`<circle cx="${sx(values[key])}" cy="${sy(current)}" r="5" fill="#426cb5" stroke="white" stroke-width="2"/>`:''}<text x="52" y="225">${control.min}</text><text x="505" y="225" text-anchor="end">${control.max}</text><text x="278" y="253" text-anchor="middle">${esc(control.label)}</text></svg>`;
 $('loss-plot-note').textContent=`Local numerical slope with respect to “${control.label}”: ${f(derivative)}. This is a one-variable slice, not a training run. At a kink or rank crossing the derivative may not exist; a finite difference there is only an estimate.`;
 document.body.dataset.loss=selected.id;
}
function selectLoss(id,{scroll=false}={}){
 selected=losses.find(s=>s.id===canonical(id))||losses[0];values=defaults(selected);
 $('loss-select').value=selected.id;$('loss-title').textContent=selected.title;$('loss-tag').textContent=selected.tag;
 for(const field of ['principle','derivation','equation','toy','modern'])$('loss-'+field).textContent=selected[field];
 $('loss-failure-text').textContent=selected.failureText;
 $('loss-controls').innerHTML=selected.controls.map(c=>c.options?`<label>${esc(c.label)}<select id="parameter-${c.key}">${c.options.map((text,i)=>`<option value="${i}">${esc(text)}</option>`).join('')}</select></label>`:`<label>${esc(c.label)} <output id="parameter-output-${c.key}" for="parameter-${c.key}">${c.value}</output><input id="parameter-${c.key}" type="range" min="${c.min}" max="${c.max}" step="${c.step}" value="${c.value}"></label>`).join('');
 for(const c of selected.controls)$('parameter-'+c.key).addEventListener('input',()=>{values[c.key]=+$('parameter-'+c.key).value;$('loss-preset-status').textContent='Custom inputs';renderNumbers();});
 const names=['Bi-encoder','Cross-encoder','Decoder'];$('loss-fit').innerHTML=selected.fit.map((text,i)=>`<div><h4>${names[i]}</h4><p>${esc(text)}</p></div>`).join('');
 $('loss-sources').innerHTML=selected.sources.map(([name,url])=>`<a href="${esc(url)}">${esc(name)} ↗</a>`).join('');
 $('loss-permalink').href='#'+selected.id;$('loss-preset-status').textContent='Default worked example';
 $('loss-count').textContent=`${losses.indexOf(selected)+1} / ${losses.length}`;
 for(const a of document.querySelectorAll('#loss-navigation a')){const active=a.hash==='#'+selected.id;a.classList.toggle('active',active);if(active)a.setAttribute('aria-current','true');else a.removeAttribute('aria-current');}
 $('loss-previous').disabled=losses.indexOf(selected)===0;$('loss-next').disabled=losses.indexOf(selected)===losses.length-1;
 renderNumbers();if(scroll)$('workbench').scrollIntoView({behavior:'instant',block:'start'});
}
function route(){const id=canonical(location.hash.slice(1));if(losses.some(s=>s.id===id))selectLoss(id,{scroll:true});}
try{
 const groups=[...new Set(losses.map(s=>s.group))];
 $('loss-navigation').innerHTML=groups.map(group=>`<div class="loss-nav-group"><span>${esc(group)}</span>${losses.filter(s=>s.group===group).map(s=>`<a href="#${s.id}">${esc(s.title)}</a>`).join('')}</div>`).join('');
 $('loss-select').innerHTML=groups.map(group=>`<optgroup label="${esc(group)}">${losses.filter(s=>s.group===group).map(s=>`<option value="${s.id}">${esc(s.title)}</option>`).join('')}</optgroup>`).join('');
 $('loss-select').addEventListener('change',()=>{if(location.hash==='#'+$('loss-select').value)selectLoss($('loss-select').value);else location.hash=$('loss-select').value;});
 $('loss-reset').addEventListener('click',()=>{values=defaults(selected);$('loss-preset-status').textContent='Default worked example';renderNumbers();});
 $('loss-failure').addEventListener('click',()=>{values={...defaults(selected),...selected.failure};$('loss-preset-status').textContent='Failure case loaded — read “When it does not work” below.';renderNumbers();});
 for(const [id,delta] of [['loss-previous',-1],['loss-next',1]])$(id).addEventListener('click',()=>{const next=losses[losses.indexOf(selected)+delta];if(next)location.hash=next.id;});
 window.addEventListener('hashchange',route);selectLoss(location.hash.slice(1));
 if(location.hash)requestAnimationFrame(()=>{if(losses.some(s=>s.id===canonical(location.hash.slice(1))))$('workbench').scrollIntoView({behavior:'instant',block:'start'});else document.getElementById(location.hash.slice(1))?.scrollIntoView({behavior:'instant'});});
 document.body.dataset.ready='true';window.dispatchEvent(new Event('losses-ready'));
}catch(error){$('loss-error').hidden=false;$('loss-error').textContent=`The loss explorer could not load: ${error.message}`;document.body.dataset.error=error.message;console.error(error);}
