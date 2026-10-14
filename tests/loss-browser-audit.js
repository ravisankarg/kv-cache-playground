export async function auditLosses(w){
 const d=w.document,assertions=[];
 const assert=(condition,label)=>{if(!condition)throw Error(label);assertions.push(label);};
 const text=id=>d.getElementById(id).textContent;
 const change=(id,value)=>{const el=d.getElementById(id);el.value=value;el.dispatchEvent(new w.Event('input',{bubbles:true}));};
 const choose=async id=>{w.location.hash=id;await new Promise(resolve=>w.setTimeout(resolve,20));assert(d.body.dataset.loss===id,'route '+id);};
 const healthy=label=>{
  assert(!/NaN|Infinity/.test(text('loss-value')+text('loss-table')+text('loss-steps')),label+' finite arithmetic');
  assert(d.documentElement.scrollWidth<=d.documentElement.clientWidth+1,label+' fits viewport');
 };
 assert(!d.body.dataset.error,'loss page initialized');
 const options=[...d.querySelectorAll('#loss-select option')].map(o=>o.value);
 assert(options.length===22,'22 loss families available');
 for(const id of options){
  await choose(id);healthy(id+' default');
  assert(d.querySelectorAll('#loss-fit>div').length===3,id+' three application notes');
  assert(d.querySelectorAll('#loss-sources a').length>0,id+' source links');
  assert(d.querySelector('#loss-plot svg'),id+' numerical curve');
  assert(text('loss-equation').trim()&&text('loss-failure-text').trim(),id+' equation and failure explanation');
  d.getElementById('loss-failure').click();healthy(id+' failure');
  assert(text('loss-preset-status').includes('Failure case loaded'),id+' failure preset runs');
  d.getElementById('loss-reset').click();
  for(const input of [...d.querySelectorAll('#loss-controls input, #loss-controls select')]){
   const previous=input.value;
   for(const value of input.tagName==='SELECT'?[...input.options].map(o=>o.value):[input.min,input.max]){
    change(input.id,value);healthy(id+' '+input.id+'='+value);
   }
   change(input.id,previous);
  }
 }
 await choose('dist');change('parameter-scale','0');assert(text('loss-value').includes('undefined'),'DIST reports zero-variance undefined correlation');
 await choose('margin-mse');change('parameter-positive','3');change('parameter-negative','1');change('parameter-shift','10');assert(text('loss-value').includes('0.0000')&&text('loss-table').includes('100.0000'),'margin loss ignores common offset but absolute MSE sees it');
 await choose('infonce');const before=text('loss-value');change('parameter-policy','1');assert(text('loss-value')!==before,'negative filtering changes objective');
 await choose('listwise');change('parameter-objective','1');const listMLE=text('loss-value');change('parameter-objective','2');assert(text('loss-value')!==listMLE,'position weighting changes list loss');
 await choose('rlcd');assert(text('loss-value').includes('0.2100'),'proper-score calibrated reference');d.getElementById('loss-failure').click();assert(text('loss-value').includes('0.2941'),'overconfidence increases proper loss');
 assert(text('loss-title').includes('Reinforcement Learning for Calibrated Decisions'),'RLCD uses the requested calibrated-decisions meaning');
 w.location.hash='calibrated';await new Promise(r=>w.setTimeout(r,20));assert(d.body.dataset.loss==='rlcd','original calibration link remains valid');
 await choose('mse');assert(d.getElementById('loss-previous').disabled,'first-family previous disabled');d.getElementById('loss-next').click();await new Promise(r=>w.setTimeout(r,20));assert(d.body.dataset.loss==='geometry','next-family navigation');
 const select=d.getElementById('loss-select');select.value='sdpo';select.dispatchEvent(new w.Event('change',{bubbles:true}));await new Promise(r=>w.setTimeout(r,20));assert(d.body.dataset.loss==='sdpo'&&d.getElementById('loss-next').disabled,'selector and last-family navigation');
 assert(d.querySelectorAll('#selection-guide tbody tr').length===9,'fine-tuning and distillation selection guide');
 return {count:assertions.length,assertions};
}
