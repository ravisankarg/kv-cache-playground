const COLORS = {mha:"#536fa7",gqa:"#37a88f",mqa:"#e1aa4d",mla:"#8c72c9"};
const LABELS = {mha:"Multi-head attention",gqa:"Grouped-query attention",mqa:"Multi-query attention",mla:"Multi-head latent attention"};
const DESCRIPTIONS = {
  mha:"One K head and one V head for every query head.",
  gqa:"Four query heads share two stored K/V heads (two queries per group).",
  mqa:"All four query heads read one shared K/V head.",
  mla:"A shared 8-value latent and 2-value RoPE key stand in for expanded K/V."
};
const $ = (id) => document.getElementById(id);
let data, variant="mha", policy="strict", step=0, layer=0, bytes=2, timer=null;

function esc(value){return String(value).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));}
function activeRun(){return data.variants[variant][policy];}
function activeStep(){return activeRun().steps[Math.min(step,activeRun().steps.length-1)];}
function tokenAt(pos){return data.token_sequence[pos] ?? `t${pos}`;}
function cacheFor(v,pol,st,ly=0){return data.variants[v][pol].steps[st]?.layers[ly]?.cache;}
function fmtShape(shape){return `[${shape.join(", ")}]`;}
function bytesAt(v){const run=data.variants[v].strict;return bytes===2?run.fp16_bytes_per_token:run.fp32_bytes_per_token;}
function shortNum(n){if(n>=1000000)return `${(n/1000000).toFixed(1)} MB`;if(n>=1000)return `${(n/1000).toFixed(1)} KB`;return `${n} B`;}

function updateAll(){
  if(!data)return;
  updateControls(); updateTrace(); updateComparison(); updateEdge();
}

function updateControls(){
  const run=activeRun(), max=run.steps.length-1;
  step=Math.min(step,max);
  $("step-range").max=String(max); $("step-range").value=String(step);
  $("step-label").textContent=`${String(step+1).padStart(2,"0")} / ${String(max+1).padStart(2,"0")}`;
  $("context-size").textContent=data.max_context;
  document.querySelectorAll(".arch-option").forEach(b=>b.classList.toggle("selected",b.dataset.variant===variant));
  document.querySelectorAll(".policy-option").forEach(b=>b.classList.toggle("selected",b.dataset.policy===policy));
  $("policy-note").textContent=policy==="strict"
    ? "At the model/runtime limit, the next token cannot be processed. The full cache stays resident."
    : `Keep the newest ${data.max_context} positions. When a new token arrives, evict the oldest row first.`;
}

function updateTrace(){
  const run=activeRun(), st=activeStep(), cfg=run.config, cache=st.layers[layer].cache;
  $("trace-status").textContent=`GPU trace · ${variant.toUpperCase()} · ${policy==="strict"?"strict limit":"sliding window"}`;
  $("trace-config").textContent=`d_model ${cfg.d_model} · ${cfg.heads} heads · ${cfg.layers} layers`;
  $("current-token").textContent=st.token;
  const atLimit=policy==="strict"&&step===data.max_context-1;
  const next=data.token_sequence[step+1];
  $("next-token").textContent=atLimit?(next?`${next} · blocked`:"limit reached"):(next??"sequence end");
  $("sequence-line").innerHTML=data.token_sequence.map((t,i)=>`<span class="seq-token ${i<step?"done":""} ${i===step?"current":""} ${i>step?"future":""}">${esc(t)}</span>`).join("");
  $("layer-picker").innerHTML=Array.from({length:cfg.layers},(_,i)=>`<option value="${i}" ${i===layer?"selected":""}>${String(i+1).padStart(2,"0")}</option>`).join("");
  $("layer-number").textContent=String(layer+1).padStart(2,"0");
  $("layer-stack").innerHTML=st.layers.map((item,i)=>{
    const count=item.cache.positions.length;
    const dots=Array.from({length:Math.min(count,8)},(_,j)=>`<i title="position ${j}"></i>`).join("");
    return `<div class="layer-node ${i===layer?"current-layer":""}"><b>L${String(i+1).padStart(2,"0")}</b><span class="node-caption">RMSNorm → attention → MLP</span><span class="node-cache">${dots}</span></div>`;
  }).join("");
  const positions=cache.positions;
  if(step===0){$("step-explain").textContent=`${variant.toUpperCase()}: first token creates the first cache row in each layer. Its query has only its own key/value available.`;}
  else if(atLimit){$("step-explain").textContent=`Position ${step} is the final allowed slot. Its query reads all ${data.max_context} rows; another strict decode step would exceed the cap.`;}
  else if(policy==="sliding"&&step>=data.max_context){$("step-explain").textContent=`Position ${step} arrives: evict position ${step-data.max_context}, keep [${positions.join(", ")}], then attend over the retained window.`;}
  else{$("step-explain").textContent=`Position ${step} creates a fresh K and V. The query compares against ${positions.length} visible position${positions.length===1?"":"s"}; the cache now has ${positions.length} row${positions.length===1?"":"s"} per layer.`;}
  const q=st.layers[layer].q_by_head[0]??[];
  $("query-preview").textContent=`[${q.map(n=>Number(n).toFixed(2)).join(", ")}]`;
  const k=cache.k?.[cache.k.length-1]??[];
  $("key-preview").textContent=`[${k.map(n=>Number(n).toFixed(2)).join(", ")}]`;
  const w=st.layers[layer].weights_by_head[0]??[];
  $("weight-preview").textContent=`[${w.map(n=>Number(n).toFixed(2)).join(", ")}]`;
  $("output-preview").textContent=`[${st.layers[layer].attention_output.map(n=>Number(n).toFixed(2)).join(", ")}]`;
  drawAttention(st.layers[layer],cache);
  drawCache(cache);
}

function drawAttention(trace,cache){
  const rows=trace.weights_by_head, positions=cache.positions;
  const max=Math.max(...rows.flat(),1e-6);
  $("attention-map").style.gridTemplateColumns=`repeat(${Math.max(positions.length,1)},minmax(8px,1fr))`;
  $("attention-map").innerHTML=rows.map((row,h)=>row.map((value,j)=>{
    const alpha=.12+.88*value/max;
    return `<i class="attention-cell ${positions[j]===step?"hot":""}" style="background:rgba(53,169,149,${alpha.toFixed(3)})" title="head ${h+1}, position ${positions[j]} (${tokenAt(positions[j])}): ${value.toFixed(4)}"></i>`;
  }).join("")).join("");
  $("axis-labels").innerHTML=positions.map(pos=>`<span title="position ${pos}: ${esc(tokenAt(pos))}">${esc(tokenAt(pos))}</span>`).join("");
}

function colorCell(n,isValue){
  const t=Math.min(Math.abs(n)*.6,.82);
  if(isValue)return n>=0?`rgba(229,177,75,${.18+t})`:`rgba(219,119,105,${.2+t})`;
  return n>=0?`rgba(56,174,153,${.18+t})`:`rgba(92,126,213,${.2+t})`;
}
function matrixHTML(rows,isValue){
  if(!rows?.length)return `<div class="mono">—</div>`;
  return rows.map(row=>`<div class="matrix-row">${row.map(n=>`<i class="matrix-cell ${isValue?"v-cell":""} ${n<0?"negative":""}" style="background:${colorCell(n,isValue)}" title="${Number(n).toFixed(5)}"></i>`).join("")}</div>`).join("");
}
function drawCache(cache){
  const shapes=Object.entries(cache.shapes).map(([name,shape])=>`${name} ${fmtShape(shape)}`).join(" · ");
  $("cache-title").textContent=variant==="mla"?"COMPRESSED CACHE ROWS":"CACHE ROWS · APPEND-ONLY";
  $("cache-shapes").textContent=shapes;
  $("matrix-k-label").textContent=cache.label_k;
  $("matrix-v-label").textContent=cache.label_v;
  $("key-matrix").innerHTML=matrixHTML(cache.k,false);
  $("value-matrix").innerHTML=matrixHTML(cache.v,true);
}

function glyphMarkup(v){
  const q=Array.from({length:4},()=>`<i class="query" title="query head"></i>`).join("");
  if(v==="mla")return `${q}<i class="latent"></i><span class="glyph-caption">cKV + kRoPE</span>`;
  const count={mha:4,gqa:2,mqa:1}[v];
  const kv=Array.from({length:count},()=>`<i title="stored K/V head"></i>`).join("");
  return `${q}<span class="glyph-caption">→</span>${kv}<span class="glyph-caption">${count} KV</span>`;
}
function updateComparison(){
  const names=["mha","gqa","mqa","mla"], maxBytes=Math.max(...names.map(bytesAt));
  $("compare-cards").innerHTML=names.map(v=>{
    const amount=bytesAt(v), ratio=Math.round(100*(1-amount/maxBytes));
    const shape=v==="mla"?"latent + RoPE":"KV heads";
    return `<article class="compare-card ${v===variant?"active":""}" data-variant="${v}" tabindex="0"><div class="compare-card-head"><b>${v.toUpperCase()}</b><span>${shape}</span></div><p>${DESCRIPTIONS[v]}</p><div class="head-glyphs">${glyphMarkup(v)}</div><div class="card-metric"><b>${shortNum(amount)}</b><span>/ token · all layers</span></div><div class="card-saving">${ratio===0?"reference footprint":`${ratio}% fewer bytes than MHA`}</div></article>`;
  }).join("");
  document.querySelectorAll(".compare-card").forEach(card=>{
    card.addEventListener("click",()=>selectVariant(card.dataset.variant));
    card.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();selectVariant(card.dataset.variant);}});
  });
  drawChart(names);
}
function drawChart(names){
  const w=640,h=190,left=39,right=12,top=11,bottom=27,plotW=w-left-right,plotH=h-top-bottom;
  const cap=data.max_context,max=Math.max(...names.map(v=>bytesAt(v)*cap));
  const points=(v)=>Array.from({length:cap+1},(_,i)=>`${(left+plotW*i/cap).toFixed(1)},${(top+plotH-plotH*(bytesAt(v)*i/max)).toFixed(1)}`).join(" ");
  const grids=Array.from({length:4},(_,i)=>{const y=top+plotH*i/3;return `<line class="chart-gridline" x1="${left}" y1="${y}" x2="${w-right}" y2="${y}"/><text class="chart-axis" x="${left-7}" y="${y+3}" text-anchor="end">${shortNum(max*(1-i/3))}</text>`;}).join("");
  const xlabels=[0,Math.round(cap/2),cap].map(n=>`<text class="chart-axis" x="${left+plotW*n/cap}" y="${h-7}" text-anchor="middle">${n} tokens</text>`).join("");
  const lines=names.map(v=>`<polyline class="chart-line" points="${points(v)}" stroke="${COLORS[v]}"/>`).join("");
  const legend=names.map(v=>`<span class="legend-item"><i style="background:${COLORS[v]}"></i>${v.toUpperCase()} · ${shortNum(bytesAt(v))}/token</span>`).join("");
  $("memory-chart").innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Projected ${$("precision-picker").selectedOptions[0].textContent} KV cache, up to ${cap} tokens">${grids}${xlabels}${lines}</svg>`;
  $("chart-legend").innerHTML=legend;
}

function snapshotMarkup(cache,index){
  const rows=cache.k??[];
  const row=rows[index]??[];
  const vrow=cache.v?.[index]??[];
  const max=8;
  const bars=[...row.slice(0,4),...vrow.slice(0,4)];
  return `<div class="snapshot-row">${bars.slice(0,max).map((n,i)=>`<span class="${i>=4?"dim":""}" style="opacity:${(.3+Math.min(Math.abs(n)*.4,.65)).toFixed(2)}"></span>`).join("")}<em>K · V</em></div>`;
}
function updateEdge(){
  const cap=data.max_context, first=cacheFor(variant,"strict",0,0), finalStep=Math.min(cap-1,data.variants[variant].strict.steps.length-1), last=cacheFor(variant,"strict",finalStep,0);
  if(!first||!last)return;
  const variantTitle=variant.toUpperCase();
  $("edge-context-label").textContent=`${Math.min(cap,last.positions.length)} / ${cap} POSITIONS · ${variantTitle}`;
  $("first-token-label").textContent=tokenAt(0);
  $("final-token-label").textContent=tokenAt(finalStep);
  $("final-position").textContent=`position ${finalStep} · cache at ${finalStep+1===cap?"capacity":"end of trace"}`;
  $("first-snapshot").innerHTML=snapshotMarkup(first,0);
  $("final-snapshot").innerHTML=snapshotMarkup(last,last.k.length-1);
  $("first-shape").textContent=Object.entries(first.shapes).map(([k,v])=>`${k}: ${fmtShape(v)}`).join(" · ");
  $("final-shape").textContent=Object.entries(last.shapes).map(([k,v])=>`${k}: ${fmtShape(v)}`).join(" · ");
  $("timeline-count").textContent=`${last.positions.length} / ${cap}`;
  $("timeline-track").innerHTML=Array.from({length:cap},(_,i)=>`<i class="${i<last.positions.length?"filled":""} ${i===last.positions.length-1?"last":""}" title="${i<last.positions.length?`position ${last.positions[i]}`:"empty slot"}"></i>`).join("");
  $("edge-state-copy").textContent=`${variantTitle}: the final in-range token writes the final row. Its query can read all ${last.positions.length} cached position${last.positions.length===1?"":"s"}. The strict trace then stops before processing another token.`;
  if(variant==="mla"){
    $("first-snapshot").setAttribute("aria-label","First token latent and RoPE cache values");
  }
}

function selectVariant(v){variant=v;layer=0;updateAll();}
function stopPlay(){if(timer){clearInterval(timer);timer=null;}$("play-button").querySelector("span:nth-child(2)").textContent="Play token updates";$("play-button").querySelector(".play-icon").textContent="▶";}
function play(){
  if(timer){stopPlay();return;}
  const button=$("play-button");button.querySelector("span:nth-child(2)").textContent="Pause updates";button.querySelector(".play-icon").textContent="Ⅱ";
  timer=setInterval(()=>{const end=activeRun().steps.length-1;if(step>=end){stopPlay();return;}step++;updateAll();},950);
}
function bind(){
  document.querySelectorAll(".arch-option").forEach(b=>b.addEventListener("click",()=>selectVariant(b.dataset.variant)));
  document.querySelectorAll(".policy-option").forEach(b=>b.addEventListener("click",()=>{policy=b.dataset.policy;step=0;stopPlay();updateAll();}));
  $("step-range").addEventListener("input",e=>{step=Number(e.target.value);updateAll();});
  $("layer-picker").addEventListener("change",e=>{layer=Number(e.target.value);updateAll();});
  $("precision-picker").addEventListener("change",e=>{bytes=Number(e.target.value);updateAll();});
  $("play-button").addEventListener("click",play);
  $("copy-command").addEventListener("click",async()=>{try{await navigator.clipboard.writeText("python kv_cache_demo.py --variant all");$("copy-command").textContent="✓";setTimeout(()=>$("copy-command").textContent="⧉",1200);}catch{}});
  document.addEventListener("keydown",e=>{if(e.code==="Space"&&!/INPUT|SELECT|TEXTAREA|BUTTON/.test(document.activeElement.tagName)){e.preventDefault();play();}});
}

async function init(){
  bind();
  try{
    const response=await fetch("data/gpu_demo.json");
    if(!response.ok)throw new Error(`trace unavailable (${response.status})`);
    data=await response.json();
    $("hero-gpu").textContent=data.device;
    $("trace-status").textContent=`GPU trace · ${data.device}`;
    $("gpu-proof-title").textContent=`Computed on ${data.device}`;
    $("gpu-proof-detail").textContent=`PyTorch ${data.torch} · CUDA ${data.cuda} · ${data.config.layers} layers · ${data.max_context}-token cap`;
    updateAll();
  }catch(error){
    $("hero-gpu").textContent="GPU trace not built yet";
    $("trace-status").textContent="Run kv_cache_demo.py to generate the trace";
    $("gpu-proof-title").textContent="GPU trace not found";
    $("gpu-proof-detail").textContent="Run python kv_cache_demo.py --variant all, then reload this page.";
    console.error(error);
  }
}
init();
