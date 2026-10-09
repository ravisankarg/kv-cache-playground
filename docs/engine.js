// The same row-vector, pre-RMSNorm, RoPE decoder as build_course.py.
// Pure numerical code is shared by the browser labs and Node audits.
export const dot = (a,b) => a.reduce((s,x,i)=>s+x*b[i],0);
export const softmax = x => {const m=Math.max(...x), e=x.map(v=>Math.exp(v-m)),s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s);};
export const mv = (x,w) => w[0].map((_,j)=>x.reduce((s,v,i)=>s+v*w[i][j],0));
export const rms = x => {const r=1/Math.sqrt(dot(x,x)/x.length+1e-6);return x.map(v=>v*r);};
export const add = (a,b) => a.map((x,i)=>x+b[i]);
export const rotate = (x,p) => {const y=[];for(let i=0;i<x.length;i+=2){const a=p*10000**(-i/x.length),c=Math.cos(a),s=Math.sin(a);y.push(x[i]*c-x[i+1]*s,x[i]*s+x[i+1]*c);}return y;};
export const tv = (a,b) => a.reduce((s,v,i)=>s+Math.abs(v-b[i]),0)/2;
export const cloneCache = c => c.map(layer=>layer.map(row=>({position:row.position,k:row.k.map(v=>[...v]),v:row.v.map(v=>[...v]),token:row.token})));

export function tokenize(text, model){
  const words=text.toLowerCase().match(/<[^>]+>|[a-z]+|[:=;?]|\S/g)||[];
  return {words,ids:words.map(w=>Math.max(0,model.vocab.indexOf(w)) || (w==='<pad>'?0:3)),unknown:words.filter(w=>!model.vocab.includes(w))};
}
export function emptyCache(model){return Array.from({length:model.config.layers},()=>[]);}
export function step(model,id,caches,position,{policy='strict',limit=model.config.max_context,keep=null}={}){
  if(policy==='strict' && position>=limit)throw Error(`Position ${position} exceeds the ${limit}-token model/runtime cap.`);
  const {heads:H,head_dim:HD}=model.config;
  let x=add(model.weights.embedding[id],model.weights.position[position]),traces=[];
  for(let l=0;l<model.config.layers;l++){
    const b=model.weights.blocks[l],z=rms(x),q0=mv(z,b.q),k0=mv(z,b.k),v0=mv(z,b.v);
    const split=a=>Array.from({length:H},(_,h)=>a.slice(h*HD,(h+1)*HD));
    const q=split(q0).map(v=>rotate(v,position)),k=split(k0).map(v=>rotate(v,position)),v=split(v0);
    caches[l].push({position,k,v,token:model.vocab[id]});
    if(keep && position===keep.at){caches[l]=caches[l].filter(row=>keep.positions.includes(row.position)||row.position===position);}
    if(policy==='sliding')caches[l]=caches[l].slice(-limit);
    if(policy==='sink' && caches[l].length>limit)caches[l]=[caches[l][0],...caches[l].slice(-(limit-1))];
    const scores=q.map((qh,h)=>caches[l].map(row=>dot(qh,row.k[h])/Math.sqrt(HD)));
    const a=scores.map(softmax);
    const heads=a.map((ah,h)=>Array.from({length:HD},(_,j)=>ah.reduce((s,w,i)=>s+w*caches[l][i].v[h][j],0)));
    const o=mv(heads.flat(),b.o),r=add(x,o),n=rms(r),g=mv(n,b.gate),up=mv(n,b.up);
    const mlp=mv(g.map((v,i)=>v/(1+Math.exp(-v))*up[i]),b.down),y=add(r,mlp);
    traces.push({input:x,q,k,v,scores,attention:a,weightedValues:heads,attentionOutput:o,mlp,output:y,cache:cloneCache([caches[l]])[0]});
    x=y;
  }
  const logits=mv(rms(x),model.weights.head),mtpLogits=mv(rms(x),model.weights.mtp);
  return {id,position,token:model.vocab[id],traces,hidden:x,logits,p:softmax(logits),mtp:softmax(mtpLogits),caches};
}
export function run(model,ids,options={}){
  const caches=emptyCache(model),steps=[];
  for(let i=0;i<ids.length;i++)steps.push(step(model,ids[i],caches,i,options));
  return {steps,caches,last:steps.at(-1)};
}
export function argmax(a){return a.indexOf(Math.max(...a));}
export function generate(model,ids,count=3,options={}){
  const result=run(model,ids,options),generated=[];
  let last=result.last;
  for(let i=0;i<count;i++){
    const id=argmax(last.p);generated.push({id,token:model.vocab[id],p:last.p[id],distribution:last.p});
    // Prediction itself does not write a cache row. Feeding the prediction does.
    if(id===2 || ids.length+i>=model.config.max_context)break;
    last=step(model,id,result.caches,ids.length+i,options);result.steps.push(last);
  }
  return {...result,generated};
}

// Independent matrix prefill. All prompt positions in one layer are evaluated
// before the next layer, with an explicit causal mask, rather than run(step).
export function prefill(model,ids){
  const {heads:H,head_dim:HD}=model.config;
  let x=ids.map((id,t)=>add(model.weights.embedding[id],model.weights.position[t])),caches=[];
  for(const b of model.weights.blocks){
    const z=x.map(rms),split=a=>Array.from({length:H},(_,h)=>a.slice(h*HD,(h+1)*HD));
    const q=z.map((v,t)=>split(mv(v,b.q)).map(h=>rotate(h,t)));
    const k=z.map((v,t)=>split(mv(v,b.k)).map(h=>rotate(h,t))),v=z.map(a=>split(mv(a,b.v)));
    caches.push(ids.map((id,t)=>({position:t,k:k[t],v:v[t],token:model.vocab[id]})));
    x=x.map((row,t)=>{
      const heads=q[t].map((qh,h)=>{
        const weights=softmax(k.slice(0,t+1).map(kk=>dot(qh,kk[h])/Math.sqrt(HD)));
        return Array.from({length:HD},(_,d)=>weights.reduce((s,w,j)=>s+w*v[j][h][d],0));
      });
      const r=add(row,mv(heads.flat(),b.o)),n=rms(r),gate=mv(n,b.gate),up=mv(n,b.up);
      return add(r,mv(gate.map((a,i)=>a/(1+Math.exp(-a))*up[i]),b.down));
    });
  }
  const logits=x.map(v=>mv(rms(v),model.weights.head));
  return {logits,caches,hidden:x};
}

// Exact single-position speculative-sampling identity: accepted mass plus
// rejection probability times the normalized positive residual equals p.
export function speculativeMass(p,q){
  const accepted=p.map((v,i)=>Math.min(v,q[i])),alpha=accepted.reduce((a,b)=>a+b,0);
  const residual=p.map((v,i)=>Math.max(v-q[i],0)),r=residual.reduce((a,b)=>a+b,0);
  const corrected=residual.map(v=>r>1e-12?v/r:0);
  return {accepted,alpha,residual:corrected,output:accepted.map((v,i)=>v+(1-alpha)*corrected[i])};
}
export function quantize(a,bits){
  const lo=Math.min(...a),hi=Math.max(...a),levels=2**bits-1,scale=(hi-lo)/levels||1;
  const integers=a.map(v=>Math.max(0,Math.min(levels,Math.round((v-lo)/scale))));
  return {lo,scale,integers,values:integers.map(v=>lo+scale*v)};
}
export function onlineAttention(scores,values,blockSize){
  let m=-Infinity,l=0,o=values[0].map(()=>0),states=[];
  for(let start=0;start<scores.length;start+=blockSize){
    const ss=scores.slice(start,start+blockSize),vs=values.slice(start,start+blockSize),nm=Math.max(m,...ss);
    const rescale=Math.exp(m-nm),w=ss.map(s=>Math.exp(s-nm));
    l=rescale*l+w.reduce((a,b)=>a+b,0);
    o=o.map((v,d)=>rescale*v+w.reduce((s,a,i)=>s+a*vs[i][d],0));m=nm;
    states.push({start,m,l,o:[...o]});
  }
  return {output:o.map(v=>v/l),states};
}
