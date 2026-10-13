// Small, deterministic teaching calculations. No pretrained diffusion/NCM weights.
export const sentences = [
  {words:'a red circle rests on the white table'.split(' '), weight:0.4},
  {words:'a blue square rests on the white table'.split(' '), weight:0.35},
  {words:'a green square rests on the white table'.split(' '), weight:0.25},
];
const sum = a => a.reduce((s,x)=>s+x,0);
export function posterior(visible) {
  const compatible=sentences.filter(s=>visible.every((w,i)=>w===null||w===s.words[i]));
  const mass=sum(compatible.map(s=>s.weight));
  const probabilities=visible.map((_,i)=>{
    const p={};
    for(const s of compatible)p[s.words[i]]=(p[s.words[i]]||0)+s.weight/mass;
    return Object.entries(p).sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0]));
  });
  return {mass, probabilities};
}
export function diffusionTrace(rounds=8) {
  let visible=Array(8).fill(null);
  const trace=[{visible:[...visible],newPositions:[],...posterior(visible)}];
  for(let s=1;s<=rounds;s++) {
    const {probabilities,mass}=posterior(visible);
    if(!mass)break;
    const proposals=visible.flatMap((w,i)=>w===null?[{i,word:probabilities[i][0][0],confidence:probabilities[i][0][1]}]:[])
      .sort((a,b)=>b.confidence-a.confidence||a.i-b.i);
    const quota=Math.ceil(8*s/rounds)-Math.ceil(8*(s-1)/rounds);
    const chosen=proposals.slice(0,quota);
    // Every proposal in this round uses the SAME pre-update visible sequence.
    for(const p of chosen)visible[p.i]=p.word;
    trace.push({visible:[...visible],newPositions:chosen.map(p=>p.i),...posterior(visible)});
  }
  return trace;
}
const noise=[.82,.18,.64,.39,.93,.27,.71,.49];
export function corrupt(t,example=0) {
  return sentences[example].words.map((w,i)=>noise[i]<t?null:w);
}
export const codebooks=[[0,2,4,6,8],[-4,-2,0,2,4],[-1,-.5,0,.5,1],[-1,-.5,0,.5,1]];
export const signals={rising:[1,2,3,2,5,6,7,6],falling:[6,5,6,7,2,1,2,3]};
export const sizes=[1,2,4,8];
export function encodeHierarchy(z) {
  let reconstruction=z.map(()=>0);
  return sizes.map((n,k)=>{
    const residual=z.map((v,i)=>v-reconstruction[i]),block=z.length/n;
    const pooled=Array.from({length:n},(_,j)=>sum(residual.slice(j*block,(j+1)*block))/block);
    const codes=pooled.map(v=>codebooks[k].reduce((best,c,j)=>Math.abs(c-v)<Math.abs(codebooks[k][best]-v)?j:best,0));
    const contribution=z.map((_,i)=>codebooks[k][codes[Math.floor(i/block)]]);
    reconstruction=reconstruction.map((v,i)=>v+contribution[i]);
    return {pooled,codes,contribution,reconstruction:[...reconstruction],mse:sum(z.map((v,i)=>(v-reconstruction[i])**2))/z.length};
  });
}
export function reconstructHierarchy(encoded,level,edit=null) {
  const reconstruction=Array(8).fill(0);
  for(let k=0;k<level;k++)for(let i=0;i<8;i++) {
    const j=Math.floor(i/(8/sizes[k]));
    const code=edit&&edit.level===k&&edit.position===j?edit.code:encoded[k].codes[j];
    reconstruction[i]+=codebooks[k][code];
  }
  return reconstruction;
}
export function generationWork(n,rounds,growth=2) {
  const levels=[1];
  while(levels.at(-1)<n)levels.push(Math.min(n,levels.at(-1)*growth));
  return {
    ar:{calls:n,positions:n,pairs:n*(n+1)/2,widths:Array(n).fill(1)},
    diffusion:{calls:rounds,positions:n*rounds,pairs:n*n*rounds,widths:Array(rounds).fill(n)},
    ncm:{calls:levels.length,positions:sum(levels),pairs:sum(levels.map(x=>x*x)),widths:levels},
  };
}
// WQ=WK=0 gives uniform attention. WV=WK(layer 2)=1; h1=x+Attention(x).
export function dependencyTrace(position=6,delta=4,causal=false) {
  const before=[1,2,3,4,5,6,7,8],after=before.map((v,i)=>v+(i===position?delta:0));
  const layer2=x=>x.map((v,i)=>{const visible=causal?x.slice(0,i+1):x;return v+sum(visible)/visible.length;});
  const oldK=layer2(before),newK=layer2(after);
  return {before,after,oldK,newK,differences:newK.map((v,i)=>v-oldK[i])};
}
