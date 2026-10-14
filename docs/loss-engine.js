// Deterministic objective arithmetic; no neural training or benchmark claims.
export const sum=a=>a.reduce((s,x)=>s+x,0);
export const mean=a=>sum(a)/a.length;
export const softplus=x=>Math.max(0,x)+Math.log1p(Math.exp(-Math.abs(x)));
export const sigmoid=x=>1/(1+Math.exp(-x));
export function logsumexp(x){const m=Math.max(...x);return m+Math.log(sum(x.map(v=>Math.exp(v-m))));}
export function softmax(x,t=1){const z=x.map(v=>v/t),l=logsumexp(z);return z.map(v=>Math.exp(v-l));}
export const mse=(a,b)=>mean(a.map((v,i)=>(v-b[i])**2));
export function cosine(a,b){const norm=Math.hypot(...a)*Math.hypot(...b);return norm===0?null:sum(a.map((x,i)=>x*b[i]))/norm;}
export function pearson(a,b){const ma=mean(a),mb=mean(b);return cosine(a.map(v=>v-ma),b.map(v=>v-mb));}
export function kl(p,q){return sum(p.map((v,i)=>v===0?0:q[i]===0?Infinity:v*Math.log(v/q[i])));}
export const ce=(target,logits)=>logsumexp(logits)-sum(target.map((v,i)=>v*logits[i]));
export function infoNCE(scores,t=1,positives=[0],keep=scores.map((_,i)=>i)){
  const z=scores.map(v=>v/t),logDen=logsumexp(keep.map(i=>z[i]));
  const p=scores.map((_,i)=>keep.includes(i)?Math.exp(z[i]-logDen):0);
  return {loss:mean(positives.map(i=>logDen-z[i])),p,gradient:p.map((v,i)=>(v-(positives.includes(i)?1/positives.length:0))/t)};
}
export function listwise(scores,grades){
  const n=scores.length,order=grades.map((_,i)=>i).sort((a,b)=>grades[b]-grades[a]||a-b);
  const rank=scores.map((_,i)=>i).sort((a,b)=>scores[b]-scores[a]||a-b),pos=Array(n);rank.forEach((j,i)=>pos[j]=i);
  const gain=grades.map(v=>2**v-1),discount=i=>1/Math.log2(i+2);
  const idcg=sum(order.map((j,i)=>gain[j]*discount(i))),dcg=sum(rank.map((j,i)=>gain[j]*discount(i)));
  const terms=order.map((j,i)=>logsumexp(order.slice(i).map(k=>scores[k]))-scores[j]);
  const pairs=[];
  for(let i=0;i<n;i++)for(let j=0;j<n;j++)if(grades[i]>grades[j]){
    const weight=idcg===0?0:Math.abs((gain[i]-gain[j])*(discount(pos[i])-discount(pos[j])))/idcg;
    pairs.push({i,j,weight,term:weight*softplus(scores[j]-scores[i])});
  }
  return {listnet:ce(softmax(grades),scores),listmle:sum(terms),plistmle:sum(terms.map((v,i)=>v*(2**(n-i)-1))),lambda:sum(pairs.map(p=>p.term)),ndcg:idcg?dcg/idcg:null,terms,pairs,order,rank};
}
export function distLoss(teacher,student){
  const rows=teacher.map((r,i)=>pearson(r,student[i]));
  const cols=teacher[0].map((_,j)=>pearson(teacher.map(r=>r[j]),student.map(r=>r[j])));
  return {rows,cols,loss:[...rows,...cols].includes(null)?null:mean(rows.map(v=>1-v))+mean(cols.map(v=>1-v))};
}
export function clipped(ratio,advantage,epsilon=.2){return Math.min(ratio*advantage,Math.max(1-epsilon,Math.min(1+epsilon,ratio))*advantage);}
export function groupAdvantages(rewards,scale=true){const avg=mean(rewards),sd=Math.sqrt(mean(rewards.map(r=>(r-avg)**2)));return rewards.map(r=>scale?(sd<1e-12?0:(r-avg)/sd):r-avg);}
export function calibration(p,q){return {brier:q*(p-1)**2+(1-q)*p*p,logloss:-q*Math.log(p)-(1-q)*Math.log1p(-p),sampleReward:q*p+(1-q)*(1-p)};}

export function evaluateLoss(id,v){
  const out=(value,rows,steps,metrics=[])=>({value,rows,steps,metrics});
  switch(id){
  case 'mse': {
    const e=v.prediction-v.target,errors=[.1,-.2,e],square=mean(errors.map(x=>x*x)),mae=mean(errors.map(Math.abs));
    const huber=mean(errors.map(x=>Math.abs(x)<=v.delta?.5*x*x:v.delta*(Math.abs(x)-.5*v.delta)));
    return out(square,[['MSE',square,2*e/3],['MAE',mae,Math.sign(e)/3],['Huber',huber,Math.max(-v.delta,Math.min(v.delta,e))/3]],
      [`Residuals = [0.1, −0.2, ${e.toFixed(3)}].`,`MSE = (0.01 + 0.04 + ${e*e}) / 3.`,`A single outlier supplies ${e*e} of ${.05+e*e} squared-error mass.`]);
  }
  case 'geometry': {
    const a=v.angle*Math.PI/180,t=[1,0],s=[v.norm*Math.cos(a),v.norm*Math.sin(a)],c=cosine(t,s);
    return out(1-c,[['1 − cosine',1-c],['Coordinate MSE',mse(s,t)],['Cosine-score MSE',(c-v.target)**2]],
      [`Teacher = [1, 0]; student = [${s.map(x=>x.toFixed(3)).join(', ')}].`,`cos = dot / norms = ${c.toFixed(6)}; cosine-score target = ${v.target}.`,`Changing magnitude alone cannot change cosine. Both vectors must be nonzero.`]);
  }
  case 'classification': {
    const logits=[v.logit,0,-1],target=[1-v.smooth+v.smooth/3,v.smooth/3,v.smooth/3],p=softmax(logits),b=sigmoid(v.logit),pt=v.label?b:1-b;
    return out(ce(target,logits),[['Categorical CE',ce(target,logits),p[0]-target[0]],['Binary BCE',softplus(v.logit)-v.label*v.logit,b-v.label],['Focal (α = 1)',-((1-pt)**v.gamma)*Math.log(pt)]],
      [`Softmax([${logits}]) = [${p.map(x=>x.toFixed(4))}]. Smoothed target = [${target.map(x=>x.toFixed(3))}].`,`CE = log Σ exp(z) − target·z. BCE = softplus(z) − y z, y = ${v.label}.`,`Focal multiplies binary CE by (1 − p_correct)^γ = ${(1-pt)**v.gamma}.`]);
  }
  case 'contrastive': {
    const loss=v.label*v.distance**2+(1-v.label)*Math.max(0,v.margin-v.distance)**2;
    return out(loss,[['Positive branch d²',v.distance**2],['Negative branch [m−d]₊²',Math.max(0,v.margin-v.distance)**2],['Selected label y',v.label]],
      [`Distance d = ${v.distance}; label y = ${v.label} (1 = similar).`,`L = y·d² + (1−y)·max(0, m−d)² = ${loss}.`,`For a negative beyond margin ${v.margin}, gradient is zero; it gives no further separation.`]);
  }
  case 'infonce': {
    const scores=[v.positive,v.negative,.1],keep=v.policy===1?[0,2]:[0,1,2],positives=v.policy===2?[0,1]:[0],r=infoNCE(scores,v.temp,positives,keep);
    return out(r.loss,scores.map((s,i)=>[`doc ${i}${positives.includes(i)?' · positive':''}${!keep.includes(i)?' · excluded':''}`,r.p[i],r.gradient[i]]),
      [`Scores = [${scores}], τ = ${v.temp}; denominator includes docs [${keep}].`,`Loss averages −log p over positives [${positives}].`,`∂L/∂s_j = (p_j − target_j)/τ. A positive derivative lowers that score under gradient descent.`]);
  }
  case 'cosent': {
    const d=v.high-v.low,loss=softplus(-v.scale*d);
    return out(loss,[['Higher-label pair similarity',v.high],['Lower-label pair similarity',v.low],['Scaled ordering gap',v.scale*d],['∂L / ∂high',-v.scale*sigmoid(-v.scale*d)]],
      [`Only one pair-of-pairs ordering: L = log(1 + exp(λ(s_low−s_high))).`,`λ(s_low−s_high) = ${-v.scale*d}.`,`These are similarity scores of two labeled examples, not necessarily two documents for the same query.`]);
  }
  case 'triplet': {
    const gap=v.negative-v.positive,loss=Math.max(0,v.margin-gap);
    return out(loss,[['Positive distance',v.positive],['Negative distance',v.negative],['Hinge loss',loss],['Soft-margin loss',softplus(-gap)]],
      [`Distance gap d_negative − d_positive = ${gap}.`,`L = max(0, ${v.margin} − ${gap}) = ${loss}.`,`A zero loss says this triplet clears the chosen margin; it says nothing about unseen corpus negatives.`]);
  }
  case 'margin-mse': {
    const student=[v.positive+v.shift,v.negative+v.shift],teacher=[3,1],gap=student[0]-student[1];
    return out((gap-2)**2,[['Student score gap',gap],['Teacher score gap',2],['Margin MSE',(gap-2)**2],['Absolute score MSE',mse(student,teacher)]],
      [`Teacher scores [3,1] have gap 2. Student scores [${student}] have gap ${gap}.`,`L = ((${student[0]} − ${student[1]}) − (3 − 1))².`,`Adding ${v.shift} to both scores cancels in the gap. Multiplying their gap by a scale would not cancel.`]);
  }
  case 'ranknet': {
    const d=v.positive-v.negative,loss=softplus(-d)+(1-v.preference)*d;
    return out(loss,[['Score difference',d],['P(first beats second)',sigmoid(d)],['Pairwise BCE',loss],['∂L / ∂first',sigmoid(d)-v.preference]],
      [`P(first ≻ second) = sigmoid(${d}). Target preference = ${v.preference}.`,`L = softplus(−Δ) + (1−y)Δ.`,`At equal scores and a strict preference, L = ln 2 ≈ 0.6931.`]);
  }
  case 'listwise': {
    const r=listwise([v.first,v.second,v.third],[2,1,0]);
    return out(r[v.objective===0?'listnet':v.objective===1?'listmle':v.objective===2?'plistmle':'lambda'],[['ListNet top-one CE',r.listnet],['ListMLE (sum)',r.listmle],['Position-weighted ListMLE',r.plistmle],['ΔNDCG-weighted pair loss',r.lambda],['NDCG@3 (evaluation)',r.ndcg]],
      [`Grades [2,1,0], predicted order [${r.rank.map(i=>'doc '+i)}].`,`ListMLE suffix terms = [${r.terms.map(x=>x.toFixed(4))}]; position weights [7,3,1].`,`ΔNDCG weights for preference pairs = [${r.pairs.map(p=>`${p.i}≻${p.j}: ${p.weight.toFixed(4)}`).join('; ')}]. We freeze these weights within a gradient step.`]);
  }
  case 'kd': {
    const teacher=softmax([2,1,0],v.temp),student=softmax([v.first,v.second,0],v.temp),m=teacher.map((p,i)=>(p+student[i])/2);
    return out(v.temp**2*kl(teacher,student),[['T² KL(teacher ‖ student)',v.temp**2*kl(teacher,student)],['T² KL(student ‖ teacher)',v.temp**2*kl(student,teacher)],['JS (unscaled)',.5*(kl(teacher,m)+kl(student,m))]],
      [`Teacher p = [${teacher.map(x=>x.toFixed(4))}]; student q = [${student.map(x=>x.toFixed(4))}].`,`KL(p‖q) terms = [${teacher.map((p,i)=>(p*Math.log(p/student[i])).toFixed(4))}]. Individual terms may be negative; their sum is nonnegative.`,`Multiply the soft-target loss by T² here; hard-label CE would normally retain T = 1.`]);
  }
  case 'dist': {
    const p=[[.7,.2,.1],[.2,.6,.2],[.1,.3,.6]],q=p.map(row=>row.map(x=>v.scale*x+(1-v.scale)/3)),r=distLoss(p,q);
    return out(r.loss,[...r.rows.map((x,i)=>[`Row ${i} correlation`,x]),...r.cols.map((x,i)=>[`Column ${i} correlation`,x]),['Mean KL(p ‖ q)',mean(p.map((row,i)=>kl(row,q[i])))]],
      [`Student probabilities q = a p + (1−a)/3 with a = ${v.scale}. Every row still sums to 1.`,`DIST core = mean_rows(1−ρ) + mean_columns(1−ρ), with T=1 and equal component weights.`,`At a>0, every correlation is 1 although confidence can change. At a=0, variance is zero and Pearson correlation is undefined.`]);
  }
  case 'matryoshka': {
    const q=[1,v.tail],p=[1,v.tail],n=[1,-v.tail],r1=infoNCE([1,1],.5),r2=infoNCE([cosine(q,p),cosine(q,n)],.5),loss=v.weight*r1.loss+(1-v.weight)*r2.loss;
    return out(loss,[['1D prefix InfoNCE',r1.loss],['2D full InfoNCE',r2.loss],['Weighted objective',loss]],
      [`q = positive = [1, ${v.tail}], negative = [1, ${-v.tail}]. Normalize again after each truncation.`,`In 1D both cosine scores equal 1. In 2D the negative similarity is ${cosine(q,n).toFixed(4)}.`,`L = ${v.weight}·L_1D + ${1-v.weight}·L_2D. This fixed fixture reveals a bad prefix; training would change the vectors.`]);
  }
  case 'boundary': {
    const y=v.label?1:-1,hinge=Math.max(0,v.margin-y*(v.score-v.boundary)),phi=[-2,-1,1,2],prob=[.9,.8,v.outside,.1],terms=phi.map((x,i)=>x*prob[i]);
    return out(hinge,[['Signed decision margin',y*(v.score-v.boundary)],['Decision hinge',hinge],['Spatial boundary sum',sum(terms)],['Spatial loss change / outside probability',1]],
      [`Decision boundary b = ${v.boundary}; label sign y = ${y}; L = [${v.margin} − y(s−b)]₊.`,`Separate segmentation toy: signed distances φ = [−2,−1,1,2], foreground probabilities = [${prob}].`,`Spatial sum Σφp = ${terms.map(x=>x.toFixed(3)).join(' + ')} = ${sum(terms)}. Negative values are permitted; this is not a ranking objective.`]);
  }
  case 'reinforce': {
    const p=sigmoid(v.logit),adv=v.reward-v.baseline,loss=-adv*Math.log(p),expected=p*1+(1-p)*0;
    return out(loss,[['P(sampled action)',p],['Frozen advantage r−b',adv],['Sample loss',loss],['∂L / ∂action logit',-adv*(1-p)],['Expected reward if true rewards [1,0]',expected]],
      [`For one sampled action: L = −stopgrad(r−b) log π(a).`,`Here r−b = ${adv}. Holding this sample fixed, the logit gradient is ${(-adv*(1-p)).toFixed(4)}.`,`An action-independent baseline preserves the expected gradient; a well-chosen one can reduce variance. The sample loss is not negative expected reward itself.`]);
  }
  case 'ppo': {
    const value=-clipped(v.ratio,v.advantage,v.epsilon);
    return out(value,[['Raw surrogate r A',v.ratio*v.advantage],['Clipped surrogate',Math.max(1-v.epsilon,Math.min(1+v.epsilon,v.ratio))*v.advantage],['Actor loss = −min',value]],
      [`Probability ratio r = π_new(a|s)/π_old(a|s) = ${v.ratio}; stored advantage A = ${v.advantage}.`,`Clip interval [${1-v.epsilon}, ${1+v.epsilon}]. Take the smaller surrogate BEFORE negating.`,`For positive A only excess increases flatten; for negative A only excess decreases flatten. This omits critic, entropy and KL terms.`]);
  }
  case 'grpo': {
    const rewards=[1,v.reward,0],a=groupAdvantages(rewards,!!v.normalize),lengths=[2,4,6],per=a.map(x=>-clipped(v.ratio,x,.2));
    const sequence=mean(per),token=sum(per.map((x,i)=>x*lengths[i]))/sum(lengths),fixed=sum(per.map((x,i)=>x*lengths[i]))/(3*6);
    return out(v.reduction===0?sequence:v.reduction===1?token:fixed,[['Sequence mean (GRPO)',sequence],['Active-token mean (DAPO style)',token],['Fixed B·Lmax (Dr. GRPO style)',fixed],...a.map((x,i)=>[`Completion ${i} advantage`,x])],
      [`One prompt, rewards [${rewards}], lengths [2,4,6], common ratio ${v.ratio}; population SD for this toy.`,`Each token within a completion shares its advantage. Sequence reduction weights examples equally; token reduction weights their lengths.`,`Zero within-group reward variance yields zero centered advantages. Changing reduction alone is not the complete DAPO or Dr. GRPO recipe.`]);
  }
  case 'dpo': {
    const gap=v.chosen-v.rejected,logit=v.beta*(gap-v.reference);
    return out(softplus(-logit),[['Student log-probability gap',gap],['Reference log-probability gap',v.reference],['β-adjusted gap',logit],['DPO loss',softplus(-logit)]],
      [`log π(chosen) = ${v.chosen}; log π(rejected) = ${v.rejected}; reference gap = ${v.reference}.`,`L = −log σ(β[(logπ_c−logπ_r)−(logπref_c−logπref_r)]).`,`Only the gap is constrained. Both response likelihoods can decrease while this preference loss improves.`]);
  }
  case 'simpo': {
    const gap=v.chosen/v.chosenLength-v.rejected/v.rejectedLength,logit=v.beta*gap-v.margin;
    return out(softplus(-logit),[['Chosen average log probability',v.chosen/v.chosenLength],['Rejected average log probability',v.rejected/v.rejectedLength],['β·average gap − γ',logit]],
      [`Sum log probabilities [${v.chosen}, ${v.rejected}], lengths [${v.chosenLength}, ${v.rejectedLength}].`,`L = softplus(γ − β(logπ_c/|c| − logπ_r/|r|)).`,`Length normalization can reverse the preference implied by total log probability; it does not guarantee factual quality.`]);
  }
  case 'contrastive-rlcd': {
    const gap=v.preferred-v.rejected,p=sigmoid(gap),loss=softplus(-gap)+(1-v.label)*gap;
    return out(loss,[['Reward-model preference probability',p],['Synthetic preference target',v.label],['Reward-model BCE',loss],['∂L / ∂preferred score',p-v.label]],
      [`A positive prompt generates response A; a contrasting prompt generates B. Synthetic label assumes A is preferable (adjust it here).`,`Fit a preference/reward model using Δr = ${gap} and BCE. Then use the learned reward in an RL update.`,`This lab computes the preference-fitting stage; contrastive prompting is data construction, not a new primitive loss.`]);
  }
  case 'rlcd': {
    const c=calibration(v.probability,v.frequency),p1=v.probability,p2=v.other,avg=(p1+p2)/2;
    const per=(calibration(p1,v.frequency).brier+calibration(p2,v.frequency).brier)/2,mix=calibration(avg,v.frequency).brier,variance=((p1-avg)**2+(p2-avg)**2)/2;
    return out(c.brier,[['Expected binary Brier',c.brier],['Expected log loss',c.logloss],['Expected sampled correctness reward',c.sampleReward],['Mean per-rationale Brier',per],['Brier of mean prediction',mix],['Variance across rationales',variance]],
      [`True event frequency q = ${v.frequency}; reported probability p = ${v.probability}.`,`E[(p−y)²] = q(p−1)² + (1−q)p². Its minimum is p=q, while sampled correctness is linear in p.`,`Two-rationale identity: mean Brier = Brier(mean p) + variance = ${mix.toFixed(4)} + ${variance.toFixed(4)}. Binary scalar convention; full two-class Brier is twice this.`]);
  }
  case 'sdpo': {
    const p=softmax([v.logit,0]),teacher=softmax([v.feedback,0]),contributions=p.map((x,i)=>x*Math.log(x/teacher[i]));
    return out(kl(p,teacher),[['Policy P(correct token)',p[0]],['Feedback teacher P(correct token)',teacher[0]],['KL(policy ‖ teacher)',sum(contributions)]],
      [`At a fixed on-policy prefix, π = [${p.map(x=>x.toFixed(4))}], feedback-conditioned self-teacher = [${teacher.map(x=>x.toFixed(4))}].`,`Reverse-KL slice = Σ π log(π/p_teacher) = ${contributions.map(x=>x.toFixed(4)).join(' + ')}.`,`Teacher probabilities are held fixed during this calculation. A full trainer also samples trajectories, constructs feedback and handles policy updates.`]);
  }
  default:throw Error(`Unknown loss family: ${id}`);
  }
}
