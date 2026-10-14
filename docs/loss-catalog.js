const torch=['PyTorch loss definitions','https://docs.pytorch.org/docs/stable/nn.html#loss-functions'];
const bi=['Sentence Transformers: embedding objectives','https://www.sbert.net/docs/sentence_transformer/loss_overview.html'];
const biRef=['Embedding loss reference','https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html'];
const cross=['Sentence Transformers: ranking objectives','https://www.sbert.net/docs/cross_encoder/loss_overview.html'];
const crossRef=['Cross-encoder loss reference','https://www.sbert.net/docs/package_reference/cross_encoder/losses.html'];
const paper=(title,id)=>[title,`https://arxiv.org/abs/${id}`];
const c=(key,label,min,max,step,value)=>({key,label,min,max,step,value});
const choice=(key,label,options,value=0)=>({key,label,options,value});
// Text is original teaching material. Sources identify definitions and published methods.
export const losses=[
{
 id:'mse',group:'Match targets',title:'MSE, MAE & Huber',tag:'Regression · feature distillation',
 principle:'A prediction should be close to a numeric target. A Gaussian observation model makes squared error a negative log likelihood up to a constant and scale; a Laplace model leads to absolute error. These assumptions decide how expensive an outlier becomes.',
 derivation:'Write the residual e = prediction − target. Squaring gives gradient 2e; absolute error gives sign(e). Huber is quadratic near zero and linear beyond δ, capping the residual gradient at δ. Reductions matter: a mean over three examples divides every gradient by three.',
 equation:'MSE = mean(e²); MAE = mean(|e|)\nHuberδ(e) = ½e² if |e| ≤ δ; otherwise δ(|e| − ½δ)',
 toy:'Three errors share one objective. Move the third prediction; compare the loss and its contribution to the gradient.',
 controls:[c('prediction','Third prediction',-5,10,.1,2),c('target','Third target',-2,2,.1,1),c('delta','Huber δ',.1,3,.1,1)],failure:{prediction:10,target:0,delta:1},
 failureText:'The outlier preset gives errors [0.1, −0.2, 10]. MSE = 33.35, almost entirely from one example. MAE resists outliers but has a kink; Huber still needs a useful scale δ. Coordinate matching also penalizes a harmless change of embedding basis. A small regression loss need not preserve ranking.',
 fit:['Good for aligned teacher embeddings or absolute similarity labels; use a projection if dimensions differ.','Good for calibrated teacher scores with a shared scale; relative ranking may favor MarginMSE.','Useful for hidden-state/value-head distillation; vocabulary IDs are categorical, so never regress their numeric IDs.'],
 modern:'Current embedding APIs support projected feature matching. Huber is a robustness choice, not a universally newer or better MSE. Keep a retrieval or ranking validation metric alongside this training loss.',sources:[torch,biRef]
},
{
 id:'geometry',group:'Match targets',title:'Cosine loss & cosine-score regression',tag:'Direction versus coordinate magnitude',
 principle:'When an embedding represents direction, positive scaling should not change its meaning. Normalize the dot product by both vector norms to remove magnitude. Matching a labeled similarity score is a separate regression problem.',
 derivation:'For u and v, cos(u,v) = u·v/(||u|| ||v||). Positive-pair alignment uses 1−cos. Similarity regression uses (cos−y)². If both vectors are unit length, ||u−v||² = 2(1−cos); this is squared distance, not mean coordinate MSE.',
 equation:'L_direction = 1 − cos(u,v)\nL_score = (cos(u,v) − y)²\nFor unit vectors: ||u−v||² = 2 − 2 cos(u,v)',
 toy:'The teacher points along [1,0]. Rotate and scale the student to distinguish matching direction from matching coordinates.',
 controls:[c('angle','Student angle (degrees)',0,180,1,60),c('norm','Student norm',.1,5,.1,2),c('target','Similarity target y',-1,1,.1,1)],failure:{angle:0,norm:5,target:1},
 failureText:'At angle 0 and norm 5, cosine alignment loss is zero while coordinate MSE is 8. Norm information is invisible. Attraction-only training can map every input to the same direction; negatives or other supervision are needed. At 0° and 180°, the angle derivative is zero, including at the antipodal maximum. Zero vectors make cosine undefined.',
 fit:['Natural for semantic similarity and feature-direction distillation; validate norms if dot-product retrieval is used.','Use only if the output is an embedding or a cosine-based score; a scalar relevance head does not need cosine.','Auxiliary hidden-state alignment after a justified projection; token CE is still needed for language behavior.'],
 modern:'CoSENT changes supervision from absolute similarity to ordering. AnglE explores angle-based optimization. Neither licenses replacing a calibrated numeric target with an uncalibrated ranking objective without evaluating that change.',sources:[bi,biRef,paper('AnglE','2309.12871')]
},
{
 id:'classification',group:'Match targets',title:'Cross-entropy, BCE, smoothing & focal loss',tag:'Likelihood · decoder SFT · pointwise relevance',
 principle:'Assign high probability to observed events by minimizing negative log likelihood. A Bernoulli event produces binary cross-entropy; a categorical choice produces softmax cross-entropy. Teacher probabilities can replace hard labels.',
 derivation:'Softmax normalizes competing logits. CE = logsumexp(z) − target·z, so its logit gradient is p−target. BCE uses a single logit with a sigmoid. Smoothing spreads some target mass; focal loss downweights easy binary examples rather than modeling a new probability distribution.',
 equation:'CE = −Σ y_j log softmax(z)_j; ∂CE/∂z_j = p_j−y_j\nBCE(z,y) = softplus(z) − yz\ny_smooth = (1−ε)y + ε/K; focal = −(1−p_correct)^γ log p_correct',
 toy:'Categorical CE uses logits [z,0,−1] with class 0 as the target. The separate binary calculation uses the selected label and the same scalar z. Focal uses α=1.',
 controls:[c('logit','Class-0 / binary logit',-6,6,.1,1),c('smooth','Smoothing ε',0,.6,.05,0),choice('label','Binary label',['0 · negative','1 · positive'],1),c('gamma','Focal γ',0,4,.1,2)],failure:{logit:-6,smooth:0,label:1,gamma:2},
 failureText:'A confidently wrong label creates a large CE penalty. Mislabeled data therefore teaches the wrong answer with confidence. Focal emphasizes hard cases, including label errors, and suppresses easy-case gradients; it is not a calibration guarantee. Smoothing deliberately prevents a one-hot optimum. Decoder teacher forcing does not train recovery from every self-generated prefix.',
 fit:['Use CE over candidates or classes; binary relevance training needs representative negatives.','Strong pointwise baseline for relevance or class probabilities; soft labels support score distillation.','Foundation for next-token pretraining and SFT. Average over intended target tokens and mask prompt/padding tokens.'],
 modern:'Stable logit-based CE/BCE remains a strong baseline. Focal or class weights address imbalance only when the sampling and label quality justify them. Distillation changes targets; SFT changes data and token masks, not the CE identity.',sources:[torch,cross,paper('Focal Loss','1708.02002')]
},
{
 id:'contrastive',group:'Retrieve & rank',title:'Distance contrastive loss',tag:'Labeled similar / dissimilar pairs',
 principle:'A representation should put similar items together and dissimilar items far enough apart. The negative branch uses a safety margin so already separated examples stop consuming gradient.',
 derivation:'Use distance d ≥ 0 and label y=1 for similar. The positive branch is d². The negative branch squares the remaining margin violation. This distance contrastive loss is different from the softmax contrastive loss called InfoNCE.',
 equation:'L = y d² + (1−y) max(0,m−d)²',
 toy:'Set the pair label, distance, and margin. The table exposes both branches before choosing the labeled one.',
 controls:[c('distance','Pair distance d',0,4,.05,.5),c('margin','Negative margin m',.1,3,.1,1),choice('label','Pair type',['Dissimilar · y=0','Similar · y=1'],1)],failure:{distance:.2,margin:1,label:0},
 failureText:'If a true semantic neighbor is mislabeled as dissimilar, the negative branch pushes it apart. Easy negatives beyond the margin give zero signal. Without enough dissimilar pairs, attraction can collapse embeddings.',
 fit:['Direct fit for labeled pairs in metric learning.','Possible on a learned distance or pair representation; BCE is often more direct for a scalar relevance score.','Auxiliary representation learning; not a complete next-token generation objective.'],
 modern:'Online hard/semi-hard mining and multi-negative objectives change which comparisons matter. Mine difficult negatives while checking whether they are actually relevant.',sources:[biRef,torch]
},
{
 id:'infonce',group:'Retrieve & rank',title:'InfoNCE / MNRL, multiple positives & guided negatives',tag:'Candidate discrimination · embedding fine-tuning',
 principle:'Treat retrieval within a sampled candidate set as classification: the relevant document should receive probability mass relative to the other candidates. Temperature sets how strongly score differences affect the competition.',
 derivation:'Divide candidate scores by τ, normalize with softmax, and take negative log probability of the positive. The score gradient is (p−target)/τ. For several positives this lab averages their log losses, giving a uniform positive target, not a log-of-summed-positive-mass objective.',
 equation:'L = −log exp(s+)/Σ_j exp(s_j), using scaled scores s/τ\nMulti-positive: L = −mean_{i∈P} log p_i\nGuided filtering: omit a suspect negative from the denominator',
 toy:'Document 0 is relevant. Document 1 is hard and may be another relevant answer. Compare treating it as negative, excluding it, and labeling both as positives.',
 controls:[c('positive','Relevant score',-1,1,.05,.7),c('negative','Hard candidate score',-1,1,.05,.8),c('temp','Temperature τ',.05,1,.05,.2),choice('policy','Candidate treatment',['Single positive','Exclude suspect doc 1','Two labeled positives'])],failure:{positive:.7,negative:.95,temp:.1,policy:0},
 failureText:'The failure preset gives the hard candidate a positive gradient, pushing its score down. If it is another valid answer, that is a false-negative error. Removing too many negatives makes the task trivial; a mistaken guide can discard useful competitors. Low batch loss does not establish recall over the full corpus.',
 fit:['Primary retrieval fine-tuning choice for query–positive pairs with checked negatives.','Applicable to jointly scored query/candidate lists, at higher all-pairs computation cost.','Useful for candidate/representation contrast; separate from token CE and sequence-policy optimization.'],
 modern:'Cached MNRL uses gradient caching to enlarge the effective contrastive batch without changing its objective. GISTEmbed uses guide scores to filter negatives; the manual exclusion here isolates that effect, not its complete filtering procedure.',sources:[bi,paper('GISTEmbed (2024)','2402.16829'),paper('Gradient caching','2101.06983')]
},
{
 id:'cosent',group:'Retrieve & rank',title:'CoSENT: order labeled similarities',tag:'Similarity supervision without score calibration',
 principle:'If one sentence pair has a higher similarity label than another, its predicted similarity should be higher. The exact numeric difference between their labels is not required.',
 derivation:'Collect every unequal-label ordering and apply log(1 + Σ exp(λ(s_low−s_high))). The added 1 gives a zero baseline in the large-correct-margin limit. With one ordering this is a softplus. Label ties contribute no ordering constraint.',
 equation:'L = log[1 + Σ_{y_i>y_j} exp(λ(s_j−s_i))]',
 toy:'Two labeled sentence pairs supply one inequality. Watch a reversed ordering and the effect of score scale λ.',
 controls:[c('high','Higher-label pair score',-1,1,.05,.6),c('low','Lower-label pair score',-1,1,.05,.2),c('scale','Scale λ',1,30,1,10)],failure:{high:.2,low:.8,scale:20},
 failureText:'Reversing the two scores produces a large ordering penalty. Conversely, correct ordering can give a tiny loss even when absolute similarity values are wrong. Noisy ordinal labels at large λ give forceful wrong gradients.',
 fit:['Good for ordinal semantic-similarity labels; current embedding APIs offer it as an alternative to cosine-score MSE.','A scalar-score ordering loss is possible; group comparisons carefully so scores have meaningful comparability.','Auxiliary similarity/ranking only; does not define a token-generation distribution.'],
 modern:'AnglE alters the similarity geometry, while CoSENT alters the score-order objective. Select with held-out similarity/ranking metrics rather than assuming either dominates calibrated regression.',sources:[biRef,paper('AnglE','2309.12871')]
},
{
 id:'triplet',group:'Retrieve & rank',title:'Triplet margin & soft margin',tag:'One anchor, one positive, one negative',
 principle:'Relative distance matters: the anchor should be nearer to its positive than its negative by at least m. This removes the need to assign absolute target distances.',
 derivation:'The desired inequality is d(a,n) ≥ d(a,p)+m. The hinge is the positive part of its violation. A common smooth alternative uses softplus(d(a,p)−d(a,n)); it never becomes exactly zero at a finite gap.',
 equation:'L_hinge = max(0, d(a,p) − d(a,n) + m)\nL_soft = log(1 + exp(d(a,p)−d(a,n)))',
 toy:'Distances are directly controlled scalars; no hidden embedding optimization is implied.',
 controls:[c('positive','Anchor–positive distance',0,3,.05,.4),c('negative','Anchor–negative distance',0,3,.05,1),c('margin','Margin m',0,2,.05,.5)],failure:{positive:.1,negative:3,margin:.5},
 failureText:'The easy triplet has zero hinge loss and no gradient, even if an unseen document outranks the positive. Hard mining increases signal but can select mislabeled or pathological examples. A margin incompatible with bounded distances cannot be satisfied.',
 fit:['Natural for triplet data, hard/semi-hard mining and metric learning.','Use score-margin hinge max(0,m−s+ + s−), with the reversed sign because larger scores are better.','Auxiliary ranking of candidates or representations; not a standalone LM likelihood.'],
 modern:'Batch-hard and semi-hard losses are mining strategies around this geometry. Soft margins keep a diminishing signal after an inequality is satisfied.',sources:[biRef]
},
{
 id:'margin-mse',group:'Distill a teacher',title:'MarginMSE: transfer score differences',tag:'Cross-encoder → bi-encoder or reranker distillation',
 principle:'A teacher may assign relevance scores on an arbitrary query-specific scale. Preserve the score difference between two candidates rather than their absolute offset.',
 derivation:'Subtract negative from positive for both models, then regress the teacher gap with squared error. A shared additive shift cancels; multiplicative scale does not. The teacher must score the same query and candidates as the student.',
 equation:'L = [(s(q,d₁)−s(q,d₂)) − (t(q,d₁)−t(q,d₂))]²',
 toy:'Teacher scores stay [3,1]. Shift both student scores together to separate margin matching from absolute score matching.',
 controls:[c('positive','Student first score',-3,5,.1,2),c('negative','Student second score',-3,5,.1,.5),c('shift','Common student offset',-10,10,.5,0)],failure:{positive:3,negative:1,shift:10},
 failureText:'The shifted preset has zero margin error despite badly mismatched absolute scores. That is desirable for offset-invariant ranking, but unsuitable if probabilities or thresholds must match. Incorrect teacher gaps, score scale and unrepresentative negatives transfer their errors to the student.',
 fit:['Strong distillation baseline when a cross-encoder supplies candidate score gaps.','Useful for compressing a teacher reranker with pairwise supervision.','Can match reward/ranking gaps; token distribution distillation needs a separate vocabulary objective.'],
 modern:'Multi-negative MarginMSE supplies several gaps per query. Compare with candidate-list KL when teacher uncertainty over a whole list matters.',sources:[bi,cross,paper('Improving Efficient Neural Ranking Models with Cross-Architecture KD','2010.02666')]
},
{
 id:'ranknet',group:'Retrieve & rank',title:'RankNet / Bradley–Terry preference loss',tag:'Pairwise ordering · reward models',
 principle:'Model a preference as a probability: a score difference represents log odds that one candidate beats another. A pairwise Bernoulli likelihood turns that probability into a loss.',
 derivation:'P(i ≻ j)=σ(s_i−s_j). For preference y, apply BCE to the score difference. With a strict first-candidate preference, L=softplus(s_j−s_i). A common score offset is unidentifiable.',
 equation:'Δ = s_i−s_j; P(i≻j)=σ(Δ)\nL = −y log σ(Δ) − (1−y) log[1−σ(Δ)]',
 toy:'Use y=0.5 for an uncertain/tied preference. Use y=1 for a strict preference.',
 controls:[c('positive','First score',-4,4,.1,1),c('negative','Second score',-4,4,.1,0),c('preference','Preference target y',0,1,.05,1)],failure:{positive:4,negative:-4,preference:0},
 failureText:'A confidently reversed preference gets a large loss. Correct but increasingly large margins can keep growing with separable labels; this need not calibrate absolute relevance. Click bias and synthetic preference errors remain supervision errors.',
 fit:['Works on query–document similarity differences with sampled pairs.','Direct fit for reranking pairs and scalar reward-model training.','Fits a reward model or candidate scorer; a language policy requires probabilities or an RL objective.'],
 modern:'Lambda-style weighting prioritizes pairs according to metric impact. DPO reuses a logistic preference form with policy/reference log-probability ratios, not ordinary relevance scores.',sources:[crossRef,paper('DPO','2305.18290')]
},
{
 id:'listwise',group:'Retrieve & rank',title:'ListNet, ListMLE, PListMLE & Lambda weighting',tag:'Whole candidate lists · ranking distillation',
 principle:'Ranking is about the arrangement of a list. Different surrogates preserve different things: top-one probability mass, a permutation likelihood, or the metric cost of swapping two items.',
 derivation:'Top-one ListNet is CE between softmax grades and softmax scores. ListMLE factorizes the teacher order as sequential choices from remaining items. Position-weighted ListMLE emphasizes early choices. The LambdaRank-style toy weights logistic pairs by the absolute NDCG change from a swap; weights are treated as fixed for a local gradient.',
 equation:'ListNet = CE(softmax(grades), softmax(scores))\nListMLE = Σ_i [log Σ_{j≥i} exp(s_{π_j}) − s_{π_i}]\nPListMLE toy = Σ_i (2^(N−i)−1) · suffix_term_i, zero-based i\nLambda toy = Σ_{g_i>g_j} |ΔNDCG_ij| · softplus(s_j−s_i)',
 toy:'Three documents have relevance grades [2,1,0]. Select the objective plotted; all four are computed below. NDCG uses gains 2^grade−1 and logarithmic discounts.',
 controls:[c('first','Doc 0 score (grade 2)',-3,3,.1,1),c('second','Doc 1 score (grade 1)',-3,3,.1,.5),c('third','Doc 2 score (grade 0)',-3,3,.1,0),choice('objective','Objective to inspect',['ListNet','ListMLE','PListMLE toy','Lambda-weighted pairs'],3)],failure:{first:-2,second:0,third:2,objective:3},
 failureText:'The reversed list hurts the most useful positions. ListNet depends on grade scaling; ListMLE needs tie handling; position weights add a top-rank bias. Lambda weights change at rank crossings, so the objective is piecewise. All four see only the candidate list: none can recover a relevant document absent from retrieval.',
 fit:['Can distill a reranker list into query/document similarities; candidate coverage and in-batch bias matter.','Natural for list-supervised reranking; evaluate NDCG/MRR on realistic retrieved candidates.','Applicable to reranking generated candidates; not ordinary next-token CE.'],
 modern:'Current reranker APIs include LambdaLoss and PListMLE. Their exact reductions/weighting options differ; this lab declares its own equations and is not a byte-for-byte reproduction of every library variant.',sources:[cross,crossRef]
},
{
 id:'kd',group:'Distill a teacher',title:'Soft-target KL, reverse KL & Jensen–Shannon',tag:'Distributions, temperature and support',
 principle:'A teacher distribution contains uncertainty that a single winning label discards. Match distributions on the same event space: candidate documents, classes, or vocabulary tokens.',
 derivation:'KL(p_teacher ‖ q_student)=Σp log(p/q). Since teacher entropy is constant, forward KL and soft-target CE have the same student gradient. Reverse KL weights disagreement by the student. JS compares both to their mean. Temperature changes the targets; a T² factor compensates the softened-logit gradient scale in classic KD.',
 equation:'p_T = softmax(z_teacher/T); q_T = softmax(z_student/T)\nL_KD = T² KL(p_T ‖ q_T)\nReverse KL = KL(q ‖ p); JS = ½KL(p‖m)+½KL(q‖m), m=(p+q)/2',
 toy:'Teacher logits are [2,1,0]. Move the student logits or temperature and inspect all divergence terms.',
 controls:[c('first','Student logit 0',-5,5,.1,0),c('second','Student logit 1',-5,5,.1,2),c('temp','Temperature T',.5,5,.1,1)],failure:{first:5,second:-5,temp:1},
 failureText:'A student concentrated on one outcome can miss teacher-supported alternatives. Forward KL strongly penalizes missing teacher mass; reverse KL penalizes student mass where the teacher assigns little. These tendencies are not universal quality guarantees. Zero support can make KL infinite. Misaligned tokenizers/candidates or an incorrect teacher invalidate direct matching.',
 fit:['Candidate-list KL transfers cross-encoder rankings; align candidate IDs exactly.','Class or candidate distributions can be distilled; raw unbounded scores need normalization.','Token-level KD when vocabulary events align; teacher-generated SFT is hard-target sequence distillation, not equivalent to full-logit KD.'],
 modern:'Generalized KD explores divergence choice and student-generated trajectories. On-policy training changes the prefixes being learned, not just KL direction. DistiLLM-2 adds a contrastive teacher/student-response design; it is a larger recipe than swapping this scalar loss.',sources:[paper('Classic KD','1503.02531'),paper('Generalized Knowledge Distillation','2306.13649'),paper('DistiLLM-2 (2025)','2503.07067')]
},
{
 id:'dist',group:'Distill a teacher',title:'DIST: correlation-based distillation',tag:'Named method · distinct from generic “distillation loss”',
 principle:'A smaller student may preserve how outputs vary even when it cannot match every probability. Pearson correlation compares centered directions, discarding a positive affine scale and offset.',
 derivation:'DIST matches probability relations across classes within each example and across examples within each class. Compute ρ on rows and columns of aligned batch-by-class matrices, then minimize 1−ρ. This toy uses equal weights and T=1, showing only the relational terms.',
 equation:'ρ(a,b) = [(a−ā)·(b−b̄)] / [||a−ā|| ||b−b̄||]\nL_DIST core = mean_rows(1−ρ) + mean_columns(1−ρ)',
 toy:'The teacher has three nonuniform class distributions. The student mixes each with uniform probability: q = a p + (1−a)/3.',
 controls:[c('scale','Teacher mixture a',0,1,.05,.5)],failure:{scale:.05},
 failureText:'At a=0.05, both correlation terms are zero although the student is almost uniform. Correlation matching does not preserve calibration. At a=0 the variance vanishes and correlation is mathematically undefined; implementations need an explicit numerical convention. Column relations require the same class semantics across rows, not unrelated candidate positions.',
 fit:['Possible representation/relation research adaptation; not an established universal retrieval replacement. Candidate columns must align.','Potential for aligned class-output distillation; scalar scores alone do not supply a class relation matrix.','Vocabulary-aligned distributions could support the relation, but this paper’s evidence is primarily vision; validate language outcomes independently.'],
 modern:'DIST (NeurIPS 2022) is a specific correlation objective, not the newest name for every KD loss. Keep task supervision and calibration/ranking evaluation when adapting it.',sources:[paper('Knowledge Distillation from A Stronger Teacher','2205.10536')]
},
{
 id:'matryoshka',group:'Distill a teacher',title:'Matryoshka: train useful embedding prefixes',tag:'A wrapper around a base objective',
 principle:'If deployment truncates an embedding, train the prefixes that will actually be used. Optimizing only the full vector does not force its leading coordinates to retain the useful signal.',
 derivation:'Evaluate a base loss independently at several prefix dimensions and sum weighted terms. For cosine objectives, renormalize each truncated vector. The wrapper changes which representations are supervised, not the definition of the underlying contrastive or distillation loss.',
 equation:'L = Σ_{d∈D} w_d L_base(normalize(u[:d]), normalize(v[:d]))',
 toy:'The first coordinate cannot distinguish the positive and negative. The second can. Compare losses at dimensions 1 and 2 and change the prefix weight.',
 controls:[c('tail','Useful second coordinate',0,3,.1,2),c('weight','Weight on 1D loss',0,1,.05,.5)],failure:{tail:3,weight:0},
 failureText:'A nearly perfect full vector can still have 1D loss ln 2 because both candidates share the same prefix. With zero prefix weight there is no training pressure to fix that. Heavy low-dimension weighting can sacrifice full-vector quality; a random already-trained embedding is not automatically truncatable.',
 fit:['Useful when one embedding model must serve several storage/latency budgets; compatible base losses still need suitable labels.','A scalar cross-encoder relevance head has no embedding prefix to truncate; only relevant to special representation outputs.','Possible hidden-representation research objective; not a drop-in LM loss or KV-cache compression guarantee.'],
 modern:'Matryoshka and adaptive-layer wrappers target different deployment dimensions. Evaluate every chosen dimension/layer, not only the full model.',sources:[paper('Matryoshka Representation Learning','2205.13147'),biRef]
},
{
 id:'boundary',group:'Match targets',title:'Boundary losses: name the boundary first',tag:'Decision margin versus spatial segmentation',
 principle:'“Boundary loss” is not one universal objective. A relevance threshold defines a decision boundary in score space. A segmentation boundary is a geometric contour in an image. Their targets and units are different.',
 derivation:'For a decision threshold b and label sign y∈{−1,+1}, enforce y(s−b)≥m with a hinge. Separately, the named segmentation Boundary Loss integrates foreground probabilities against a signed ground-truth distance field φ. It is normally combined with a regional objective.',
 equation:'Decision hinge = max(0, m − y(s−b))\nSpatial boundary term = Σ_pixel φ_G(pixel) · p_foreground(pixel)',
 toy:'The headline and plot use the score-boundary hinge. The same table separately computes a four-pixel signed-distance example, so the two meanings cannot be mistaken for each other.',
 controls:[c('score','Relevance score s',-2,2,.1,.4),c('boundary','Decision threshold b',-1,1,.1,0),c('margin','Required margin m',0,1,.1,.5),choice('label','Decision label',['Negative','Positive'],1),c('outside','Foreground probability outside contour',0,1,.05,.2)],failure:{score:1,boundary:0,margin:.5,label:0,outside:.9},
 failureText:'The wrong-side example violates the score boundary. A fixed threshold can fail after score calibration shifts, even with unchanged rankings. The spatial term may be negative and has distance units; applying it to document ranks without a geometric target is not meaningful. Background-only collapse and regional accuracy require joint supervision in segmentation.',
 fit:['Decision margins can train thresholded similarity; ordinary retrieval uses relative ordering. Spatial boundary loss is not a generic bi-encoder loss.','Decision margins can supervise relevance cutoffs; validate threshold calibration. Spatial loss needs a segmentation task/head.','A text decoder does not acquire a spatial contour from its name. Only specialized structured outputs supply such geometry.'],
 modern:'For ranking, margin, pairwise and metric-weighted objectives name the relevant boundary precisely. Do not label an arbitrary score penalty as a reproduction of a published Boundary Loss.',sources:[['Boundary Loss (Kervadec et al.)','https://proceedings.mlr.press/v102/kervadec19a.html'],torch]
},
{
 id:'reinforce',group:'Policy & preference',title:'RL: REINFORCE and the score-function gradient',tag:'Expected reward, not supervised target matching',
 principle:'A policy chooses an action or completion and receives a reward. Optimize expected reward even when the reward function is not differentiable through that sampled action.',
 derivation:'Differentiate J=Σ_a πθ(a)r(a): ∇J=E[r(a)∇logπθ(a)]. Subtracting an action-independent baseline leaves the expectation unchanged because E[∇logπ]=0. A loss for gradient descent negates this estimator and treats reward/advantage as fixed.',
 equation:'J = E_{a~πθ}[r(a)]\n∇J = E[(r−b)∇logπθ(a)]\nSample actor loss = −stopgrad(r−b) logπθ(a)',
 toy:'A binary policy has one sampled action. Change its reward, baseline and logit. The sample gradient is explicit; the true expected-reward illustration assumes rewards [1,0].',
 controls:[c('logit','Sampled-action logit',-4,4,.1,0),c('reward','Observed reward',-1,2,.1,1),c('baseline','Action-independent baseline',-1,2,.1,.5)],failure:{logit:2,reward:0,baseline:1},
 failureText:'A negative advantage reduces the sampled action’s likelihood even if its raw reward is nonnegative. One noisy sample can move the wrong way. Reward hacking optimizes the specified reward, and a baseline that depends on the sampled action can bias the estimator unless corrected. Sparse rewards make credit assignment difficult.',
 fit:['Possible for sampled retrieval/slate policies with rewards and exploration; supervised contrastive training is usually simpler.','Possible for interactive ranking policies; a deterministic score alone is not a policy distribution.','Direct fit for sequence generation, RLHF or RLVR; reward source and optimizer are separate choices.'],
 modern:'A verifier supplies RLVR rewards; a learned preference model supplies RLHF-style rewards. PPO/GRPO change update estimation and stabilization, not the need for a trustworthy reward.',sources:[paper('PPO','1707.06347'),['REINFORCE (Williams)','https://link.springer.com/article/10.1007/BF00992696']]
},
{
 id:'ppo',group:'Policy & preference',title:'PPO: clipped policy improvement',tag:'An actor surrogate, not a hard probability constraint',
 principle:'Reuse samples from an older policy, but limit the incentive to move too far in their rewarded direction. The probability ratio corrects which policy produced the samples.',
 derivation:'Multiply the stored advantage by π_new/π_old. Compare that surrogate with a clipped-ratio surrogate and take the pessimistic minimum. Negate it for minimization. Clipping is asymmetric after multiplying by a signed advantage.',
 equation:'r = π_new(a|s)/π_old(a|s)\nL_actor = −min(r A, clip(r,1−ε,1+ε) A)',
 toy:'One action isolates the actor term. Keep A fixed while inspecting the probability ratio.',
 controls:[c('ratio','Probability ratio r',.1,2,.05,1.1),c('advantage','Advantage A',-2,2,.1,1),c('epsilon','Clip ε',.05,.5,.05,.2)],failure:{ratio:1.8,advantage:1,epsilon:.2},
 failureText:'With A=1, r=1.8, ε=0.2 the actor loss is −1.2 and has no local incentive to increase r further. Clipping does not enforce a global trust region or prevent other samples from moving that probability. Noisy advantages, stale rollouts and bad rewards remain problems. Full PPO also trains a value estimator and may add entropy/KL terms.',
 fit:['Only for an explicit stochastic retrieval/action policy; not required for pair-label training.','Only when reranking is framed as a policy with reward and rollout probabilities.','Common policy-update building block for decoder RL; monitor reward, KL, entropy and length together.'],
 modern:'Sequence/token aggregation and KL choices materially change LM implementations. Do not interpret the clipped actor loss alone as the complete training objective.',sources:[paper('PPO','1707.06347')]
},
{
 id:'grpo',group:'Policy & preference',title:'GRPO, DAPO-style & Dr. GRPO-style reductions',tag:'Group-relative advantage · normalization matters',
 principle:'Several completions for the same prompt give a relative reward baseline. A learned value critic is not necessary for this group-centered advantage. How token losses are normalized then determines each completion’s weight.',
 derivation:'Center each reward by the group mean, optionally divide by its standard deviation, and use a clipped policy surrogate per token. Compare a mean of sequence means, a mean over all active tokens, and a fixed B·Lmax denominator. These are different gradients when lengths differ.',
 equation:'A_i = (r_i−mean(r))/(std(r)+ε), or unscaled r_i−mean(r)\nGRPO reduction: mean_i mean_token ℓ_it\nDAPO-style reduction: Σ_it ℓ_it / Σ_i length_i\nDr. GRPO-style reduction: Σ_it ℓ_it / (B·Lmax)',
 toy:'One prompt has three completions, lengths [2,4,6], rewards [1,r,0]. The same ratio is used at each token to isolate weighting. Population standard deviation is declared here.',
 controls:[c('ratio','Shared probability ratio',.5,1.5,.05,1.1),c('reward','Middle completion reward',0,1,.1,.5),choice('normalize','Advantage normalization',['Center only','Center and divide by group SD'],1),choice('reduction','Reduction to inspect',['Sequence mean','Active-token mean','Fixed B·Lmax'],0)],failure:{ratio:1.1,reward:.5,normalize:1,reduction:1},
 failureText:'Here sequence-mean loss cancels across centered advantages while the token mean does not: long low-reward completions get more weight. That is a weighting difference, not a quality verdict. An all-equal reward group gives zero relative signal. SD scaling also changes relative weights between easy and difficult prompts.',
 fit:['A research option for rewarded sampled retrieval actions; requires a meaningful action policy.','A research option for stochastic ranking/slate policies, not a default scalar scorer loss.','Applicable to sampled decoder completions with verifiable or learned rewards.'],
 modern:'DAPO and Dr. GRPO (2025) address issues beyond this normalization slice. Current TRL documents these reductions and further variants such as CISPO/SAPO. The lab isolates reductions and explicitly does not claim to reproduce the full recipes or identify a universal winner.',sources:[paper('DeepSeekMath / GRPO','2402.03300'),paper('DAPO','2503.14476'),paper('Dr. GRPO','2503.20783'),['Current TRL GRPO trainer','https://huggingface.co/docs/trl/main/grpo_trainer']]
},
{
 id:'dpo',group:'Policy & preference',title:'DPO: preferences relative to a reference policy',tag:'Offline preference fine-tuning',
 principle:'A preferred completion should improve relative to a rejected one, measured against a reference policy. Under a KL-regularized reward model, policy/reference log ratios act as implicit rewards.',
 derivation:'Insert implicit rewards β log(π/πref) into a Bradley–Terry preference model. The partition term cancels within a prompt, leaving a logistic loss on a difference of log-probability differences. Sequence log probabilities sum over response tokens.',
 equation:'h = [logπ(y+)−logπ(y−)] − [logπref(y+)−logπref(y−)]\nL_DPO = −log σ(β h)',
 toy:'The two sequence probabilities sum to at most one throughout the slider ranges; remaining mass can belong to other responses. The reference gap is fixed context, not a teacher relevance score.',
 controls:[c('chosen','Chosen log probability',-12,-.7,.1,-2),c('rejected','Rejected log probability',-12,-.7,.1,-4),c('reference','Reference log-probability gap',-5,5,.1,1),c('beta','β',.1,2,.1,.5)],failure:{chosen:-10,rejected:-12,reference:1,beta:.5},
 failureText:'Both the default pair [−2,−4] and failure pair [−10,−12] have the same gap and DPO loss. The objective alone does not require raising chosen likelihood. Label noise, reference mismatch, response length and poor coverage can make preference improvement fail to improve downstream behavior.',
 fit:['Possible for normalized retrieval policies with reference probabilities; ordinary relevance scores are not log-policy ratios.','Useful for an explicit candidate policy; use RankNet for an ordinary deterministic relevance head.','Direct fit for chosen/rejected completion data, with consistent response-token masks and reference model.'],
 modern:'SimPO replaces the reference-relative reward with length-normalized sequence likelihood and a margin. IPO and other preference losses change the surrogate; do not mix their β conventions or targets without checking their definitions.',sources:[paper('DPO','2305.18290'),paper('SimPO','2405.14734')]
},
{
 id:'simpo',group:'Policy & preference',title:'SimPO: length-normalized preference with a margin',tag:'Reference-free preference optimization',
 principle:'The average log probability per token supplies a reference-free response score. A target margin asks the chosen response to exceed the rejected score by more than a tie.',
 derivation:'Divide each sequence’s summed log probability by its token count. Scale their difference by β, subtract margin γ, and apply negative log sigmoid. Length normalization can change which response the score favors.',
 equation:'r(y) = β logπ(y)/|y|\nL = −log σ(r(y+)−r(y−)−γ)',
 toy:'Change lengths while keeping total log probabilities fixed. Watch the preference reverse.',
 controls:[c('chosen','Chosen total log probability',-20,-1,.5,-6),c('rejected','Rejected total log probability',-20,-1,.5,-8),c('chosenLength','Chosen tokens',1,10,1,3),c('rejectedLength','Rejected tokens',1,10,1,4),c('beta','β',.1,4,.1,2),c('margin','Margin γ',0,3,.1,1)],failure:{chosen:-6,rejected:-8,chosenLength:1,rejectedLength:8,beta:2,margin:1},
 failureText:'The failure pair has a higher total probability for the chosen response but a much lower per-token score. Length normalization is a modeling choice, not a guarantee against all length bias. A large margin or unreliable preferences can over-optimize stylistic likelihood instead of quality.',
 fit:['Only if outputs are meaningful variable-length policies; ordinary fixed-vector retrieval does not have response token lengths.','Not a natural default for scalar cross-encoder relevance. A generative reranker may have an appropriate sequence-policy formulation.','Fits offline completion preference training; compare with DPO using the same data and evaluation.'],
 modern:'SimPO (2024) removes the reference-policy forward pass for this reward definition. Evaluate retention and calibration rather than treating reference-free as automatically superior.',sources:[paper('SimPO','2405.14734')]
},
{
 id:'rlcd',group:'Policy & preference',title:'RLCD: Reinforcement Learning for Calibrated Decisions',tag:'Proper scoring foundations · TypeSafe goal / independent OpenJev example',
 principle:'TypeSafe uses RLCD for training Jev to return calibrated decisions. The mathematical goal is that probabilities reflect outcome frequencies. If an event occurs with frequency q, a truthful report p=q should minimize expected loss. Brier and log loss are proper scores; sampled correctness alone favors the most likely class.',
 derivation:'For a binary event, E[(p−y)²] = q(p−1)² + (1−q)p² = (p−q)² + q(1−q). The derivative is 2(p−q), so the optimum is q. Expected correctness of a sampled decision is qp+(1−q)(1−p), linear in p. Proper-score rewards can evaluate each rationale’s final distribution.',
 equation:'Binary Brier = (p−y)²; expected Brier = (p−q)² + q(1−q)\nLog loss = −y log p − (1−y) log(1−p)\nE_r Brier(p_r,y) = Brier(E_r p_r,y) + Var_r(p_r)\nIllustrative OpenJev-style reasoning objective (not a Jev reconstruction):\nmaximize E_r[−Brier(p_r,y)] − β KL(π_reasoning‖π_ref)',
 toy:'This proper-score toy is not a reconstruction of Jev’s trainer. Compare calibrated p=0.7 with overconfident p=0.99 at event frequency q=0.7. The independent OpenJev reasoning example adds a second rationale to distinguish scoring the mean from scoring each prediction.',
 controls:[c('probability','Reported probability p',.01,.99,.01,.7),c('frequency','True frequency q',.05,.95,.05,.7),c('other','Second rationale probability',.01,.99,.01,.3)],failure:{probability:.99,frequency:.7,other:.01},
 failureText:'At q=0.7, overconfidence increases expected Brier even though sampled correctness reward rises. Two disagreeing rationales can have a good average probability while each is poor. A good score in one dataset does not ensure calibration after distribution shift, and uncertainty from annotator disagreement is not necessarily removable by more reasoning.',
 fit:['Brier/log loss can supervise relevance probabilities; normalized embedding similarity is not automatically a probability. The RLCD name does not prescribe a bi-encoder architecture.','Natural for probabilistic relevance or classification; validate calibration under deployment sampling. Direct supervised proper-score training remains a baseline.','The independent OpenJev formulation trains a reasoning decoder. Jev itself exposes typed decisions without generated text; its architecture and exact trainer cannot be inferred from the RLCD name.'],
 modern:'TypeSafe’s launch states the calibration goal but does not specify a reproducible loss or training algorithm. The proper-score equations here explain that goal, not proprietary internals. Independently, OpenJev-RLCD (September 2026) calibrates a readout before reinforcing reasoning; its task-specific evidence and joint-training failures do not establish a universal replacement for CE or GRPO.',sources:[['TypeSafe: introducing Jev and RLCD','https://typesafe.ai/blog/introducing-system-one-models-and-jev?render=full'],['TypeSafe: typed decision interface','https://docs.typesafe.ai/introduction'],paper('OpenJev-RLCD: independent reasoning implementation','2609.38850')]
},
{
 id:'contrastive-rlcd',group:'Policy & preference',title:'Contrastive distillation: the earlier RLCD',tag:'2023 paper / ICLR 2024 · synthetic preference pipeline',
 principle:'Create preference supervision by eliciting contrasting outputs, then fit a preference model and optimize a language policy with its reward. The key intervention is how the training pairs are generated.',
 derivation:'A principle-following prompt supplies one response; an opposing prompt supplies another. Assume the former is preferred and fit logistic pairwise reward loss. The resulting reward model feeds a policy objective such as expected reward minus a reference KL penalty.',
 equation:'Pair stage: L_RM = −log σ(rφ(x,y+)−rφ(x,y−))\nPolicy stage: maximize E_{y~πθ}[rφ(x,y)] − β KL(πθ‖πref)',
 toy:'Inspect the reward-model pair stage. The synthetic label can be changed when the positive prompt actually failed to produce the better answer.',
 controls:[c('preferred','Reward score for prompted A',-3,3,.1,1),c('rejected','Reward score for contrasting B',-3,3,.1,0),c('label','Probability A is preferred',0,1,.05,1)],failure:{preferred:2,rejected:0,label:0},
 failureText:'The prompt polarity is only a proxy label. If A is actually worse, training on the assumed y=1 rewards the wrong behavior. Flipping the label here reveals the correct opposing gradient. Synthetic artifacts and preference-model shortcuts can survive into policy training.',
 fit:['Synthetic relevance pairs may be useful, but that adaptation is not automatically the published RLCD pipeline.','Could serve as the learned preference/reward scorer for paired outputs; validate synthetic label quality.','The published pipeline aligns a generative language policy via learned reward.'],
 modern:'This RLCD is distinct from reinforcement learning for calibrated decisions (2026), covered in its own lesson. It is also distinct from RLCSD, the 2026 contrastive on-policy self-distillation method.',sources:[paper('RLCD: Contrastive Distillation','2307.12950'),paper('RLCSD (distinct method)','2606.11709')]
},
{
 id:'sdpo',group:'Policy & preference',title:'SDPO: feedback-conditioned self-distillation',tag:'2026 research · dense feedback for decoder training',
 principle:'A model may infer a better next-token distribution after seeing a test failure, correction or successful attempt. Use that feedback-conditioned model as a self-teacher for the policy that must act without the feedback.',
 derivation:'At a sampled policy prefix, hold the self-teacher target fixed and compare the policy distribution to it. This lab isolates a reverse-KL slice. Full SDPO also specifies feedback construction, on-policy trajectories and the update estimator; the toy divergence alone is not the whole algorithm.',
 equation:'At a fixed prefix: L_slice = KL(πθ(·|x,prefix) ‖ stopgrad(p_teacher(·|x,prefix,feedback)))',
 toy:'A two-token vocabulary labels token 0 as correct for the example. Move the feedback teacher toward or away from it.',
 controls:[c('logit','Policy correct-token logit',-4,4,.1,0),c('feedback','Feedback-teacher correct-token logit',-4,4,.1,2)],failure:{logit:2,feedback:-4},
 failureText:'Bad feedback makes the self-teacher favor the wrong token; minimizing divergence faithfully copies that error. A weak self-teacher may not understand the feedback. On-policy prefixes reduce one train–test mismatch but do not make teacher corrections reliable.',
 fit:['Feedback-guided embedding distillation is conceivable, but not the sequence algorithm evaluated by this paper.','A scorer can consume richer labels/feedback; direct supervised or KD objectives are usually simpler.','A direct research fit when execution or judge feedback informs a generative policy.'],
 modern:'SDPO (January 2026) and DistiLLM-2 (2025) show that distillation includes the choice of trajectories, teacher context and response comparisons. “Latest” changes the learning setup as well as the scalar loss.',sources:[paper('Reinforcement Learning via Self-Distillation','2601.20802'),['TRL SDPO implementation','https://huggingface.co/docs/trl/main/sdpo_trainer']]
}
];
export const defaults=spec=>Object.fromEntries(spec.controls.map(c=>[c.key,c.value]));
