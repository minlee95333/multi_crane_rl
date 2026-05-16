
const W=100,H=100; let cranes=[], lifts=[], qTables=[], history=[], best=null, training=false, lastEval=null, rng=Math.random, replayTimer=null, replayTime=0, replayPlaying=false, mappoModel=null, lastPlanningResult=null, lastPlanningCompare=[], dragItem=null;
const $=id=>document.getElementById(id); const val=id=>parseFloat($(id).value);
function cfg(){return {nC:+$('numCranes').value,nL:+$('numLifts').value,craneType:$('craneType').value,fixedDuration:val('fixedDuration'),setupTime:val('setupTime'),craneRadius:val('craneRadius'),episodes:+$('episodes').value,maxSteps:+$('maxSteps').value,alpha:val('alpha'),gamma:val('gamma'),epsStart:val('epsStart'),epsEnd:val('epsEnd'),seedRuns:+$('seedRuns').value,baseSeed:+$('baseSeed').value,learningMode:$('learningMode').value,learningAlgorithm:$('learningAlgorithm')?$('learningAlgorithm').value:'q',actorShareMode:$('actorShareMode')?$('actorShareMode').value:'typeShared',actorLr:$('actorLr')?val('actorLr'):0.012,criticLr:$('criticLr')?val('criticLr'):0.02,gaeLambda:$('gaeLambda')?val('gaeLambda'):0.95,ppoClip:$('ppoClip')?val('ppoClip'):0.2,candidateK:+$('candidateK').value,liftingRadius:val('liftingRadius'),safetyRadius:Math.max(val('safetyRadius'),val('liftingRadius')),rSingle:val('rSingle'),rAll:val('rAll'),rSame:val('rSame'),pIdle:val('pIdle'),pInterSoft:val('pInterSoft'),pInterHard:val('pInterHard'),pInter:val('pInterHard'),pTime:val('pTime'),pMove:val('pMove')}}

function pad2(n){return String(n).padStart(2,'0');}
function modelFileBaseName(){
  const d=new Date(), c=cfg();
  return `${d.getFullYear()}${pad2(d.getMonth()+1)}${pad2(d.getDate())}_${pad2(d.getHours())}${pad2(d.getMinutes())}_${c.episodes}episode_학습`;
}
function qTableStats(){
  return qTables.map((qt,i)=>({agent:`C${i+1}`,states:Object.keys(qt||{}).length,actions:Object.values(qt||{}).reduce((n,row)=>n+Object.keys(row||{}).length,0)}));
}
function compactEval(r){
  if(!r)return null;
  return {done:r.done,total:r.total,makespan:r.makespan,reward:r.reward,softInter:r.softInter||0,hardExecuted:r.hardExecuted||0,hardMask:r.hardMask??r.hardInter??0,hardInter:r.hardInter??r.hardMask??0,interPenaltyTotal:r.interPenaltyTotal||0,same:r.same,setupTotal:r.setupTotal,travelTotal:r.travelTotal,liftTotal:r.liftTotal,moveTotal:r.moveTotal,craneFinish:(r.cranes||[]).map(c=>({id:c.id,done:c.done||0,finish:Math.round(c.availableTime||0),reward:c.reward||0,moveDistance:c.moveDistance||0}))};
}
function modelPayload(){
  const c=cfg(), modelName=modelFileBaseName();
  return {
    modelName,
    version:'stage1-candidate-k-tabular-qlearning-hardmask-v1',
    createdAt:new Date().toISOString(),
    algorithm:{
      family:'Reinforcement Learning',
      method:c.learningAlgorithm==='mappo'?'MAPPO type-shared actor, browser-light':'Independent tabular Q-learning',
      actionSpace:'Candidate-K lift slot + idle',
      agents:'one crane = one independent Q-learning agent',
      hardConstraints:['completed lift excluded','same lift duplicate selection masked','hard lifting-radius overlap masked'],
      softRewards:['single lift completion','team all-complete bonus','same setup/radius bonus','task time penalty','move-distance penalty','soft safety-radius interference penalty','idle penalty']
    },
    config:c,
    training:{episodes:c.episodes,learningMode:c.learningMode,baseSeed:c.baseSeed,seedRuns:c.seedRuns,alpha:c.alpha,gamma:c.gamma,epsStart:c.epsStart,epsEnd:c.epsEnd,historyLength:history.length},
    qTables:cloneQTables(),
    qTableStats:qTableStats(),
    mappoModel:mappoModel?cloneMappoModel():null,
    scenario:{cranes:cranes.map(cr=>({id:cr.id,type:cr.type,capacityTon:cr.capacityTon,home:cr.home,x:cr.x,y:cr.y,setupX:cr.setupX,setupY:cr.setupY})),lifts:lifts.map(l=>({id:l.id,x:l.x,y:l.y,duration:l.duration,zone:l.zone,status:l.status,by:l.by,start:l.start,finish:l.finish}))},
    best:compactEval(best),
    lastEval:compactEval(lastEval),
    recentHistory:history.slice(-20).map(compactEval),
    notes:'파일명 규칙: 날짜_시간_학습에피소드수. 이 JSON은 브라우저 프로토타입의 학습된 Q-table과 실험 설정을 재현하기 위한 export입니다.'
  };
}
function saveModelJson(){
  if(cfg().learningAlgorithm!=='mappo' && (!qTables.length || qTables.every(q=>!q||Object.keys(q).length===0))){log('저장할 학습 Q-table이 없습니다. 먼저 학습을 실행하세요.'); return;}
  if(cfg().learningAlgorithm==='mappo' && !mappoModel){log('저장할 MAPPO 모델이 없습니다. 먼저 학습을 실행하세요.'); return;}
  const payload=modelPayload(), fileName=`${payload.modelName}.json`, text=JSON.stringify(payload,null,2);
  if(typeof Blob!=='undefined' && document.createElement){
    const blob=new Blob([text],{type:'application/json'}), url=URL.createObjectURL(blob), a=document.createElement('a');
    a.href=url; a.download=fileName; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  }
  log(`모델 저장 JSON 생성: ${fileName} · ${payload.mappoModel?'MAPPO shared actor updates '+payload.mappoModel.stats.updates:'Q states '+payload.qTableStats.map(x=>`${x.agent}:${x.states}`).join(', ')}`);
  return {fileName,payload};
}

function resetRewards(){ $('rSingle').value=10;$('rAll').value=100;$('rSame').value=3;$('pIdle').value=-0.5;$('pInterSoft').value=-3;$('pInterHard').value=-15;$('liftingRadius').value=8;$('safetyRadius').value=14;$('pTime').value=-0.1;$('pMove').value=-0.02; log('reward 계수를 추천 초기값으로 복원했습니다. hard 간섭은 penalty가 아니라 action mask로 처리됩니다.');}
function makeRng(seed){let s=(seed>>>0)||1; return function(){s=(1664525*s+1013904223)>>>0; return s/4294967296;}}
function setSeed(seed){rng=seed?makeRng(seed):Math.random;}
function rand(a,b){return a+rng()*(b-a)} function dist(a,b){return Math.hypot(a.x-b.x,a.y-b.y)}
function initQTables(){const c=cfg(); qTables=[]; for(let i=0;i<c.nC;i++)qTables.push({});}

function initMappoModel(){
  // Browser-light MAPPO-style model: one actor shared by the crane type and one centralized critic.
  // It intentionally keeps Q-learning tables separate so the old model remains a baseline.
  const c=cfg(), actorDim=21, criticDim=14;
  const randW=()=> (rng()-.5)*0.02;
  mappoModel={
    version:'browser-light-mappo-type-shared-v3-actual-radius-margin-features',
    typeActors:{'50톤 모바일 크레인':{w:Array.from({length:actorDim},randW)}},
    critic:{w:Array.from({length:criticDim},randW)},
    stats:{updates:0,episodes:0,actorShareMode:c.actorShareMode,craneType:c.craneType,actorDim,criticDim}
  };
}
function ensureMappoModel(){
  if(!mappoModel)initMappoModel();
  const actorDim=21, criticDim=14, randW=()=> (rng()-.5)*0.02;
  Object.values(mappoModel.typeActors||{}).forEach(a=>{while(a.w.length<actorDim)a.w.push(randW()); if(a.w.length>actorDim)a.w=a.w.slice(0,actorDim);});
  if(mappoModel.critic){while(mappoModel.critic.w.length<criticDim)mappoModel.critic.w.push(randW()); if(mappoModel.critic.w.length>criticDim)mappoModel.critic.w=mappoModel.critic.w.slice(0,criticDim);}
  mappoModel.actorDim=actorDim; mappoModel.criticDim=criticDim; if(mappoModel.stats){mappoModel.stats.actorDim=actorDim; mappoModel.stats.criticDim=criticDim;}
  return mappoModel;
}
function cloneMappoModel(){return JSON.parse(JSON.stringify(ensureMappoModel()));}
function restoreMappoModel(m){mappoModel=JSON.parse(JSON.stringify(m));}
function dot(a,b){let s=0; for(let i=0;i<Math.min(a.length,b.length);i++)s+=a[i]*b[i]; return s;}
function softmax(scores){const m=Math.max(...scores), ex=scores.map(x=>Math.exp(Math.max(-40,Math.min(40,x-m)))), z=ex.reduce((a,b)=>a+b,0)||1; return ex.map(x=>x/z);}
function setupTargetForLift(sp,lift,c){
  if(!lift) return {x:sp.x,y:sp.y};
  const d=dist(sp,lift);
  if(d<=c.craneRadius || d<=1e-9) return {x:sp.x,y:sp.y};
  const ux=(lift.x-sp.x)/d, uy=(lift.y-sp.y)/d;
  // Move the crane/setup point only far enough that the selected lift is reachable.
  // The lift remains separate from the crane; actual lifting radius is target→lift.
  return {x:lift.x-ux*c.craneRadius, y:lift.y-uy*c.craneRadius};
}
function actualLiftCircleFor(cr,lift){
  if(!cr || !lift) return null;
  const c=cfg(), sp=setupPoint(cr), target=setupTargetForLift(sp,lift,c);
  return {x:target.x,y:target.y,r:dist(target,lift),liftX:lift.x,liftY:lift.y};
}
function actualCircleRelation(ca,cb,c=cfg()){
  if(!ca||!cb) return {level:'none',margin:999};
  const d=dist(ca,cb), hardLimit=ca.r+cb.r, softBuffer=Math.max(0,(c.safetyRadius||0)-(c.liftingRadius||0)), margin=d-hardLimit;
  if(margin<0) return {level:'hard',margin};
  if(margin<softBuffer) return {level:'soft',margin};
  return {level:'none',margin};
}
function normMargin(v,scale=30){return Math.max(-1,Math.min(1,(Number.isFinite(v)?v:scale)/scale));}
function nearestLiftForCrane(cr,liftsRun,excludeLift=null){
  const remaining=(liftsRun||[]).filter(l=>l.status==='todo' && l!==excludeLift && (!excludeLift || l.id!==excludeLift.id));
  if(!remaining.length) return null;
  const sp=setupPoint(cr);
  return remaining.reduce((best,l)=>dist(sp,l)<dist(sp,best)?l:best,remaining[0]);
}
function candidateCircleRisk(cr,lift,cranesRun=[],liftsRun=[]){
  const c=cfg(), own=actualLiftCircleFor(cr,lift), softBuffer=Math.max(0,(c.safetyRadius||0)-(c.liftingRadius||0)); let hardRisk=0, softRisk=0, minMargin=999, minDangerMargin=999, nearestCircleDist=999;
  if(!own) return {hardRisk,softRisk,minMargin,minDangerMargin,nearestCircleDist};
  for(const other of cranesRun.filter(o=>o.id!==cr.id)){
    const otherLift=nearestLiftForCrane(other,liftsRun,lift); if(!otherLift) continue;
    const otherCircle=actualLiftCircleFor(other,otherLift); if(!otherCircle) continue;
    const rel=actualCircleRelation(own,otherCircle,c), centerD=dist(own,otherCircle);
    nearestCircleDist=Math.min(nearestCircleDist,centerD);
    minMargin=Math.min(minMargin,rel.margin);
    minDangerMargin=Math.min(minDangerMargin,rel.margin-softBuffer);
    if(rel.level==='hard') hardRisk=1; else if(rel.level==='soft') softRisk=1;
  }
  return {hardRisk,softRisk,minMargin,minDangerMargin,nearestCircleDist};
}
function candidateOutcome(cr,lift,cranesRun=[],liftsRun=[]){
  const c=cfg(), sp=setupPoint(cr), same=lift?inCraneRadius(cr,lift):false, target=lift?setupTargetForLift(sp,lift,c):sp, moveDist=lift?(same?0:dist(sp,target)):0, travel=moveDist/8, setup=lift?(same?0:c.setupTime):0;
  const duration=lift?(lift.duration||c.fixedDuration):0, taskTime=travel+setup+duration, finish=(cr.availableTime||0)+taskTime;
  const risk=lift?candidateCircleRisk(cr,lift,cranesRun,liftsRun):{softRisk:0,hardRisk:0,minMargin:999,minDangerMargin:999,nearestCircleDist:999};
  return {same,moveDist,travel,setup,taskTime,finish,softRisk:risk.softRisk,hardRisk:risk.hardRisk,minCircleMargin:risk.minMargin,minDangerMargin:risk.minDangerMargin,nearestCircleDist:risk.nearestCircleDist,actualLiftRadius:lift?dist(target,lift):0,targetX:target.x,targetY:target.y};
}
function globalFeatures(cranesRun,liftsRun){
  const c=cfg(), remaining=liftsRun.filter(l=>l.status==='todo'), rem=remaining.length, times=cranesRun.map(cr=>cr.availableTime||0), maxT=Math.max(0,...times), minT=Math.min(...times), meanT=times.reduce((a,b)=>a+b,0)/Math.max(1,times.length);
  const done=liftsRun.length-rem, spread=maxT-minT;
  let nearestSum=0, radiusHits=0, softPairs=0, hardPairs=0, marginSum=0, dangerMarginSum=0, marginN=0;
  for(const cr of cranesRun){
    const sp=setupPoint(cr);
    if(remaining.length){nearestSum += Math.min(...remaining.map(l=>dist(sp,l)));}
    radiusHits += remaining.filter(l=>inCraneRadius(cr,l)).length;
  }
  for(let i=0;i<cranesRun.length;i++) for(let j=i+1;j<cranesRun.length;j++){
    const li=nearestLiftForCrane(cranesRun[i],remaining), lj=nearestLiftForCrane(cranesRun[j],remaining,li);
    const rel=actualCircleRelation(actualLiftCircleFor(cranesRun[i],li),actualLiftCircleFor(cranesRun[j],lj),c), softBuffer=Math.max(0,(c.safetyRadius||0)-(c.liftingRadius||0));
    if(rel.level==='hard') hardPairs++; else if(rel.level==='soft') softPairs++;
    marginSum+=rel.margin; dangerMarginSum+=rel.margin-softBuffer; marginN++;
  }
  const craneCount=Math.max(1,cranesRun.length), pairCount=Math.max(1,cranesRun.length*(cranesRun.length-1)/2);
  return [1, rem/Math.max(1,c.nL), done/Math.max(1,c.nL), meanT/500, maxT/500, spread/500, cranesRun.length/10, c.candidateK/12, (nearestSum/craneCount)/100, radiusHits/Math.max(1,rem*craneCount), softPairs/pairCount, hardPairs/pairCount, normMargin(marginSum/Math.max(1,marginN),40), normMargin(dangerMarginSum/Math.max(1,marginN),40)];
}
function candidateFeatureVector(cr,liftsRun,cranesRun,ch){
  const c=cfg(), sp=setupPoint(cr), remaining=liftsRun.filter(l=>l.status==='todo'), rem=remaining.length/Math.max(1,c.nL), t=(cr.availableTime||0)/500;
  if(ch.liftIdx<0) return [1, sp.x/100, sp.y/100, t, rem, 1, 0, 0, 1, 0, 1, c.candidateK/12, 0, 0, 0, t, 0, 0, 1, 1, 0];
  const lift=liftsRun[ch.liftIdx], d=dist(sp,lift), inR=inCraneRadius(cr,lift)?1:0;
  const load=Number.isFinite(ch.radiusLoad)?ch.radiusLoad:remaining.filter(l=>dist(lift,l)<=c.craneRadius).length, out=ch.outcome || candidateOutcome(cr,lift,cranesRun,liftsRun);
  const hardMarginNorm=normMargin(out.minCircleMargin,40), dangerMarginNorm=normMargin(out.minDangerMargin,40), feasible=out.hardRisk?0:1;
  return [1, sp.x/100, sp.y/100, t, rem, d/100, inR, out.setup>0?1:0, Math.min(1,(out.nearestCircleDist||100)/100), load/Math.max(1,c.nL), out.actualLiftRadius/Math.max(1,c.craneRadius), c.candidateK/12, out.travel/60, out.setup/60, out.taskTime/120, out.finish/500, out.softRisk, out.hardRisk, hardMarginNorm, dangerMarginNorm, feasible];
}
function actorForCrane(cr){const m=ensureMappoModel(); return m.typeActors[cr.type] || m.typeActors['50톤 모바일 크레인'];}
function mappoDistribution(agentIdx,cr,liftsRun,cranesRun,reserved=new Set()){
  const actor=actorForCrane(cr); let cands=candidateActions(cr,liftsRun,cranesRun).filter(c=>!reserved.has(c.idx));
  if(!cands.length){
    cands=liftsRun.map((l,idx)=>l.status==='todo'&&!reserved.has(idx)?{l,idx}:null).filter(Boolean)
      .map((item,slot)=>({...item,slot,key:stateKey(cr,liftsRun,cranesRun,item.l)}));
  }
  let choices=cands.map(c=>({i:agentIdx,key:c.key,act:c.slot,liftIdx:c.idx,slot:c.slot,outcome:c.outcome,radiusLoad:c.radiusLoad,circleRisk:c.circleRisk}));
  if(!choices.length) choices.push({i:agentIdx,key:stateKey(cr,liftsRun,cranesRun,'idle'),act:-1,liftIdx:-1,slot:-1});
  choices.forEach(ch=>{ch.features=candidateFeatureVector(cr,liftsRun,cranesRun,ch); ch.score=dot(actor.w,ch.features);});
  const probs=softmax(choices.map(ch=>ch.score)); choices.forEach((ch,i)=>{ch.prob=probs[i]; ch.logp=Math.log(Math.max(1e-9,probs[i]));});
  const expectedFeatures=Array(actor.w.length).fill(0); choices.forEach((ch,k)=>ch.features.forEach((f,j)=>expectedFeatures[j]+=probs[k]*f));
  choices.forEach(ch=>{ch.expectedFeatures=[...expectedFeatures];});
  return choices;
}
function chooseMappoAction(agentIdx,cr,liftsRun,epsilon=0,cranesRun=[]){
  const choices=mappoDistribution(agentIdx,cr,liftsRun,cranesRun), workChoices=choices.filter(ch=>ch.liftIdx>=0);
  const pool=workChoices.length?workChoices:choices;
  let chosen;
  if(rng()<epsilon) chosen=pool[Math.floor(rng()*pool.length)];
  else if(epsilon<=0) chosen=pool.reduce((best,ch)=>(ch.prob||0)>(best.prob||0)?ch:best,pool[0]);
  else {
    const z=pool.reduce((s,ch)=>s+(ch.prob||0),0)||1;
    let u=rng(), acc=0; for(const ch of pool){acc+=(ch.prob||0)/z; if(u<=acc){chosen=ch; break;}}
    if(!chosen) chosen=pool[pool.length-1];
  }
  return {...chosen, alternatives:choices};
}
function learnMappoStep(cr,choice,reward,gf,ngf){
  const c=cfg(), m=ensureMappoModel(), actor=actorForCrane(cr);
  const v=dot(m.critic.w,gf), nv=dot(m.critic.w,ngf), adv=reward + c.gamma*nv - v;
  const expected=choice.expectedFeatures || Array(actor.w.length).fill(0);
  const feat=choice.features || expected;
  const ratio=1, clipped=Math.max(1-c.ppoClip,Math.min(1+c.ppoClip,ratio));
  const scale=c.actorLr*Math.min(ratio,clipped)*adv;
  for(let j=0;j<actor.w.length;j++) actor.w[j]+=scale*(feat[j]-expected[j]);
  const td=reward + c.gamma*nv - v; for(let j=0;j<m.critic.w.length;j++) m.critic.w[j]+=c.criticLr*td*gf[j];
  m.stats.updates++;
}
function cloneQTables(){return JSON.parse(JSON.stringify(qTables));}
function restoreQTables(q){qTables=JSON.parse(JSON.stringify(q));}
function generateScenario(seed=null, silent=false, preserveLearning=false){ const c=cfg(); setSeed(seed); cranes=[]; lifts=[]; if(!preserveLearning){initQTables(); history=[]; best=null; lastEval=null;}
  for(let i=0;i<c.nC;i++){cranes.push({id:'C'+(i+1),type:c.craneType,capacityTon:50,x:rand(8,92),y:rand(8,92),home:null,lastZone:null,setupX:null,setupY:null,availableTime:0,done:0,idle:0,reward:0,moveDistance:0,schedule:[]}); cranes[i].home={x:cranes[i].x,y:cranes[i].y}; cranes[i].setupX=cranes[i].x; cranes[i].setupY=cranes[i].y;}
  for(let j=0;j<c.nL;j++){lifts.push({id:'L'+(j+1),x:rand(8,92),y:rand(8,92),duration:c.fixedDuration,zone:0,status:'todo',by:null,finish:null}); lifts[j].zone=Math.floor(lifts[j].x/25)+'-'+Math.floor(lifts[j].y/25);}
  if(!silent){log(`시나리오 생성: 크레인 ${c.nC}대, 타입 ${c.craneType}, 양중물 ${c.nL}개, 양중 시간 ${c.fixedDuration}분 고정, 준비 시간 ${c.setupTime}분, 단일 작업 반경 ${c.craneRadius}m${seed?`, seed ${seed}`:''}`); updateAll(0); drawSim();}}
function cloneScenario(){return {cranes:cranes.map(c=>({...c,home:{...c.home},schedule:[]})), lifts:lifts.map(l=>({...l}))}}
function bucket(v,a,b){return v<=a?'near':v<=b?'mid':'far'}
function countBucket(n){return n===0?'none':n===1?'one':n<=3?'few':'many'}
function timeBucket(t){return t<150?'early':t<300?'mid':'late'}
function setupPoint(cr){return {x:Number.isFinite(cr.setupX)?cr.setupX:cr.x,y:Number.isFinite(cr.setupY)?cr.setupY:cr.y};}
function inCraneRadius(cr,lift){const c=cfg(); return dist(setupPoint(cr),lift)<=c.craneRadius;}
function setupMoveNeeded(cr,lift){return !inCraneRadius(cr,lift);}
function stateKey(cr, liftsRun, cranesRun=[], candidate=null, precomputed=null){
  const c=cfg(); const remaining=liftsRun.filter(l=>l.status==='todo'); if(!remaining.length)return 'done';
  const sp=setupPoint(cr);
  let nearest=remaining.reduce((a,l)=>dist(sp,l)<dist(sp,a)?l:a,remaining[0]);
  let radiusCount=remaining.filter(l=>dist(sp,l)<=c.craneRadius).length;
  let nearestDist=dist(sp,nearest);
  let otherCranes=cranesRun.filter(o=>o.id!==cr.id);
  let nearestCraneDist=otherCranes.length?Math.min(...otherCranes.map(o=>dist(sp,setupPoint(o)))):999;
  let remainingBucket=remaining.length>10?'many':remaining.length>4?'mid':'few';
  let base=`${Math.floor(sp.x/25)}-${Math.floor(sp.y/25)}|radius:${countBucket(radiusCount)}|near:${bucket(nearestDist,15,35)}|rem:${remainingBucket}|crane:${bucket(nearestCraneDist,15,35)}|time:${timeBucket(cr.availableTime||0)}`;
  if(!candidate)return base;
  if(candidate==='idle')return base+'|cand:idle';
  let cDist=dist(sp,candidate), cInRadius=cDist<=c.craneRadius, cSetup=cInRadius?'no':'yes';
  let circleRisk=precomputed?.circleRisk || candidateCircleRisk(cr,candidate,cranesRun,liftsRun), otherRisk=circleRisk.minMargin;
  let radiusLoad=Number.isFinite(precomputed?.radiusLoad)?precomputed.radiusLoad:remaining.filter(l=>dist(candidate,l)<=c.craneRadius).length;
  return `${base}|candRadius:${cInRadius?'in':'out'}|candDist:${bucket(cDist,15,35)}|setup:${cSetup}|candCircleRisk:${circleRisk.hardRisk?'hard':circleRisk.softRisk?'soft':bucket(otherRisk,6,18)}|candDanger:${bucket(circleRisk.minDangerMargin,0,12)}|candRadiusLoad:${countBucket(radiusLoad)}`;
}
function uniquePush(arr,item){if(item && !arr.some(x=>x.idx===item.idx))arr.push(item);}
function candidateActions(cr,liftsRun,cranesRun=[],K=null){const c=cfg(); K=K||Math.max(2,Math.min(12,c.candidateK||5)); const remaining=liftsRun.map((l,idx)=>l.status==='todo'?{l,idx}:null).filter(Boolean); if(!remaining.length)return [];
  let out=[]; const sp=setupPoint(cr), outcomeCache=new Map(), riskCache=new Map(), loadCache=new Map();
  const nearestOf=pool=>pool.reduce((best,item)=>dist(sp,item.l)<dist(sp,best.l)?item:best,pool[0]);
  const outcome=item=>{if(!outcomeCache.has(item.idx)) outcomeCache.set(item.idx,candidateOutcome(cr,item.l,cranesRun,liftsRun)); return outcomeCache.get(item.idx);};
  const riskOf=item=>{if(!riskCache.has(item.idx)) riskCache.set(item.idx,candidateCircleRisk(cr,item.l,cranesRun,liftsRun)); return riskCache.get(item.idx);};
  const localLoad=item=>{if(!loadCache.has(item.idx)) loadCache.set(item.idx,remaining.filter(x=>dist(item.l,x.l)<=c.craneRadius).length); return loadCache.get(item.idx);};
  const circleSeparation=item=>riskOf(item).minMargin;
  const dangerSeparation=item=>riskOf(item).minDangerMargin;
  const outcomeScore=item=>{
    const o=outcome(item);
    // Lower is better. Prioritize predicted finish/task time and movement efficiency;
    // soft exposure is only a small shaping term because hard lifting-radius overlap is masked elsewhere.
    const hardPressure=Math.max(0,-o.minCircleMargin), dangerPressure=Math.max(0,-o.minDangerMargin);
    return o.finish + 0.35*o.taskTime + 0.18*o.moveDist + 2*(o.setup>0?1:0) + 12*o.hardRisk + 1.2*o.softRisk + 0.35*hardPressure + 0.08*dangerPressure - 0.4*localLoad(item);
  };
  const bestBy=fn=>remaining.reduce((best,item)=>fn(item)<fn(best)?item:best,remaining[0]);
  const inRadius=remaining.filter(item=>inCraneRadius(cr,item.l));
  // Candidate K now mixes operationally meaningful options instead of nearest-only fallbacks.
  uniquePush(out, inRadius.length?nearestOf(inRadius):null);                    // same setup/radius continuity
  uniquePush(out, nearestOf(remaining));                                        // global nearest baseline option
  uniquePush(out, bestBy(item=>outcome(item).finish));                          // earliest predicted finish
  uniquePush(out, bestBy(item=>outcome(item).taskTime));                        // shortest travel+setup+lift time
  uniquePush(out, bestBy(item=>outcome(item).moveDist));                        // low movement distance
  uniquePush(out, bestBy(item=>(outcome(item).setup>0?1:0)*1000 + dist(sp,item.l))); // setup-saving option
  uniquePush(out, remaining.reduce((best,item)=>circleSeparation(item)>circleSeparation(best)?item:best,remaining[0])); // hard-margin-safe option
  uniquePush(out, remaining.reduce((best,item)=>dangerSeparation(item)>dangerSeparation(best)?item:best,remaining[0])); // danger-margin-safe option
  const sorted=[...remaining].sort((a,b)=>outcomeScore(a)-outcomeScore(b));
  for(const item of sorted){uniquePush(out,item); if(out.length>=K)break;}
  return out.slice(0,K).map((item,slot)=>{const o=outcome(item), r=riskOf(item), load=localLoad(item); return {...item,slot,outcome:o,circleRisk:r,radiusLoad:load,key:stateKey(cr,liftsRun,cranesRun,item.l,{circleRisk:r,radiusLoad:load})};});}
function rlActionChoices(agentIdx, cr, liftsRun, eps, cranesRun=[]){let cands=candidateActions(cr,liftsRun,cranesRun); let choices=cands.map(c=>({slot:c.slot,liftIdx:c.idx,key:c.key})); choices.push({slot:-1,liftIdx:-1,key:stateKey(cr,liftsRun,cranesRun,'idle')}); if(rng()<eps)return choices[Math.floor(rng()*choices.length)]; return choices.reduce((best,ch)=>((qTables[agentIdx][ch.key]||{}).select??0)>((qTables[agentIdx][best.key]||{}).select??0)?ch:best,choices[0]);}
function nextMaxQ(agentIdx, cr, liftsRun, cranesRun=[]){let choices=candidateActions(cr,liftsRun,cranesRun).map(c=>c.key); choices.push(stateKey(cr,liftsRun,cranesRun,'idle')); if(!choices.length)return 0; return Math.max(...choices.map(k=>(qTables[agentIdx][k]||{}).select??0));}
function chooseBaselineAction(policy, cr, liftsRun){
  const remaining=liftsRun.map((l,idx)=>l.status==='todo'?{l,idx}:null).filter(Boolean);
  if(!remaining.length) return -1;
  const sp=setupPoint(cr);
  if(policy==='random') return remaining[Math.floor(rng()*remaining.length)].idx;
  if(policy==='nearest') return remaining.reduce((best,item)=>dist(sp,item.l)<dist(sp,best.l)?item:best,remaining[0]).idx;
  if(policy==='radiusPriority'){
    const inRadius=remaining.filter(item=>inCraneRadius(cr,item.l));
    const pool=inRadius.length?inRadius:remaining;
    return pool.reduce((best,item)=>dist(sp,item.l)<dist(sp,best.l)?item:best,pool[0]).idx;
  }
  return remaining.reduce((best,item)=>dist(sp,item.l)<dist(sp,best.l)?item:best,remaining[0]).idx;
}

function makePlan(ch,cs,ls){
  let cr=cs[ch.i], lift=ls[ch.liftIdx];
  if(ch.liftIdx<0 || !lift || lift.status!=='todo') return {...ch,valid:false,features:ch.features,expectedFeatures:ch.expectedFeatures,prob:ch.prob,logp:ch.logp};
  let c=cfg(), sp=setupPoint(cr), sameSetup=inCraneRadius(cr,lift), target=setupTargetForLift(sp,lift,c), moveDist=sameSetup?0:dist(sp,target), travel=moveDist/8, setup=sameSetup?0:c.setupTime, start=cr.availableTime, finish=start+travel+setup+lift.duration;
  const toX=target.x, toY=target.y;
  const radiusCenterX=toX, radiusCenterY=toY;
  return {...ch,valid:true,lift,features:ch.features,expectedFeatures:ch.expectedFeatures,prob:ch.prob,logp:ch.logp,travel,moveDist,same:sameSetup,setup,start,finish,taskTime:travel+setup+lift.duration,fromX:sp.x,fromY:sp.y,toX,toY,radiusCenterX,radiusCenterY,liftX:lift.x,liftY:lift.y,liftId:lift.id,actualLiftRadius:dist({x:radiusCenterX,y:radiusCenterY},lift)};
}
function liftCircleOf(x){
  if(!x) return null;
  const cx=Number.isFinite(x.radiusCenterX)?x.radiusCenterX:x.toX, cy=Number.isFinite(x.radiusCenterY)?x.radiusCenterY:x.toY;
  if(!Number.isFinite(cx) || !Number.isFinite(cy)) return null;
  const lx=Number.isFinite(x.liftX)?x.liftX:(x.lift&&x.lift.x), ly=Number.isFinite(x.liftY)?x.liftY:(x.lift&&x.lift.y);
  if(!Number.isFinite(lx) || !Number.isFinite(ly)) return null;
  const r=Number.isFinite(x.actualLiftRadius)?x.actualLiftRadius:dist({x:cx,y:cy},{x:lx,y:ly});
  return {x:cx,y:cy,r};
}
function liftingCircleRelation(a,b,c){
  return actualCircleRelation(liftCircleOf(a),liftCircleOf(b),c).level;
}
function hasHardOverlap(plan,accepted,c){
  if(!plan.valid) return false;
  return accepted.some(p=>{
    if(!overlap(plan.start,plan.finish,p.start,p.finish)) return false;
    return liftingCircleRelation(plan,p,c)==='hard';
  });
}
function rankedChoicesForAgent(policy, agentIdx, cr, liftsRun, cranesRun, reserved){
  const available=liftsRun.map((l,idx)=>l.status==='todo'&&!reserved.has(idx)?{l,idx}:null).filter(Boolean);
  if(!available.length) return [{i:agentIdx,key:stateKey(cr,liftsRun,cranesRun,'idle'),act:-1,liftIdx:-1}];
  const sp=setupPoint(cr);
  if(policy==='mappo'){
    return mappoDistribution(agentIdx,cr,liftsRun,cranesRun,reserved).sort((a,b)=>(b.prob||0)-(a.prob||0));
  }
  if(policy==='rl'){
    let cands=candidateActions(cr,liftsRun,cranesRun).filter(c=>!reserved.has(c.idx));
    if(!cands.length) cands=available.map((item,slot)=>({...item,slot,key:stateKey(cr,liftsRun,cranesRun,item.l)}));
    cands.sort((a,b)=>((qTables[agentIdx][b.key]||{}).select??0)-((qTables[agentIdx][a.key]||{}).select??0));
    return cands.map(c=>({i:agentIdx,key:c.key,act:c.slot,liftIdx:c.idx}));
  }
  let ranked=[...available];
  if(policy==='random') ranked.sort(()=>rng()-0.5);
  else if(policy==='radiusPriority') ranked.sort((a,b)=>{
    const ar=inCraneRadius(cr,a.l)?0:1, br=inCraneRadius(cr,b.l)?0:1;
    return ar-br || dist(sp,a.l)-dist(sp,b.l);
  });
  else ranked.sort((a,b)=>dist(sp,a.l)-dist(sp,b.l));
  return ranked.map(item=>({i:agentIdx,key:stateKey(cr,liftsRun,cranesRun,item.l),act:item.idx,liftIdx:item.idx}));
}
function enforceMaskedChoices(choices, policy, cranesRun, liftsRun, priorEvents=[]){
  const c=cfg(), reserved=new Set(), accepted=[...priorEvents]; let hardMasked=0, duplicateMasked=0;
  const finalChoices=[];
  for(const ch of choices){
    let candidates=[];
    if(ch.liftIdx>=0 && liftsRun[ch.liftIdx] && liftsRun[ch.liftIdx].status==='todo') candidates.push(ch);
    if(ch.liftIdx>=0 && reserved.has(ch.liftIdx)) duplicateMasked++;
    if(policy==='mappo' && Array.isArray(ch.alternatives)){
      candidates.push(...ch.alternatives.filter(a=>a.liftIdx>=0 && liftsRun[a.liftIdx] && liftsRun[a.liftIdx].status==='todo').sort((a,b)=>(b.prob||0)-(a.prob||0)));
    } else {
      candidates.push(...rankedChoicesForAgent(policy,ch.i,cranesRun[ch.i],liftsRun,cranesRun,reserved));
    }
    let chosen=null, chosenPlan=null, sawHard=false;
    for(const cand of candidates){
      if(cand.liftIdx>=0 && reserved.has(cand.liftIdx)) continue;
      const plan=makePlan(cand,cranesRun,liftsRun);
      if(hasHardOverlap(plan,accepted,c)){sawHard=true; continue;}
      chosen=cand; chosenPlan=plan; break;
    }
    if(!chosen){chosen={i:ch.i,key:stateKey(cranesRun[ch.i],liftsRun,cranesRun,'idle'),act:-1,liftIdx:-1}; chosenPlan=makePlan(chosen,cranesRun,liftsRun);}
    if(sawHard) hardMasked++;
    if(chosen.liftIdx>=0) reserved.add(chosen.liftIdx);
    accepted.push(chosenPlan); finalChoices.push(chosen);
  }
  return {choices:finalChoices,hardMasked,duplicateMasked};
}
function overlap(aStart,aEnd,bStart,bEnd){return Math.max(aStart,bStart)<Math.min(aEnd,bEnd)}
function classifyInterference(plan,plans,picked,c){
  if(!plan.valid) return {level:'none',penalty:0};
  if(picked[plan.liftIdx]>1) return {level:'none',penalty:0,reason:'same-lift-masked'};
  let hard=false, soft=false;
  for(const p of plans){
    if(!p.valid || p.i===plan.i) continue;
    if(!overlap(plan.start,plan.finish,p.start,p.finish)) continue;
    const rel=liftingCircleRelation(plan,p,c);
    if(rel==='hard') hard=true;
    else if(rel==='soft') soft=true;
  }
  if(hard) return {level:'none',penalty:0,reason:'hard-overlap-masked'};
  if(soft) return {level:'soft',penalty:c.pInterSoft,reason:'safety-radius-overlap'};
  return {level:'none',penalty:0};
}
function runEpisode(ep=0, learn=true, epsilon=0.1, policy='rl'){const c=cfg(); let sc=cloneScenario(); let cs=sc.cranes, ls=sc.lifts; let totalReward=0, inter=0, softInter=0, hardInter=0, interPenaltyTotal=0, sameCount=0, setupTotal=0, travelTotal=0, liftTotal=0, moveTotal=0, events=[];
  cs.forEach(cr=>{cr.availableTime=cr.availableTime||0;});
  for(let step=0; step<c.maxSteps && ls.some(l=>l.status==='todo'); step++){
    const stepGlobalBefore=globalFeatures(cs,ls), mappoLearnItems=[];
    let choices=[];
    for(let i=0;i<cs.length;i++){let cr=cs[i]; if(policy==='rl'){let ch=rlActionChoices(i,cr,ls,learn?epsilon:0,cs); choices.push({i,key:ch.key,act:ch.slot,liftIdx:ch.liftIdx});}
      else if(policy==='mappo'){let ch=chooseMappoAction(i,cr,ls,learn?epsilon:0,cs); choices.push(ch);}
      else {let liftIdx=chooseBaselineAction(policy,cr,ls); choices.push({i,key:stateKey(cr,ls,cs),act:liftIdx,liftIdx});}}
    const masked=enforceMaskedChoices(choices,policy,cs,ls,events); choices=masked.choices; hardInter+=masked.hardMasked;
    let picked={}; choices.forEach(ch=>{ if(ch.liftIdx>=0){ picked[ch.liftIdx]=(picked[ch.liftIdx]||0)+1; }});
    let plans=choices.map(ch=>makePlan(ch,cs,ls));
    for(const plan of plans){let cr=cs[plan.i]; let reward=0, oldKey=plan.key, act=plan.act;
      if(!plan.valid){ if(act===-1 && ls.some(l=>l.status==='todo')){reward+=c.pIdle*5; cr.idle+=5; cr.availableTime+=5;} }
      else if(plan.lift.status==='todo'){
          let interInfo = classifyInterference(plan,plans,picked,c), conflict = interInfo.level!=='none';
          reward += c.rSingle + (plan.same?c.rSame:0) + c.pTime*plan.taskTime + c.pMove*plan.moveDist + interInfo.penalty;
          if(interInfo.level==='soft'){inter++; softInter++; interPenaltyTotal+=interInfo.penalty;} if(plan.same) sameCount++; setupTotal+=plan.setup; travelTotal+=plan.travel; liftTotal+=plan.lift.duration; moveTotal+=plan.moveDist;
          if(!plan.same){cr.setupX=plan.toX; cr.setupY=plan.toY;} cr.x=cr.setupX; cr.y=cr.setupY; cr.lastZone=plan.lift.zone; cr.availableTime=plan.finish; cr.done++; cr.moveDistance=(cr.moveDistance||0)+plan.moveDist; cr.reward+=reward;
          cr.schedule.push(`${plan.lift.id} | 시작 ${Math.round(plan.start)}분 → 완료 ${Math.round(plan.finish)}분 · 이동 ${plan.travel.toFixed(1)}분 · 거리 ${plan.moveDist.toFixed(1)} · 준비 ${plan.setup}분 · 양중 ${plan.lift.duration}분${plan.same?' · 동일반경 보너스':''}${interInfo.level==='soft'?' · 안전반경 간섭':''}${interInfo.level==='hard'?' · 양중반경 간섭(예외)':''}`);
          events.push({craneId:cr.id,craneIndex:plan.i,liftId:plan.liftId,start:plan.start,finish:plan.finish,travel:plan.travel,setup:plan.setup,duration:plan.lift.duration,moveDist:plan.moveDist,actualLiftRadius:plan.actualLiftRadius,radiusCenterX:plan.radiusCenterX,radiusCenterY:plan.radiusCenterY,fromX:plan.fromX,fromY:plan.fromY,toX:plan.toX,toY:plan.toY,liftX:plan.liftX,liftY:plan.liftY,sameRadius:plan.same,conflict,interferenceLevel:interInfo.level,interferencePenalty:interInfo.penalty,interferenceReason:interInfo.reason||''});
          plan.lift.status='done'; plan.lift.by=cr.id; plan.lift.start=Math.round(plan.start); plan.lift.finish=Math.round(plan.finish);
        }
      if(!ls.some(l=>l.status==='todo')) reward += c.rAll;
      totalReward += reward; cr.reward+= (!plan.valid && act===-1?reward:0);
      if(learn && policy==='rl'){let qt=qTables[plan.i][oldKey]||(qTables[plan.i][oldKey]={}); let maxNext=nextMaxQ(plan.i,cr,ls,cs); qt.select=(qt.select??0)+c.alpha*(reward+c.gamma*maxNext-(qt.select??0));}
      if(learn && policy==='mappo'){mappoLearnItems.push({cr,choice:{...plan,key:oldKey,act,liftIdx:plan.liftIdx,features:plan.features,expectedFeatures:plan.expectedFeatures,prob:plan.prob,logp:plan.logp},reward});}
    }
    if(learn && policy==='mappo'){
      const stepGlobalAfter=globalFeatures(cs,ls);
      mappoLearnItems.forEach(u=>learnMappoStep(u.cr,u.choice,u.reward,stepGlobalBefore,stepGlobalAfter));
    }
  }
  let done=ls.filter(l=>l.status==='done').length, makespan=Math.max(0,...cs.map(c=>c.availableTime||0),...ls.map(l=>l.finish||0)); return {reward:totalReward, done, total:ls.length, makespan:Math.round(makespan), inter, softInter, hardExecuted:0, hardMask:hardInter, hardInter, interPenaltyTotal, same:sameCount, setupTotal, travelTotal, liftTotal, moveTotal, cranes:cs, lifts:ls, events};}
async function startTraining(){ if(training) return; const c=cfg(); if(!cranes.length) generateScenario(c.baseSeed); training=true; $('trainBtn').disabled=true; history=[]; best=null; if(c.learningAlgorithm==='mappo'){initMappoModel();} else {initQTables();}
  log(c.learningMode==='multi'?`여러 seed 시나리오 학습 시작: seed ${c.baseSeed}~${c.baseSeed+Math.max(1,c.seedRuns)-1}를 episode마다 순환`:'단일 시나리오 학습 시작');
  for(let ep=1; ep<=c.episodes; ep++){
    let eps=c.epsStart+(c.epsEnd-c.epsStart)*(ep/c.episodes);
    if(c.learningMode==='multi'){
      const trainSeed=c.baseSeed+((ep-1)%Math.max(1,c.seedRuns));
      generateScenario(trainSeed,true,true);
      setSeed(trainSeed*1000+ep);
    }
    let r=runEpisode(ep,true,eps,c.learningAlgorithm==='mappo'?'mappo':'rl'); history.push(r); if(r.done===r.total && (!best || r.makespan<best.makespan)) best=r;
    if(ep%5===0||ep===1){updateAll(ep); drawCharts(); drawSim(best||r); await new Promise(res=>setTimeout(res,1));}
  }
  training=false; $('trainBtn').disabled=false; log(`학습 완료 (${c.learningAlgorithm==='mappo'?'MAPPO shared actor':'Q-learning'} · ${c.learningMode==='multi'?'multi-scenario':'single-scenario'}). best makespan=${best?best.makespan:'없음'}분`);
  if(c.learningMode==='multi') generateScenario(c.baseSeed,true,true);
  evaluatePolicy();}
function evaluatePolicy(){ if(!cranes.length) generateScenario(); const pol=cfg().learningAlgorithm==='mappo'?'mappo':'rl'; lastEval=runEpisode(0,false,0,pol); log(`정책 평가: 완료 ${lastEval.done}/${lastEval.total}, makespan ${lastEval.makespan}분, soft간섭 ${lastEval.softInter||0}회 / hard마스크 ${lastEval.hardInter||0}회, reward ${lastEval.reward.toFixed(1)}`); updateAll(history.length); drawSim(lastEval); renderSchedule(lastEval); renderMetricLog(lastEval,'RL 정책 평가'); renderBaselineRows([{name:pol==='mappo'?'MAPPO shared actor':'Q-learning policy',r:lastEval}]); replayTime=0; updateReplayControls(lastEval);}
function evaluateBaselines(){ if(!cranes.length) generateScenario(); const results=[
  {name:'Q-learning policy',r:runEpisode(0,false,0,'rl')},
  ...(mappoModel?[{name:'MAPPO type-shared actor',r:runEpisode(0,false,0,'mappo')}]:[]),
  {name:'Random',r:runEpisode(0,false,0,'random')},
  {name:'Nearest-lift-first',r:runEpisode(0,false,0,'nearest')},
  {name:'Same-radius-priority',r:runEpisode(0,false,0,'radiusPriority')}
]; lastEval=(results.find(x=>x.name.includes('MAPPO'))||results[0]).r; renderBaselineRows(results); updateAll(history.length); drawSim(lastEval); renderSchedule(lastEval); renderMetricLog(results,'Baseline 비교'); replayTime=0; updateReplayControls(lastEval); log('Baseline 비교 완료: Q-learning / MAPPO-shared / Random / Nearest / Same-radius 정책을 동일 시나리오에서 평가했습니다.');}
function trainCurrentScenario(seedOffset=0){const c=cfg(); history=[]; best=null; for(let ep=1; ep<=c.episodes; ep++){setSeed((c.baseSeed||1)+seedOffset+ep*17); let eps=c.epsStart+(c.epsEnd-c.epsStart)*(ep/c.episodes); let r=runEpisode(ep,true,eps,'rl'); history.push(r); if(r.done===r.total && (!best || r.makespan<best.makespan)) best=r;} return best||history.at(-1);}
function trainMultiScenarioModel(){const c=cfg(); history=[]; best=null; initQTables(); const span=Math.max(1,c.seedRuns||1); for(let ep=1; ep<=c.episodes; ep++){const trainSeed=(c.baseSeed||101)+((ep-1)%span); generateScenario(trainSeed,true,true); setSeed(trainSeed*1000+ep); let eps=c.epsStart+(c.epsEnd-c.epsStart)*(ep/c.episodes); let r=runEpisode(ep,true,eps,'rl'); history.push(r); if(r.done===r.total && (!best || r.makespan<best.makespan)) best=r;} return cloneQTables();}

function trainMappoMultiScenarioModel(){const c=cfg(); history=[]; best=null; initMappoModel(); const span=Math.max(1,c.seedRuns||1); for(let ep=1; ep<=c.episodes; ep++){const trainSeed=(c.baseSeed||101)+((ep-1)%span); generateScenario(trainSeed,true,true); setSeed(trainSeed*2000+ep); let eps=c.epsStart+(c.epsEnd-c.epsStart)*(ep/c.episodes); let r=runEpisode(ep,true,eps,'mappo'); history.push(r); if(r.done===r.total && (!best || r.makespan<best.makespan)) best=r;} return cloneMappoModel();}
function curriculumStages(){return [
  {name:'1단계',nC:2,nL:12,episodes:100,seedRuns:10,candidateK:5,maxSteps:140,baseSeed:101},
  {name:'2단계',nC:3,nL:24,episodes:150,seedRuns:10,candidateK:5,maxSteps:220,baseSeed:101},
  {name:'3단계',nC:4,nL:36,episodes:220,seedRuns:15,candidateK:6,maxSteps:320,baseSeed:301}
];}
function snapshotControls(ids=['numCranes','numLifts','episodes','seedRuns','candidateK','maxSteps','baseSeed','learningMode','learningAlgorithm']){return Object.fromEntries(ids.map(id=>[id,$(id).value]));}
function restoreControls(snap){Object.entries(snap).forEach(([id,v])=>{if($(id))$(id).value=v;});}
function applyStageConfig(st){$('numCranes').value=st.nC; $('numLifts').value=st.nL; $('episodes').value=st.episodes; $('seedRuns').value=st.seedRuns; $('candidateK').value=st.candidateK; $('maxSteps').value=st.maxSteps; $('baseSeed').value=st.baseSeed; $('learningMode').value='multi'; $('learningAlgorithm').value='mappo';}
const uiYield=(ms=100)=>new Promise(res=>setTimeout(res,ms));
function setButtonsDisabled(disabled){document.querySelectorAll?.('button')?.forEach(b=>b.disabled=disabled);}
async function trainMappoCurriculumModel(stages=curriculumStages(), preserveUi=false){
  const snap=snapshotControls(); history=[]; best=null; initMappoModel(); let globalEp=0, stageResults=[];
  log(`Curriculum MAPPO 학습 시작: ${stages.map(s=>`${s.name} ${s.nC}C/${s.nL}L/${s.episodes}ep`).join(' → ')}`);
  for(const st of stages){
    applyStageConfig(st); const c=cfg(), span=Math.max(1,c.seedRuns||1); let stageBest=null, doneSum=0;
    for(let ep=1; ep<=c.episodes; ep++){
      globalEp++; const trainSeed=(c.baseSeed||101)+((ep-1)%span);
      generateScenario(trainSeed,true,true); setSeed(trainSeed*3000+globalEp);
      const eps=c.epsStart+(c.epsEnd-c.epsStart)*(ep/c.episodes);
      const r=runEpisode(globalEp,true,eps,'mappo'); history.push({...compactEval(r),stage:st.name}); doneSum+=r.done/r.total;
      if(r.done===r.total && (!best || r.makespan<best.makespan)) best=r;
      if(r.done===r.total && (!stageBest || r.makespan<stageBest.makespan)) stageBest=r;
      const shouldRender=(ep===1 || ep%5===0 || ep===c.episodes);
      if(shouldRender){
        updateAll(globalEp); drawCharts(); drawSim(best||r);
        $('interpretLog').textContent=`Curriculum 진행 중\n- 현재 단계: ${st.name} (${c.nC} cranes / ${c.nL} lifts)\n- 단계 episode: ${ep}/${c.episodes}\n- 전체 episode: ${globalEp}\n- 현재 best makespan: ${best?best.makespan:'-'}분\n\n브라우저 멈춤을 막기 위해 매 episode마다 제어권을 돌려주고, 5 episode마다 화면을 갱신합니다.`;
      } else {
        $('interpretLog').textContent=`Curriculum 진행 중 · ${st.name} ${ep}/${c.episodes} episode · 전체 ${globalEp} episode · best ${best?best.makespan:'-'}분`;
      }
      await uiYield();
    }
    stageResults.push({stage:st.name,nC:c.nC,nL:c.nL,episodes:c.episodes,seedRuns:c.seedRuns,candidateK:c.candidateK,bestMakespan:stageBest?stageBest.makespan:null,avgComplete:doneSum/Math.max(1,c.episodes)*100});
    log(`${st.name} 완료: ${c.nC} cranes / ${c.nL} lifts · ${c.episodes}ep · best ${stageBest?stageBest.makespan:'-'}분 · 완료율 평균 ${(doneSum/Math.max(1,c.episodes)*100).toFixed(1)}%`);
    renderCurriculumRows(stageResults,{makespan:0,makespanSd:0,softInter:0,hardInter:0,travel:0,setup:0},{makespan:0,makespanSd:0,softInter:0,hardInter:0,travel:0,setup:0});
    await uiYield();
  }
  const model=cloneMappoModel(); model.curriculum={stages:stageResults,totalEpisodes:history.length};
  if(!preserveUi) restoreControls(snap); else applyStageConfig(stages[stages.length-1]);
  return {model,stageResults};
}
function evaluateMappoModelOnConfig(model,evalCfg,seeds){const snap=snapshotControls(); applyStageConfig(evalCfg); restoreMappoModel(model); const samples=[]; for(const seed of seeds){generateScenario(seed,true,true); samples.push(runEpisode(0,false,0,'mappo'));} restoreControls(snap); return samples;}
async function evaluateMappoModelOnConfigAsync(model,evalCfg,seeds,label='평가'){
  const snap=snapshotControls(); applyStageConfig(evalCfg); restoreMappoModel(model); const samples=[];
  for(let i=0;i<seeds.length;i++){
    generateScenario(seeds[i],true,true); samples.push(runEpisode(0,false,0,'mappo'));
    if(i===0 || (i+1)%3===0 || i===seeds.length-1){$('interpretLog').textContent=`${label} 진행 중: ${i+1}/${seeds.length} seed`; await uiYield();}
  }
  restoreControls(snap); return samples;
}
async function runCurriculumTrainingEvaluation(){
  if(training) return; training=true; setButtonsDisabled(true);
  const snap=snapshotControls();
  try{
    const {model,stageResults}=await trainMappoCurriculumModel(curriculumStages(),false);
    const standard={name:'표준 2단계 평가',nC:3,nL:24,episodes:150,seedRuns:10,candidateK:5,maxSteps:220,baseSeed:101}, hard={name:'확장 3단계 평가',nC:4,nL:36,episodes:220,seedRuns:15,candidateK:6,maxSteps:320,baseSeed:301};
    const stdSeeds=Array.from({length:10},(_,i)=>101+i), hardSeeds=Array.from({length:15},(_,i)=>301+i);
    const stdSamples=await evaluateMappoModelOnConfigAsync(model,standard,stdSeeds,'표준 3C/24L 평가');
    const hardSamples=await evaluateMappoModelOnConfigAsync(model,hard,hardSeeds,'확장 4C/36L 평가');
    const stdStats=summarizePolicy(stdSamples,0,stdSeeds.length), hardStats=summarizePolicy(hardSamples,0,hardSeeds.length);
    restoreControls(snap); restoreMappoModel(model); renderCurriculumRows(stageResults,stdStats,hardStats); renderCurriculumInterpretation(stageResults,stdStats,hardStats); applyStageConfig(standard); generateScenario(101,true,true); lastEval=runEpisode(0,false,0,'mappo'); drawSim(lastEval); renderSchedule(lastEval); updateAll(history.length); drawCharts(); replayTime=0; updateReplayControls(lastEval); restoreControls(snap); log('Curriculum 1→2→3 학습 및 표준/확장 평가 완료.'); return {model,stageResults,stdStats,hardStats};
  }catch(err){log(`Curriculum 실행 오류: ${err.message}`); throw err;}
  finally{training=false; setButtonsDisabled(false); restoreControls(snap);}
}
function renderCurriculumRows(stageResults,stdStats,hardStats){$('repeatRows').innerHTML=[...stageResults.map(s=>`<tr><td>${s.stage} 학습</td><td>${s.nC} cranes / ${s.nL} lifts</td><td>${s.episodes} ep, seed ${s.seedRuns}</td><td>완료율 ${s.avgComplete.toFixed(1)}%</td><td>best ${s.bestMakespan??'-'}분 / K=${s.candidateK}</td></tr>`),`<tr><td>표준 평가</td><td>3 cranes / 24 lifts</td><td>${stdStats.makespan.toFixed(1)}±${stdStats.makespanSd.toFixed(1)}분</td><td>Soft ${stdStats.softInter.toFixed(1)} / HardMask ${stdStats.hardInter.toFixed(1)}</td><td>Travel ${stdStats.travel.toFixed(1)} / Setup ${stdStats.setup.toFixed(1)}</td></tr>`,`<tr><td>확장 평가</td><td>4 cranes / 36 lifts</td><td>${hardStats.makespan.toFixed(1)}±${hardStats.makespanSd.toFixed(1)}분</td><td>Soft ${hardStats.softInter.toFixed(1)} / HardMask ${hardStats.hardInter.toFixed(1)}</td><td>Travel ${hardStats.travel.toFixed(1)} / Setup ${hardStats.setup.toFixed(1)}</td></tr>`].join('');}
function renderCurriculumInterpretation(stageResults,stdStats,hardStats){$('interpretLog').textContent=[`Curriculum 1→2→3 학습 완료`,`- 1단계: 2 cranes / 12 lifts로 기본 작업 완료·setup 유지·idle 회피를 먼저 학습`,`- 2단계: 3 cranes / 24 lifts로 현재 표준 Stage-1.5 조건에 적응`,`- 3단계: 4 cranes / 36 lifts로 더 많은 크레인/양중물 및 실제 양중반경 간섭 조합에 적응`,`- 표준 평가 평균 makespan: ${stdStats.makespan.toFixed(1)}분, Soft ${stdStats.softInter.toFixed(1)}, HardMask ${stdStats.hardInter.toFixed(1)}`,`- 확장 평가 평균 makespan: ${hardStats.makespan.toFixed(1)}분, Soft ${hardStats.softInter.toFixed(1)}, HardMask ${hardStats.hardInter.toFixed(1)}`,`- 해석: actor/critic weight를 유지한 채 난이도만 올렸으므로, 단순 재학습이 아니라 curriculum learning 결과입니다.`].join('\n');}
function scrollToSection(id,btn){
  document.querySelectorAll?.('.tabbar button')?.forEach(b=>b.classList.remove('active'));
  if(btn)btn.classList.add('active');
  const el=$(id); if(el&&el.scrollIntoView)el.scrollIntoView({behavior:'smooth',block:'start'});
}
function applyPlanningControls(){
  const useCurrent=$('planningUseCurrent')?.checked && cranes.length && lifts.length;
  const nC=useCurrent?cranes.length:Math.max(2,Math.min(5,+$('planningCranes').value||3)), nL=useCurrent?lifts.length:Math.max(12,Math.min(50,+$('planningLifts').value||24));
  $('numCranes').value=nC; $('numLifts').value=nL; $('candidateK').value=Math.max(2,Math.min(12,+$('planningK').value||5));
  $('maxSteps').value=Math.max(50,Math.min(800,+$('planningMaxSteps').value||Math.max(160,nL*10))); $('learningAlgorithm').value='mappo'; $('learningMode').value='multi';
}
function planningPolicyLabel(policy){return policy==='mappo'?'Curriculum MAPPO':policy==='nearest'?'Nearest-lift-first':policy==='radiusPriority'?'Same-radius-priority':policy==='random'?'Random':policy;}
function loadCurrentScenarioToEditor(){
  if(!$('scenarioEditor'))return;
  const lines=[];
  cranes.forEach(c=>lines.push(['crane',c.id,(c.x??c.setupX??0).toFixed(1),(c.y??c.setupY??0).toFixed(1),c.type||'50톤 모바일 크레인'].join(',')));
  lifts.forEach(l=>lines.push(['lift',l.id,(l.x??0).toFixed(1),(l.y??0).toFixed(1),l.duration||cfg().fixedDuration,l.weight||'',l.priority||'normal'].join(',')));
  $('scenarioEditor').value=lines.join('\n');
}
function parseScenarioEditor(){
  const lines=($('scenarioEditor')?.value||'').split(/\r?\n/).map(x=>x.trim()).filter(x=>x&&!x.startsWith('#'));
  const newCranes=[], newLifts=[];
  for(const line of lines){
    const p=line.split(',').map(x=>x.trim()), type=(p[0]||'').toLowerCase();
    if(type==='crane'||type==='c'){
      const id=p[1]||`C${newCranes.length+1}`, x=Math.max(0,Math.min(100,parseFloat(p[2]))), y=Math.max(0,Math.min(100,parseFloat(p[3])));
      if(!Number.isFinite(x)||!Number.isFinite(y)) throw new Error(`크레인 좌표 오류: ${line}`);
      const craneType=p[4]||cfg().craneType;
      newCranes.push({id,type:craneType,capacityTon:parseFloat(p[5])||50,x,y,home:{x,y},lastZone:null,setupX:x,setupY:y,availableTime:0,done:0,idle:0,reward:0,moveDistance:0,schedule:[]});
    } else if(type==='lift'||type==='l'){
      const id=p[1]||`L${newLifts.length+1}`, x=Math.max(0,Math.min(100,parseFloat(p[2]))), y=Math.max(0,Math.min(100,parseFloat(p[3]))), duration=parseFloat(p[4])||cfg().fixedDuration;
      if(!Number.isFinite(x)||!Number.isFinite(y)) throw new Error(`양중물 좌표 오류: ${line}`);
      newLifts.push({id,x,y,duration,weight:p[5]?parseFloat(p[5]):null,priority:p[6]||'normal',zone:Math.floor(x/25)+'-'+Math.floor(y/25),status:'todo',by:null,start:null,finish:null});
    } else throw new Error(`type은 crane 또는 lift여야 합니다: ${line}`);
  }
  if(!newCranes.length) throw new Error('크레인이 1대 이상 필요합니다.');
  if(!newLifts.length) throw new Error('양중물이 1개 이상 필요합니다.');
  return {newCranes,newLifts};
}
function applyScenarioEditor(){
  try{
    const parsed=parseScenarioEditor(); cranes=parsed.newCranes; lifts=parsed.newLifts; $('numCranes').value=cranes.length; $('numLifts').value=lifts.length; $('planningCranes').value=cranes.length; $('planningLifts').value=lifts.length; if($('planningUseCurrent'))$('planningUseCurrent').checked=true;
    lastEval=null; best=null; drawSim(); renderAgents(null); renderSchedule({cranes}); updateReplayControls(null); log(`현장 데이터 적용: 크레인 ${cranes.length}대, 양중물 ${lifts.length}개`);
  }catch(err){log(`현장 데이터 적용 오류: ${err.message}`);}
}
function addEditorCrane(){
  if(!$('scenarioEditor'))return; const id=`C${cranes.length+1}`, line=`crane,${id},${Math.min(90,10+cranes.length*15)},10,50톤 모바일 크레인`;
  $('scenarioEditor').value=($('scenarioEditor').value.trim()?$('scenarioEditor').value.trim()+'\n':'')+line; applyScenarioEditor();
}
function addEditorLift(){
  if(!$('scenarioEditor'))return; const id=`L${lifts.length+1}`, line=`lift,${id},${Math.min(90,20+lifts.length*5)},${Math.min(90,30+lifts.length*3)},${cfg().fixedDuration},,normal`;
  $('scenarioEditor').value=($('scenarioEditor').value.trim()?$('scenarioEditor').value.trim()+'\n':'')+line; applyScenarioEditor();
}
function generatePlanningScenario(){
  const snap=snapshotControls(); const seed=+$('planningSeed').value||501; if($('planningUseCurrent'))$('planningUseCurrent').checked=false; applyPlanningControls();
  generateScenario(seed,false,true); $('planningUseCurrent').checked=true; loadCurrentScenarioToEditor(); restoreControls(snap); log(`계획 시나리오 생성: seed ${seed}, ${$('planningCranes').value} cranes / ${$('planningLifts').value} lifts`);
}
function runPlanningPolicy(policy,seed){
  const snap=snapshotControls(); applyPlanningControls();
  const useCurrent=$('planningUseCurrent')?.checked && cranes.length && lifts.length;
  if(!useCurrent) generateScenario(seed,true,true); else {cranes.forEach(c=>{c.availableTime=0;c.done=0;c.idle=0;c.reward=0;c.moveDistance=0;c.schedule=[];c.setupX=Number.isFinite(c.setupX)?c.setupX:c.x;c.setupY=Number.isFinite(c.setupY)?c.setupY:c.y;}); lifts.forEach(l=>{l.status='todo';l.by=null;l.start=null;l.finish=null;});}
  if(policy==='mappo' && !mappoModel){initMappoModel(); log('주의: 저장/학습된 MAPPO 모델이 없어 임시 초기 MAPPO weight로 계획을 생성했습니다. 먼저 Curriculum 학습 또는 모델 로드를 권장합니다.');}
  const result=runEpisode(0,false,0,policy); restoreControls(snap); return result;
}
function planningAvgRadius(r){const ev=r?.events||[], avg=arr=>arr.reduce((a,b)=>a+b,0)/Math.max(1,arr.length), sb=softBufferValue(); return {lift:avg(ev.map(e=>e.actualLiftRadius||0)), danger:avg(ev.map(e=>(e.actualLiftRadius||0)+sb))};}
function renderPlanningSummary(r,label='계획안'){
  if(!r)return; const donePct=Math.round(r.done/Math.max(1,r.total)*100), rad=planningAvgRadius(r);
  $('pMakespan').textContent=`${r.makespan}분`; $('pDone').textContent=`${donePct}%`; $('pInter').textContent=`S ${r.softInter||0} / EH ${r.hardExecuted||0} / M ${r.hardMask??r.hardInter??0}`; $('pPlanTime').textContent=`${r.travelTotal.toFixed(1)} / ${r.setupTotal.toFixed(1)}`; $('pMoveMetric').textContent=r.moveTotal.toFixed(1);
  $('planningLog').textContent=[`${label} 생성 완료`,`- 완료: ${r.done}/${r.total} (${donePct}%)`,`- Makespan: ${r.makespan}분, reward ${r.reward.toFixed(1)}`,`- Soft 노출: ${r.softInter||0}회, 실행 Hard overlap: ${r.hardExecuted||0}회, HardMask 차단: ${r.hardMask??r.hardInter??0}회`,`- Travel ${r.travelTotal.toFixed(1)}분 / Setup ${r.setupTotal.toFixed(1)}분 / Move ${r.moveTotal.toFixed(1)}`,`- 평균 실제 양중반경: ${rad.lift.toFixed(1)}, 평균 실제 위험반경: ${rad.danger.toFixed(1)}`,`- 실제 양중반경/위험반경 중심은 crane/setup point입니다. Hard overlap은 실행되지 않고 action mask로 차단됩니다.`].join('\n');
}
function renderPlanningSchedule(r){
  const cs=(r&&r.cranes)||[];
  $('planningSchedule').textContent=cs.map(c=>`${c.id}\n  ${((c.schedule||[]).join('\n  '))||'(작업 없음)'}`).join('\n\n');
}
function renderPlanningGantt(r){
  const maxT=Math.max(1,r.makespan||1);
  $('planningGantt').innerHTML=r.cranes.map((cr,i)=>{
    const evs=(r.events||[]).filter(e=>e.craneId===cr.id).sort((a,b)=>a.start-b.start), bars=evs.map(e=>{
      const left=Math.max(0,e.start/maxT*100), width=Math.max(1,(e.finish-e.start)/maxT*100), color=e.interferenceLevel==='soft'?'#f59e0b':craneColor(i);
      return `<div class="gantt-bar" title="${e.craneId} ${e.liftId}: ${Math.round(e.start)}~${Math.round(e.finish)}분" style="left:${left}%;width:${width}%;background:${color}">${e.liftId}</div>`;
    }).join('');
    return `<div class="gantt-row"><b>${cr.id}</b><div class="gantt-line">${bars}</div></div>`;
  }).join('')+`<div class="note">0 ~ ${maxT}분 · 노랑/주황 bar는 soft 노출 포함 작업입니다.</div>`;
}
function renderPlanningCompareRows(results){
  $('planningCompareRows').innerHTML=results.map(x=>`<tr><td>${x.name}</td><td>${x.r.done}/${x.r.total}</td><td>${x.r.makespan}분</td><td>${x.r.reward.toFixed(1)}</td><td>Soft ${x.r.softInter||0} / 실행Hard ${x.r.hardExecuted||0} / HardMask ${x.r.hardMask??x.r.hardInter??0}</td><td>${x.r.travelTotal.toFixed(1)}</td><td>${x.r.setupTotal.toFixed(1)}</td><td>${x.r.moveTotal.toFixed(1)}</td></tr>`).join('');
}
function runPlanningWithModel(){
  const seed=+$('planningSeed').value||501, policy=$('planningPolicy').value;
  const r=runPlanningPolicy(policy,seed); lastPlanningResult=r; lastPlanningCompare=[{name:planningPolicyLabel(policy),r}]; lastEval=r; replayTime=0;
  renderPlanningSummary(r,planningPolicyLabel(policy)); renderPlanningSchedule(r); renderPlanningGantt(r); renderPlanningCompareRows(lastPlanningCompare); drawSim(r); renderSchedule(r); renderMetricLog(r,`${planningPolicyLabel(policy)} 계획 생성`); updateAll(history.length); updateReplayControls(r); log(`${planningPolicyLabel(policy)} 계획 생성 완료: seed ${seed}, makespan ${r.makespan}분`);
}
function comparePlanningModels(){
  const seed=+$('planningSeed').value||501, policies=['mappo','nearest','radiusPriority'];
  const results=policies.map(p=>({name:planningPolicyLabel(p),r:runPlanningPolicy(p,seed)}));
  lastPlanningCompare=results; const bestPlan=results.reduce((a,b)=>b.r.makespan<a.r.makespan?b:a,results[0]); lastPlanningResult=bestPlan.r; lastEval=bestPlan.r; replayTime=0;
  renderPlanningCompareRows(results); renderPlanningSummary(bestPlan.r,`최상 계획안: ${bestPlan.name}`); renderPlanningSchedule(bestPlan.r); renderPlanningGantt(bestPlan.r); drawSim(bestPlan.r); renderSchedule(bestPlan.r); renderMetricLog(results,'AI vs Heuristic 계획 비교'); updateAll(history.length); updateReplayControls(bestPlan.r);
  const m=results.find(x=>x.name.includes('MAPPO')), near=results.find(x=>x.name.includes('Nearest')); let msg=`AI vs Heuristic 비교 완료: 최상 makespan은 ${bestPlan.name} ${bestPlan.r.makespan}분입니다.`;
  if(m&&near) msg+=` MAPPO-Nearest 차이: ${(m.r.makespan-near.r.makespan).toFixed(1)}분.`; $('planningLog').textContent += `\n\n${msg}`; log(msg);
}
function downloadTextFile(fileName,text,type='text/plain'){
  if(typeof Blob==='undefined')return null; const blob=new Blob([text],{type}), url=URL.createObjectURL(blob), a=document.createElement('a'); a.href=url; a.download=fileName; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url); return fileName;
}
function planningExportPayload(){return {createdAt:new Date().toISOString(),model:mappoModel?{version:mappoModel.version,stats:mappoModel.stats}:null,config:cfg(),planning:{seed:+$('planningSeed').value||501,policy:$('planningPolicy').value,cranes:+$('planningCranes').value,lifts:+$('planningLifts').value,candidateK:+$('planningK').value},result:compactEval(lastPlanningResult),events:lastPlanningResult?.events||[],craneSchedules:(lastPlanningResult?.cranes||[]).map(c=>({id:c.id,finish:c.availableTime,done:c.done,moveDistance:c.moveDistance,schedule:c.schedule})),comparison:lastPlanningCompare.map(x=>({name:x.name,result:compactEval(x.r)}))};}
function exportPlanningJson(){if(!lastPlanningResult){log('저장할 계획안이 없습니다. 먼저 AI 계획 생성 또는 비교를 실행하세요.'); return;} const seed=+$('planningSeed').value||501, name=`planning_${seed}_${Date.now()}.json`; downloadTextFile(name,JSON.stringify(planningExportPayload(),null,2),'application/json'); log(`계획 JSON 저장: ${name}`);}
function exportPlanningCsv(){
  if(!lastPlanningResult){log('저장할 계획안이 없습니다. 먼저 AI 계획 생성 또는 비교를 실행하세요.'); return;}
  const rows=[['craneId','liftId','start','finish','travel','setup','duration','moveDist','radiusCenterX','radiusCenterY','liftX','liftY','actualLiftRadius','dangerRadius','sameRadius','softExposure']];
  const sb=softBufferValue(); (lastPlanningResult.events||[]).forEach(e=>rows.push([e.craneId,e.liftId,e.start.toFixed(2),e.finish.toFixed(2),e.travel.toFixed(2),e.setup.toFixed(2),e.duration,e.moveDist.toFixed(2),e.radiusCenterX.toFixed(2),e.radiusCenterY.toFixed(2),e.liftX.toFixed(2),e.liftY.toFixed(2),(e.actualLiftRadius||0).toFixed(2),((e.actualLiftRadius||0)+sb).toFixed(2),e.sameRadius?1:0,e.interferenceLevel==='soft'?1:0]));
  const csv=rows.map(r=>r.map(v=>`"${String(v).replaceAll('"','""')}"`).join(',')).join('\n'), name=`planning_${+$('planningSeed').value||501}_${Date.now()}.csv`;
  downloadTextFile(name,csv,'text/csv'); log(`계획 CSV 저장: ${name}`);
}
function pyPolicyLabel(k){return k==='mappo'?'PyTorch MAPPO':k==='nearest'?'Nearest-lift-first':k==='radiusPriority'?'Same-radius-priority':k==='random'?'Random':k;}
function pySummaryToResult(s,total=cfg().nL){return {done:Math.round((s.completeRate||0)/100*total),total,makespan:+(s.makespan||0).toFixed(2),reward:s.reward||0,softInter:s.soft||0,hardExecuted:s.hardExecuted||0,hardMask:s.hardMask||0,hardInter:s.hardMask||0,travelTotal:s.travel||0,setupTotal:s.setup||0,moveTotal:s.move||0,liftTotal:total*cfg().fixedDuration,same:0,cranes:[],events:[]};}
function normalizeImportedEpisode(ep,label='PyTorch MAPPO'){
  if(!ep)return null; const events=(ep.events||[]).map((e,idx)=>{const mv=e.moveDist??e.move??0, dur=e.duration??cfg().fixedDuration, soft=Number(e.softConflict||e.softExposure||0); return {...e,craneId:e.craneId||`C${(e.craneIndex??0)+1}`,liftId:e.liftId||`L${(e.liftIndex??idx)+1}`,start:+(e.start||0),finish:+(e.finish||0),travel:+(e.travel||0),setup:+(e.setup||0),duration:dur,moveDist:mv,radiusCenterX:e.radiusCenterX??e.toX??0,radiusCenterY:e.radiusCenterY??e.toY??0,liftX:e.liftX??0,liftY:e.liftY??0,actualLiftRadius:e.actualLiftRadius??0,dangerRadius:e.dangerRadius??((e.actualLiftRadius??0)+softBufferValue()),interferenceLevel:soft>0?'soft':(e.interferenceLevel||'none'),softConflict:soft,hardExecuted:Number(e.hardExecuted||0),hardMask:Number(e.hardMask||0)};});
  const craneIds=[...new Set([...(ep.cranes||[]).map(c=>c.id),...events.map(e=>e.craneId)])].sort();
  const cranesNorm=craneIds.map((id,i)=>{const evs=events.filter(e=>e.craneId===id).sort((a,b)=>a.start-b.start), src=(ep.cranes||[]).find(c=>c.id===id)||{}; return {id,type:src.type||'50톤 모바일 크레인',done:evs.length,availableTime:Math.max(0,...evs.map(e=>e.finish),src.availableTime||src.available||0),idle:src.idle||0,reward:src.reward||0,moveDistance:evs.reduce((a,e)=>a+(e.moveDist||0),0),schedule:evs.map(e=>`${e.liftId}: ${Math.round(e.start)}~${Math.round(e.finish)}분 · setup ${Number(e.setup||0).toFixed(1)} · actualR ${Number(e.actualLiftRadius||0).toFixed(1)}${e.interferenceLevel==='soft'?' ⚠S':''}`)};});
  const total=ep.total||Math.max(events.length,cfg().nL), done=ep.done||events.length, makespan=ep.makespan??Math.round(Math.max(0,...events.map(e=>e.finish)));
  const liftsNorm=events.map(e=>({id:e.liftId,x:e.liftX,y:e.liftY,status:'done',by:e.craneId,start:e.start,finish:e.finish}));
  return {label,done,total,makespan,reward:ep.reward||0,softInter:ep.softInter??events.reduce((a,e)=>a+Number(e.softConflict||0),0),hardExecuted:ep.hardExecuted??events.reduce((a,e)=>a+Number(e.hardExecuted||0),0),hardMask:ep.hardMask??ep.hardInter??events.reduce((a,e)=>a+Number(e.hardMask||0),0),hardInter:ep.hardInter??ep.hardMask??events.reduce((a,e)=>a+Number(e.hardMask||0),0),travelTotal:ep.travelTotal??events.reduce((a,e)=>a+Number(e.travel||0),0),setupTotal:ep.setupTotal??events.reduce((a,e)=>a+Number(e.setup||0),0),moveTotal:ep.moveTotal??events.reduce((a,e)=>a+Number(e.moveDist||0),0),liftTotal:ep.liftTotal??done*cfg().fixedDuration,same:ep.same||0,events,cranes:cranesNorm,lifts:liftsNorm,source:'PyTorch MAPPO import'};
}

function renderLearningCurve(raw){
  const el=$('learningCurveLog'); if(!el)return;
  const curve=raw.learningCurve||raw.trainLog||[];
  if(!curve.length){el.textContent='learningCurve 배열이 없습니다. 새 PyTorch 학습을 실행하면 JSON에 episode별 makespan/reward가 저장됩니다.'; return;}
  const first=curve.slice(0,Math.min(10,curve.length));
  const last=curve.slice(Math.max(0,curve.length-20));
  const avg=(arr,k)=>arr.reduce((a,r)=>a+(Number(r[k])||0),0)/Math.max(1,arr.length);
  const best=curve.reduce((a,b)=>Number(b.makespan)<Number(a.makespan)?b:a,curve[0]);
  const checkpoints=curve.filter((_,i)=>i===0 || i===curve.length-1 || (i+1)%Math.max(1,Math.round(curve.length/10))===0)
    .map(r=>`ep ${r.ep}: makespan ${Number(r.makespan).toFixed(1)}, reward ${Number(r.reward).toFixed(1)}, S ${r.soft||0}/EH ${r.hardExecuted||0}/M ${r.hardMask||0}`);
  el.textContent=[
    `episodes: ${curve.length}`,
    `best: ep ${best.ep}, seed ${best.seed}, makespan ${Number(best.makespan).toFixed(1)}`,
    `first ${first.length} avg makespan: ${avg(first,'makespan').toFixed(1)}`,
    `last ${last.length} avg makespan: ${avg(last,'makespan').toFixed(1)}`,
    'checkpoints:', ...checkpoints
  ].join('\n');
}
function renderPyTorchAggregate(raw){
  renderLearningCurve(raw);
  const block=raw.seen||raw.out||null; if(!block)return;
  const rows=Object.entries(block).map(([k,s])=>({name:pyPolicyLabel(k),r:pySummaryToResult(s)})); renderBaselineRows(rows); renderPlanningCompareRows(rows); const lines=[];
  if(raw.seen&&raw.unseen){['seen','unseen'].forEach(split=>{const m=raw[split].mappo, near=raw[split].nearest; if(m&&near)lines.push(`${split}: PyTorch MAPPO makespan ${m.makespan.toFixed(1)}분, Nearest 대비 ${((near.makespan-m.makespan)/near.makespan*100).toFixed(1)}% 개선`);});}
  $('interpretLog').textContent=['PyTorch MAPPO 결과 JSON 집계 불러오기 완료',...lines,'집계 행은 Baseline/계획 비교표에 표시했고, 대표 episode가 있으면 2D 리플레이에도 연결됩니다.'].join('\n');
}
function applyImportedPyTorchResult(raw,source='PyTorch JSON'){
  renderPyTorchAggregate(raw); const rep=(raw.representative&&(raw.representative.mappo||raw.representative.episode||raw.representative))||raw.episode||null;
  if(rep&&rep.events&&rep.events.length){const r=normalizeImportedEpisode(rep,'PyTorch MAPPO 대표 계획'); lastPlanningResult=r; lastPlanningCompare=[{name:'PyTorch MAPPO',r}]; lastEval=r; replayTime=0; renderPlanningSummary(r,'PyTorch MAPPO 대표 계획'); renderPlanningSchedule(r); renderPlanningGantt(r); renderPlanningCompareRows([{name:'PyTorch MAPPO',r}]); drawSim(r); renderSchedule(r); renderMetricLog(r,'PyTorch MAPPO import'); updateAll(history.length); updateReplayControls(r); log(`${source} 불러오기 완료: 대표 episode ${r.done}/${r.total}, makespan ${r.makespan}분`);} else log(`${source} 불러오기 완료: 집계 지표만 표시했습니다. 리플레이용 events는 JSON의 representative.mappo.events가 필요합니다.`);
}
function loadPyTorchResultFile(file){if(!file)return; const fr=new FileReader(); fr.onload=()=>{try{applyImportedPyTorchResult(JSON.parse(fr.result),file.name);}catch(e){log(`PyTorch 결과 JSON 파싱 오류: ${e.message}`);}}; fr.readAsText(file);}
async function loadBundledPyTorchResult(){try{const res=await fetch('python_mappo/outputs/pytorch_mappo_validation_result.json',{cache:'no-store'}); if(!res.ok)throw new Error(`HTTP ${res.status}`); applyImportedPyTorchResult(await res.json(),'내장 PyTorch 결과');}catch(e){log(`내장 PyTorch 결과를 불러오지 못했습니다: ${e.message}. npm run train:pytorch-mappo 후 다시 시도하세요.`);}}
function summarizePolicy(samples,wins,total){const vals=k=>samples.map(s=>s[k]??0); const avg=a=>a.reduce((x,y)=>x+y,0)/Math.max(1,a.length); const sd=a=>{let m=avg(a); return Math.sqrt(avg(a.map(x=>(x-m)*(x-m))));}; return {n:samples.length,makespan:avg(vals('makespan')),makespanSd:sd(vals('makespan')),reward:avg(vals('reward')),rewardSd:sd(vals('reward')),inter:avg(vals('inter')),softInter:avg(vals('softInter')),hardInter:avg(vals('hardInter')),interPenalty:avg(vals('interPenaltyTotal')),travel:avg(vals('travelTotal')),setup:avg(vals('setupTotal')),move:avg(vals('moveTotal')),winRate:wins/Math.max(1,total)};}
function runRepeatedValidation(){const c=cfg(); const n=Math.max(3,Math.min(100,c.seedRuns||10)), base=c.baseSeed||101; const names=['RL-before','RL-single','RL-multi','MAPPO-shared','Random','Nearest-lift-first','Same-radius-priority']; const bag=Object.fromEntries(names.map(n=>[n,[]])); const wins=Object.fromEntries(names.map(n=>[n,0])); let lastResults=[];
  log(`Seed 반복 검증 시작: ${n}개 scenario, seed ${base}~${base+n-1}. RL-single은 각 seed별 단일 학습, RL-multi는 여러 seed를 episode마다 바꿔 학습합니다.`);
  const multiQ=trainMultiScenarioModel();
  const mappoShared=trainMappoMultiScenarioModel();
  for(let s=0;s<n;s++){
    const seed=base+s; generateScenario(seed,true);
    initQTables();
    const before=runEpisode(0,false,0,'rl');
    initQTables(); generateScenario(seed,true,true);
    trainCurrentScenario(seed*1000);
    const single=runEpisode(0,false,0,'rl');
    restoreQTables(multiQ); generateScenario(seed,true,true);
    const multi=runEpisode(0,false,0,'rl');
    restoreMappoModel(mappoShared); generateScenario(seed,true,true);
    const mappo=runEpisode(0,false,0,'mappo');
    setSeed(seed+700000); const random=runEpisode(0,false,0,'random');
    const nearest=runEpisode(0,false,0,'nearest');
    const sameZone=runEpisode(0,false,0,'radiusPriority');
    lastResults=[{name:'RL-before',r:before},{name:'RL-single',r:single},{name:'RL-multi',r:multi},{name:'MAPPO-shared',r:mappo},{name:'Random',r:random},{name:'Nearest-lift-first',r:nearest},{name:'Same-radius-priority',r:sameZone}];
    lastResults.forEach(x=>bag[x.name].push(x.r));
    const minMake=Math.min(...lastResults.map(x=>x.r.makespan)); lastResults.filter(x=>x.r.makespan===minMake).forEach(x=>wins[x.name]+=1/lastResults.filter(y=>y.r.makespan===minMake).length);
  }
  const summary=names.map(name=>({name,stats:summarizePolicy(bag[name],wins[name],n)}));
  lastEval=(lastResults.find(x=>x.name==='MAPPO-shared')||lastResults.find(x=>x.name==='RL-multi')).r; renderBaselineRows(lastResults); renderRepeatRows(summary); renderMetricLog(lastResults,'마지막 Seed 상세 비교'); renderInterpretation(summary); updateAll(history.length); drawCharts(); drawSim(lastEval); renderSchedule(lastEval); replayTime=0; updateReplayControls(lastEval); log('Seed 반복 검증 완료: RL-single / RL-multi / baseline의 평균/표준편차/승률 및 자동 해석을 갱신했습니다.');}

function evaluatePolicySetForSeed(seed,singleQ,multiQ,mappoShared){
  const names=['RL-before','RL-single','RL-multi','MAPPO-shared','Random','Nearest-lift-first','Same-radius-priority'];
  generateScenario(seed,true); initQTables(); const before=runEpisode(0,false,0,'rl');
  restoreQTables(singleQ); generateScenario(seed,true,true); const single=runEpisode(0,false,0,'rl');
  restoreQTables(multiQ); generateScenario(seed,true,true); const multi=runEpisode(0,false,0,'rl');
  restoreMappoModel(mappoShared); generateScenario(seed,true,true); const mappo=runEpisode(0,false,0,'mappo');
  setSeed(seed+700000); const random=runEpisode(0,false,0,'random');
  const nearest=runEpisode(0,false,0,'nearest');
  const sameRadius=runEpisode(0,false,0,'radiusPriority');
  return [{name:names[0],r:before},{name:names[1],r:single},{name:names[2],r:multi},{name:names[3],r:mappo},{name:names[4],r:random},{name:names[5],r:nearest},{name:names[6],r:sameRadius}];
}
function runSeenUnseenEvaluation(){const c=cfg(); const seenN=Math.max(3,Math.min(30,c.seedRuns||10)), unseenN=Math.max(seenN,Math.min(60,seenN*3)); const base=c.baseSeed||101, unseenBase=base+100; const names=['RL-before','RL-single','RL-multi','MAPPO-shared','Random','Nearest-lift-first','Same-radius-priority'];
  log(`Seen/Unseen 평가 시작: seen seed ${base}~${base+seenN-1}, unseen seed ${unseenBase}~${unseenBase+unseenN-1}. MAPPO-shared도 포함합니다.`);
  initQTables(); generateScenario(base,true,true); trainCurrentScenario(0); const singleQ=cloneQTables();
  const multiQ=trainMultiScenarioModel();
  const mappoShared=trainMappoMultiScenarioModel();
  const seen=Object.fromEntries(names.map(n=>[n,[]])), unseen=Object.fromEntries(names.map(n=>[n,[]])); let lastResults=[];
  for(let s=0;s<seenN;s++){lastResults=evaluatePolicySetForSeed(base+s,singleQ,multiQ,mappoShared); lastResults.forEach(x=>seen[x.name].push(x.r));}
  for(let s=0;s<unseenN;s++){lastResults=evaluatePolicySetForSeed(unseenBase+s,singleQ,multiQ,mappoShared); lastResults.forEach(x=>unseen[x.name].push(x.r));}
  const summary=names.map(name=>({name,seen:summarizePolicy(seen[name],0,seenN),unseen:summarizePolicy(unseen[name],0,unseenN)}));
  renderSeenUnseenRows(summary); renderSeenUnseenInterpretation(summary,base,seenN,unseenBase,unseenN); renderBaselineRows(lastResults); lastEval=(lastResults.find(x=>x.name==='MAPPO-shared')||lastResults.find(x=>x.name==='RL-multi')).r; updateAll(history.length); drawCharts(); drawSim(lastEval); renderSchedule(lastEval); replayTime=0; updateReplayControls(lastEval); log('Seen/Unseen 평가 완료: MAPPO-shared 포함, 학습 seed와 미학습 seed의 일반화 gap을 계산했습니다.');}
function renderSeenUnseenRows(summary){$('repeatRows').innerHTML=summary.map(x=>{let gap=x.unseen.makespan-x.seen.makespan, pct=gap/Math.max(1,x.seen.makespan)*100; return `<tr><td>${x.name}</td><td>Seen ${x.seen.makespan.toFixed(1)}분</td><td>Unseen ${x.unseen.makespan.toFixed(1)}분</td><td>${gap>=0?'+':''}${gap.toFixed(1)}분 (${pct.toFixed(1)}%)</td><td>Unseen Soft ${x.unseen.softInter.toFixed(1)} / HardMask ${x.unseen.hardInter.toFixed(1)}</td></tr>`;}).join('');}
function renderSeenUnseenInterpretation(summary,base,seenN,unseenBase,unseenN){const by=Object.fromEntries(summary.map(x=>[x.name,x])); const bestUnseen=summary.reduce((a,b)=>b.unseen.makespan<a.unseen.makespan?b:a,summary[0]); const single=by['RL-single'], multi=by['RL-multi']; let lines=[];
  lines.push(`Seen/Unseen 일반화 평가`);
  lines.push(`- Seen: 학습에 사용한 seed 구간 ${base}~${base+seenN-1}, Unseen: 학습에 사용하지 않은 seed 구간 ${unseenBase}~${unseenBase+unseenN-1}`);
  lines.push(`- Unseen 평균 makespan 기준 최상 정책은 ${bestUnseen.name}, ${bestUnseen.unseen.makespan.toFixed(1)}분입니다.`);
  if(single&&multi){let d=single.unseen.makespan-multi.unseen.makespan; lines.push(`- RL-multi는 RL-single 대비 unseen makespan이 ${d>=0?'낮거나 같다':'높다'}고 볼 수 있습니다. 차이는 ${Math.abs(d).toFixed(1)}분입니다.`); let gapS=single.unseen.makespan-single.seen.makespan, gapM=multi.unseen.makespan-multi.seen.makespan; lines.push(`- 일반화 gap: RL-single ${gapS.toFixed(1)}분, RL-multi ${gapM.toFixed(1)}분입니다.`);}
  if(by['Same-radius-priority']&&multi&&by['Same-radius-priority'].unseen.makespan<multi.unseen.makespan) lines.push(`- Same-radius-priority가 unseen에서도 RL-multi보다 빠릅니다. 현재 단순 Stage-1 문제에서는 반경 기반 준비시간 절감 규칙이 강한 기준선입니다.`);
  lines.push(`- 해석: 이 결과는 연구 보고서에서 '학습 seed 성능'과 '미학습 배치 일반화 성능'을 분리해 제시하는 근거로 사용할 수 있습니다.`);
  $('interpretLog').textContent=lines.join('\n');}
function runRewardSensitivity(){const c=cfg(); const base=c.baseSeed||101, n=Math.max(3,Math.min(20,c.seedRuns||10)), originalP=$('pInterSoft').value; const penalties=[-1,-3,-5,-8]; let rows=[], detail=[];
  log(`Soft penalty 민감도 참고 분석 시작: ${penalties.join(', ')} / seed ${base}~${base+n-1}. hard 양중반경 중첩은 penalty가 아니라 action mask로 차단됩니다.`);
  for(const pVal of penalties){$('pInterSoft').value=pVal; const multiQ=trainMultiScenarioModel(); let samples=[]; for(let s=0;s<n;s++){restoreQTables(multiQ); generateScenario(base+s,true,true); samples.push(runEpisode(0,false,0,'rl'));} let stats=summarizePolicy(samples,0,n); rows.push({p:pVal,stats}); detail.push({name:`pSoft=${pVal}`,r:samples.at(-1)});}
  $('pInterSoft').value=originalP; renderSensitivityRows(rows); renderSensitivityInterpretation(rows); renderBaselineRows(detail); lastEval=detail.at(-1).r; updateAll(history.length); drawCharts(); drawSim(lastEval); renderSchedule(lastEval); replayTime=0; updateReplayControls(lastEval); log('Soft penalty 참고 분석 완료: soft 노출은 보조 지표이며, 선택 기준은 makespan·hard mask 정상작동·travel/setup/move 균형입니다.');}
function renderSensitivityRows(rows){$('repeatRows').innerHTML=rows.map(x=>`<tr><td>pSoft=${x.p}</td><td>${x.stats.makespan.toFixed(1)}±${x.stats.makespanSd.toFixed(1)}분</td><td>${x.stats.reward.toFixed(1)}±${x.stats.rewardSd.toFixed(1)}</td><td>Soft ${x.stats.softInter.toFixed(1)} / HardMask ${x.stats.hardInter.toFixed(1)}</td><td>Travel ${x.stats.travel.toFixed(1)} / Setup ${x.stats.setup.toFixed(1)} / Move ${x.stats.move.toFixed(1)}</td></tr>`).join('');}
function renderSensitivityInterpretation(rows){const bestMake=rows.reduce((a,b)=>b.stats.makespan<a.stats.makespan?b:a,rows[0]); const bestInter=rows.reduce((a,b)=>b.stats.inter<a.stats.inter?b:a,rows[0]); let balanced=rows.reduce((a,b)=>{let score=b.stats.makespan+8*b.stats.inter; let best=a.stats.makespan+8*a.stats.inter; return score<best?b:a;},rows[0]); let lines=[];
  lines.push(`Soft penalty 민감도 참고 분석: 안전반경 근접 작업은 허용 가능한 보조 지표`);
  lines.push(`- Makespan 기준 최상: pSoft=${bestMake.p}, 평균 ${bestMake.stats.makespan.toFixed(1)}분입니다.`);
  lines.push(`- Soft 노출 최소: pSoft=${bestInter.p}, 평균 ${bestInter.stats.softInter.toFixed(1)}회입니다. 단, soft는 실제 양중반경 중첩이 아니라 안전/영향 반경의 근접 작업 노출입니다.`);
  lines.push(`- 권장 해석: pSoft는 작은 음수로 유지하고, soft 노출 감소만을 위해 makespan이나 이동/setup 효율을 희생하지 않습니다.`);
  lines.push(`- 해석: hard 양중반경 중첩은 action mask로 실행 차단되므로 안전 제약의 핵심 검증 대상입니다. soft penalty는 근접 작업을 약하게 억제하는 shaping 항으로만 사용합니다.`);
  $('interpretLog').textContent=lines.join('\n');}

function renderRepeatRows(summary){$('repeatRows').innerHTML=summary.map(x=>`<tr><td>${x.name}</td><td>${x.stats.makespan.toFixed(1)}±${x.stats.makespanSd.toFixed(1)}분</td><td>${x.stats.reward.toFixed(1)}±${x.stats.rewardSd.toFixed(1)}</td><td>${x.stats.inter.toFixed(1)}회 (Soft ${x.stats.softInter.toFixed(1)} / HardMask ${x.stats.hardInter.toFixed(1)})</td><td>${(x.stats.winRate*100).toFixed(1)}%</td></tr>`).join('');}
function renderInterpretation(summary){const by=Object.fromEntries(summary.map(x=>[x.name,x.stats])); const bestMake=summary.reduce((a,b)=>b.stats.makespan<a.stats.makespan?b:a,summary[0]); const bestInter=summary.reduce((a,b)=>b.stats.inter<a.stats.inter?b:a,summary[0]); const before=by['RL-before'], single=by['RL-single'], multi=by['RL-multi'], mappo=by['MAPPO-shared']; let lines=[];
  lines.push(`반복 검증 해석`);
  lines.push(`- 평균 makespan 기준 최상 정책은 ${bestMake.name}입니다. 평균 ${bestMake.stats.makespan.toFixed(1)}분, 승률 ${(bestMake.stats.winRate*100).toFixed(1)}%입니다.`);
  lines.push(`- Hard 양중반경 중첩은 실행 불가능 action으로 mask되며, 표의 HardMask는 위험 후보가 차단된 횟수입니다. Soft는 허용 가능한 안전/영향 반경 근접 노출로 보조 지표입니다.`);
  if(before&&single){let d=before.makespan-single.makespan; lines.push(`- 단일 시나리오 RL 학습 전/후 비교: RL-single의 makespan은 ${d>=0?'감소':'증가'}했습니다. 변화량은 ${Math.abs(d).toFixed(1)}분입니다.`);}
  if(single&&multi){let d=single.makespan-multi.makespan; lines.push(`- RL-single과 RL-multi 비교: RL-multi는 RL-single 대비 makespan이 ${d>=0?'낮거나 같다':'높다'}고 볼 수 있습니다. 평균 차이는 ${Math.abs(d).toFixed(1)}분입니다.`);}
  if(mappo&&multi){let d=multi.makespan-mappo.makespan; lines.push(`- MAPPO-shared는 RL-multi 대비 makespan이 ${d>=0?'낮거나 같다':'높다'}고 볼 수 있습니다. 차이는 ${Math.abs(d).toFixed(1)}분입니다.`);}
  if(by['Same-radius-priority']&&multi){if(by['Same-radius-priority'].makespan<multi.makespan) lines.push(`- Same-radius-priority가 RL-multi보다 makespan이 낮습니다. 현재 Stage 1 환경에서는 동일 설치반경 내 준비시간 절감 규칙이 성능을 강하게 지배한다는 의미입니다.`); else lines.push(`- RL-multi가 Same-radius-priority와 같거나 더 낮은 makespan을 보였습니다. 여러 현장 배치에 공통으로 통하는 정책을 일부 학습했을 가능성이 있습니다.`);}
  if(by['Nearest-lift-first']&&multi&&by['Nearest-lift-first'].makespan<multi.makespan) lines.push(`- Nearest-lift-first도 RL-multi보다 빠릅니다. 다음 단계에서는 soft penalty 강화보다 candidate outcome feature와 travel/setup/finish time 기반 선택 품질을 개선하는 것이 좋습니다.`);
  if(bestInter.name==='RL-multi'||bestInter.name==='RL-single') lines.push(`- RL 계열 정책의 soft 노출이 낮더라도, 현재 해석에서는 makespan과 hard mask 정상 작동을 우선 판단합니다.`);
  lines.push(`- 결론: 이 결과는 prototype-valid 수준의 검증이며, 연구 검증으로 확장하려면 seed 수를 늘리고 scenario 조건별 결과를 분리해 보고해야 합니다.`);
  $('interpretLog').textContent=lines.join('\n');}
function updateAll(ep){let h=history.slice(-20); let avg=h.length?h.reduce((s,x)=>s+x.reward,0)/h.length:0; $('mEpisode').textContent=ep; $('mReward').textContent=avg.toFixed(1); $('mBest').textContent=best?best.makespan+'분':'-'; let base=lastEval||best||history.at(-1); $('mDone').textContent=base?Math.round(base.done/base.total*100)+'%':'0%'; renderSummaryMetrics(base); renderAgents(base);}
function renderSummaryMetrics(r){$('mEvalMakespan').textContent=r?r.makespan+'분':'-'; $('mInter').textContent=r?`Soft ${r.softInter||0}회 / HardMask ${r.hardInter||0}회`:'-'; $('mTravel').textContent=r?r.travelTotal.toFixed(1)+'분':'-'; $('mSetup').textContent=r?r.setupTotal.toFixed(1)+'분':'-'; if($('mMove'))$('mMove').textContent=r?r.moveTotal.toFixed(1):'-';}
function renderBaselineRows(results){$('baselineRows').innerHTML=results.map(x=>`<tr><td>${x.name}</td><td>${x.r.done}/${x.r.total}</td><td>${x.r.makespan}분</td><td>${x.r.reward.toFixed(1)}</td><td>Soft ${x.r.softInter||0} / 실행Hard ${x.r.hardExecuted||0} / HardMask ${x.r.hardMask??x.r.hardInter??0}</td><td>${x.r.travelTotal.toFixed(1)}</td><td>${x.r.setupTotal.toFixed(1)}</td><td>${x.r.moveTotal.toFixed(1)}</td></tr>`).join('');}
function renderMetricLog(data,title){
  if(Array.isArray(data)){$('metricLog').textContent=`${title}\n`+data.map(x=>`${x.name}: 완료 ${x.r.done}/${x.r.total}, makespan ${x.r.makespan}분, reward ${x.r.reward.toFixed(1)}, soft노출 ${x.r.softInter||0}회 / 실행Hard ${x.r.hardExecuted||0}회 / hard마스크 ${x.r.hardMask??x.r.hardInter??0}회, soft penalty ${(x.r.interPenaltyTotal||0).toFixed(1)}, 동일 반경 연속 ${x.r.same}회, travel ${x.r.travelTotal.toFixed(1)}분, 이동거리 ${x.r.moveTotal.toFixed(1)}, setup ${x.r.setupTotal.toFixed(1)}분, lift ${x.r.liftTotal.toFixed(1)}분`).join('\n')+`\n\n해석 기준: 실제 양중반경 hard overlap은 action mask로 실행 차단됩니다. Soft는 허용 가능한 안전/영향 반경 근접 노출이므로 실패 지표가 아니라 보조 지표로 봅니다.`; return;}
  $('metricLog').textContent=`${title}\n완료: ${data.done}/${data.total}\nMakespan: ${data.makespan}분\nReward: ${data.reward.toFixed(1)}\nSoft 노출: ${data.softInter||0}회, 허용 가능한 안전/영향 반경 근접 작업\n실행 Hard overlap: ${data.hardExecuted||0}회
Hard action mask: ${data.hardMask??data.hardInter??0}회, 실제 양중반경 overlap 후보 차단\n- Soft penalty 합계: ${(data.interPenaltyTotal||0).toFixed(1)}\n동일 반경 연속: ${data.same}회\nTravel: ${data.travelTotal.toFixed(1)}분\n이동거리: ${data.moveTotal.toFixed(1)}\nSetup: ${data.setupTotal.toFixed(1)}분\nLift: ${data.liftTotal.toFixed(1)}분\n크레인별 완료시각: ${data.cranes.map(c=>`${c.id} ${Math.round(c.availableTime)}분`).join(', ')}\n\n평가 우선순위: makespan → 완료율/HardMask 정상 작동 → travel/setup/move 효율 → soft 노출 → reward`;
}
function renderAgents(r){let rows=''; let cs=(r?r.cranes:cranes); cs.forEach(c=>rows+=`<tr><td>${c.id}<br><span class="note">${c.type||'50톤 모바일 크레인'}</span></td><td>${c.done||0}</td><td>${Math.round(c.availableTime||0)}분</td><td>${Math.round(c.idle||0)}분</td><td>${(c.reward||0).toFixed(1)}</td><td>${(c.moveDistance||0).toFixed(1)}</td></tr>`); $('agentRows').innerHTML=rows;}
function renderSchedule(r){
  const cs=(r&&r.cranes)||[];
  $('scheduleLog').textContent=cs.map(c=>`${c.id}\n  ${((c.schedule||[]).join('\n  '))||'(작업 없음)'}`).join('\n\n');
}
function craneColor(i){return ['#38bdf8','#ef4444','#a78bfa','#22c55e','#f59e0b','#fb7185','#34d399'][i%7];}
function updateReplayControls(r){
  if(!$('replaySlider'))return; const base=r||lastEval||best; const maxT=base?Math.max(1,base.makespan):1;
  $('replaySlider').max=maxT; $('replaySlider').value=Math.min(replayTime,maxT);
  $('replayInfo').textContent=base?`리플레이 준비: 0~${maxT}분, event ${(base.events||[]).length}개`:'정책 평가 또는 Baseline 비교 후 리플레이를 볼 수 있습니다.';
}
function replayReset(){replayPause(); replayTime=0; const base=lastEval||best; drawReplayFrame(base,replayTime); updateReplayInfo(base,replayTime);}
function replaySeek(v){replayPause(); replayTime=+v; const base=lastEval||best; drawReplayFrame(base,replayTime); updateReplayInfo(base,replayTime);}
function replayStart(){const base=lastEval||best; if(!base||!base.events){log('리플레이할 평가 결과가 없습니다. 먼저 정책 평가 또는 Baseline 비교를 실행하세요.'); return;} replayPause(); replayPlaying=true; let last=Date.now(); replayTimer=setInterval(()=>{let now=Date.now(), speed=+$('replaySpeed').value||1; replayTime += (now-last)/1000*12*speed; last=now; if(replayTime>=base.makespan){replayTime=base.makespan; replayPause();} drawReplayFrame(base,replayTime); updateReplayInfo(base,replayTime);},80);}
function replayPause(){if(replayTimer){clearInterval(replayTimer); replayTimer=null;} replayPlaying=false;}
function craneReplayPos(base, crane, t){
  const evs=(base.events||[]).filter(e=>e.craneId===crane.id).sort((a,b)=>a.start-b.start); let x=crane.home?.x??crane.x, y=crane.home?.y??crane.y, active=null;
  for(const e of evs){
    if(t<e.start)break;
    if(t>=e.finish){x=e.toX; y=e.toY; continue;}
    active=e; const moveEnd=e.start+e.travel; if(e.travel>0 && t<moveEnd){let u=(t-e.start)/Math.max(0.001,e.travel); x=e.fromX+(e.toX-e.fromX)*u; y=e.fromY+(e.toY-e.fromY)*u;} else {x=e.toX; y=e.toY;} break;
  }
  return {x,y,active};
}
function updateReplayInfo(base,t){if(!$('replayInfo')||!base)return; $('replaySlider').value=Math.round(t); const active=(base.events||[]).filter(e=>t>=e.start&&t<e.finish).map(e=>`${e.craneId}→${e.liftId}${e.sameRadius?'(동일반경)':''}${e.interferenceLevel==='soft'?'⚠S':''}${e.interferenceLevel==='hard'?'MASK-H':''}`); $('replayInfo').textContent=`현재 ${Math.round(t)}분 / ${base.makespan}분 · 진행 중: ${active.join(', ')||'없음'} · 점선 타원=설치/작업반경, 실선 원=실제 양중반경, 주황 점선=실제 위험반경(soft buffer 포함)`;}
function drawCraneRadius(ctx,cv,sx,sy,x,y,color,label){
  const rx=cfg().craneRadius/100*cv.width, ry=cfg().craneRadius/100*cv.height;
  ctx.save();
  ctx.globalAlpha=.08; ctx.fillStyle=color; ctx.beginPath(); ctx.ellipse(sx(x),sy(y),rx,ry,0,0,Math.PI*2); ctx.fill();
  ctx.globalAlpha=.55; ctx.strokeStyle=color; ctx.lineWidth=2; ctx.setLineDash([7,5]); ctx.beginPath(); ctx.ellipse(sx(x),sy(y),rx,ry,0,0,Math.PI*2); ctx.stroke();
  ctx.setLineDash([]); ctx.globalAlpha=1; ctx.fillStyle=color; ctx.font='11px sans-serif'; ctx.fillText(`${label} 작업반경 ${cfg().craneRadius}`,sx(x)+10,sy(y)-12);
  ctx.restore();
}
function drawActualLiftRadius(ctx,cv,sx,sy,cx,cy,lx,ly,color,label='',active=false){
  const r=dist({x:cx,y:cy},{x:lx,y:ly});
  const rx=r/100*cv.width, ry=r/100*cv.height;
  ctx.save();
  ctx.globalAlpha=active?.10:.045; ctx.fillStyle=color; ctx.beginPath(); ctx.ellipse(sx(cx),sy(cy),rx,ry,0,0,Math.PI*2); ctx.fill();
  ctx.globalAlpha=active?.9:.5; ctx.strokeStyle=color; ctx.lineWidth=active?2.6:1.5; ctx.setLineDash(active?[]:[5,4]); ctx.beginPath(); ctx.ellipse(sx(cx),sy(cy),rx,ry,0,0,Math.PI*2); ctx.stroke();
  ctx.globalAlpha=active?.95:.55; ctx.lineWidth=active?2:1.2; ctx.setLineDash([]); ctx.beginPath(); ctx.moveTo(sx(cx),sy(cy)); ctx.lineTo(sx(lx),sy(ly)); ctx.stroke();
  // Explicit center marker: the actual lifting-radius circle is centered on the crane/setup point, never on the lift.
  ctx.globalAlpha=1; ctx.fillStyle=color; ctx.strokeStyle='#e5e7eb'; ctx.lineWidth=1.4;
  ctx.beginPath(); ctx.rect(sx(cx)-5,sy(cy)-5,10,10); ctx.fill(); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(sx(cx)-9,sy(cy)); ctx.lineTo(sx(cx)+9,sy(cy)); ctx.moveTo(sx(cx),sy(cy)-9); ctx.lineTo(sx(cx),sy(cy)+9); ctx.stroke();
  const mx=(cx+lx)/2, my=(cy+ly)/2;
  if(label || active){ctx.globalAlpha=.95; ctx.fillStyle=color; ctx.font='10px sans-serif'; ctx.fillText(`${label?label+' ':''}크레인중심 양중반경 ${r.toFixed(1)}`,sx(mx)+6,sy(my)-6);}
  ctx.restore();
}
function softBufferValue(){const c=cfg(); return Math.max(0,(c.safetyRadius||0)-(c.liftingRadius||0));}
function drawActualDangerRadius(ctx,cv,sx,sy,cx,cy,lx,ly,level='none',active=false){
  const liftR=dist({x:cx,y:cy},{x:lx,y:ly}), buffer=softBufferValue(), dangerR=liftR+buffer;
  if(dangerR<=0 || buffer<=0) return;
  const rx=dangerR/100*cv.width, ry=dangerR/100*cv.height;
  ctx.save();
  const dangerColor=level==='soft'?'#f59e0b':level==='hard'?'#ef4444':'#f97316';
  ctx.globalAlpha=active?.08:.035; ctx.fillStyle=dangerColor; ctx.beginPath(); ctx.ellipse(sx(cx),sy(cy),rx,ry,0,0,Math.PI*2); ctx.fill();
  ctx.globalAlpha=active?.85:.42; ctx.strokeStyle=dangerColor; ctx.lineWidth=active?2.2:1.3; ctx.setLineDash([10,5]); ctx.beginPath(); ctx.ellipse(sx(cx),sy(cy),rx,ry,0,0,Math.PI*2); ctx.stroke();
  if(active){ctx.setLineDash([]); ctx.globalAlpha=.95; ctx.fillStyle=dangerColor; ctx.font='10px sans-serif'; ctx.fillText(`실제 위험반경 ${dangerR.toFixed(1)} = 양중 ${liftR.toFixed(1)} + buffer ${buffer.toFixed(1)}`,sx(cx)+12,sy(cy)+18);}
  ctx.restore();
}
function drawRadiusLegend(ctx,cv){
  ctx.save(); ctx.font='11px sans-serif'; ctx.fillStyle='#cbd5e1'; ctx.globalAlpha=.95;
  ctx.fillText(`점선 타원=설치/작업반경 ${cfg().craneRadius}, 실선 원=실제 양중반경, 주황 점선=실제 위험반경(양중반경+soft buffer ${softBufferValue()}), 중심=크레인/setup point`,12,cv.height-12);
  ctx.restore();
}
function drawReplayFrame(base,t){
  if(!base){drawSim();return;} let cv=$('sim'),ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);ctx.fillStyle='#020617';ctx.fillRect(0,0,cv.width,cv.height);function sx(x){return x/100*cv.width} function sy(y){return y/100*cv.height}
  ctx.strokeStyle='#1e293b'; for(let i=0;i<=4;i++){ctx.beginPath();ctx.moveTo(i*cv.width/4,0);ctx.lineTo(i*cv.width/4,cv.height);ctx.stroke();ctx.beginPath();ctx.moveTo(0,i*cv.height/4);ctx.lineTo(cv.width,i*cv.height/4);ctx.stroke();}
  const activeByLift=new Map((base.events||[]).filter(e=>t>=e.start&&t<e.finish).map(e=>[e.liftId,e]));
  const doneByLift=new Map(); (base.events||[]).filter(e=>e.finish<=t).forEach(e=>doneByLift.set(e.liftId,e));
  base.lifts.forEach(l=>{let active=activeByLift.get(l.id), done=doneByLift.get(l.id), ev=active||done, color=ev?craneColor(ev.craneIndex):'#f59e0b'; ctx.globalAlpha=(done&&!active)?0.62:1; ctx.fillStyle=color;ctx.beginPath();ctx.arc(sx(l.x),sy(l.y),active?6:5,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1; ctx.fillStyle='#e5e7eb';ctx.font='10px sans-serif';ctx.fillText(l.id,sx(l.x)+6,sy(l.y)-4);});
  base.cranes.forEach((c,i)=>{let pos=craneReplayPos(base,c,t), color=craneColor(i); drawCraneRadius(ctx,cv,sx,sy,pos.x,pos.y,color,c.id); if(pos.active && t>=pos.active.start+(pos.active.travel||0)){const rcx=Number.isFinite(pos.active.radiusCenterX)?pos.active.radiusCenterX:pos.x, rcy=Number.isFinite(pos.active.radiusCenterY)?pos.active.radiusCenterY:pos.y; drawActualDangerRadius(ctx,cv,sx,sy,rcx,rcy,pos.active.liftX,pos.active.liftY,pos.active.interferenceLevel,true); drawActualLiftRadius(ctx,cv,sx,sy,rcx,rcy,pos.active.liftX,pos.active.liftY,pos.active.interferenceLevel==='soft'?'#f59e0b':color,pos.active.liftId,true);} ctx.fillStyle=color;ctx.fillRect(sx(pos.x)-7,sy(pos.y)-7,14,14);ctx.fillStyle='#e5e7eb';ctx.fillText(c.id,sx(pos.x)+10,sy(pos.y)+4);});
  ctx.fillStyle='#e5e7eb';ctx.font='13px sans-serif';ctx.fillText(`Replay ${Math.round(t)} / ${base.makespan} min`,12,20); drawRadiusLegend(ctx,cv);
}
function log(msg){let now=new Date().toLocaleTimeString(); $('log').textContent+=`[${now}] ${msg}\n`; $('log').scrollTop=$('log').scrollHeight;}
function drawSim(r){let cv=$('sim'),ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);ctx.fillStyle='#020617';ctx.fillRect(0,0,cv.width,cv.height);function sx(x){return x/100*cv.width} function sy(y){return y/100*cv.height}
  let ls=(r?r.lifts:lifts), cs=(r?r.cranes:cranes); ctx.strokeStyle='#1e293b'; for(let i=0;i<=4;i++){ctx.beginPath();ctx.moveTo(i*cv.width/4,0);ctx.lineTo(i*cv.width/4,cv.height);ctx.stroke();ctx.beginPath();ctx.moveTo(0,i*cv.height/4);ctx.lineTo(cv.width,i*cv.height/4);ctx.stroke();}
  const eventByLift=new Map(((r&&r.events)||[]).map(e=>[e.liftId,e]));
  ls.forEach(l=>{let byIdx=cs.findIndex(c=>c.id===l.by), assigned=byIdx>=0, color=l.status==='done'&&assigned?craneColor(byIdx):'#f59e0b', ev=eventByLift.get(l.id); if(assigned&&ev){const rcx=Number.isFinite(ev.radiusCenterX)?ev.radiusCenterX:ev.toX, rcy=Number.isFinite(ev.radiusCenterY)?ev.radiusCenterY:ev.toY; drawActualDangerRadius(ctx,cv,sx,sy,rcx,rcy,ev.liftX,ev.liftY,ev.interferenceLevel,false); drawActualLiftRadius(ctx,cv,sx,sy,rcx,rcy,ev.liftX,ev.liftY,color,'',false);} ctx.globalAlpha=l.status==='done'?0.85:1; ctx.fillStyle=color;ctx.beginPath();ctx.arc(sx(l.x),sy(l.y),assigned?5:5,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1; ctx.fillStyle='#cbd5e1';ctx.font='10px sans-serif';ctx.fillText(l.id,sx(l.x)+6,sy(l.y)-4);});
  cs.forEach((c,i)=>{let color=craneColor(i); drawCraneRadius(ctx,cv,sx,sy,c.x,c.y,color,c.id); ctx.fillStyle=color;ctx.fillRect(sx(c.x)-7,sy(c.y)-7,14,14);ctx.fillStyle='#e5e7eb';ctx.fillText(c.id,sx(c.x)+10,sy(c.y)+4);}); drawRadiusLegend(ctx,cv);}
function canvasPoint(evt){
  const cv=$('sim'), rect=cv.getBoundingClientRect?cv.getBoundingClientRect():{left:0,top:0,width:cv.width,height:cv.height};
  const px=(evt.clientX-rect.left)/Math.max(1,rect.width)*cv.width, py=(evt.clientY-rect.top)/Math.max(1,rect.height)*cv.height;
  return {x:Math.max(0,Math.min(100,px/cv.width*100)), y:Math.max(0,Math.min(100,py/cv.height*100))};
}
function nearestEditableItem(pt){
  const target=$('editTarget')?.value||'auto', items=[];
  if(target!=='lift') cranes.forEach((c,i)=>items.push({kind:'crane',i,x:c.x,y:c.y,id:c.id}));
  if(target!=='crane') lifts.forEach((l,i)=>items.push({kind:'lift',i,x:l.x,y:l.y,id:l.id}));
  if(!items.length)return null;
  const best=items.reduce((a,b)=>Math.hypot(pt.x-b.x,pt.y-b.y)<Math.hypot(pt.x-a.x,pt.y-a.y)?b:a,items[0]);
  return Math.hypot(pt.x-best.x,pt.y-best.y)<=8?best:null;
}
function moveEditableItem(item,pt){
  if(!item)return;
  if(item.kind==='crane'){
    const c=cranes[item.i]; c.x=pt.x; c.y=pt.y; c.home={x:pt.x,y:pt.y}; c.setupX=pt.x; c.setupY=pt.y; c.availableTime=0; c.done=0; c.idle=0; c.reward=0; c.moveDistance=0; c.schedule=[];
  } else {
    const l=lifts[item.i]; l.x=pt.x; l.y=pt.y; l.zone=Math.floor(pt.x/25)+'-'+Math.floor(pt.y/25); l.status='todo'; l.by=null; l.start=null; l.finish=null;
  }
  lastEval=null; best=null; loadCurrentScenarioToEditor(); drawSim(); renderAgents(null);
}
function setupCanvasEditing(){
  const cv=$('sim'); if(!cv||cv._editingBound)return; cv._editingBound=true;
  cv.addEventListener?.('mousedown',e=>{if(!$('editMode')?.checked)return; const pt=canvasPoint(e), item=nearestEditableItem(pt); if(item){dragItem=item; moveEditableItem(item,pt); e.preventDefault?.();}});
  cv.addEventListener?.('mousemove',e=>{if(!dragItem||!$('editMode')?.checked)return; moveEditableItem(dragItem,canvasPoint(e)); e.preventDefault?.();});
  cv.addEventListener?.('mouseup',()=>{if(dragItem){log(`${dragItem.kind==='crane'?'크레인':'양중물'} ${dragItem.id} 위치 수정 완료`);} dragItem=null;});
  cv.addEventListener?.('mouseleave',()=>{dragItem=null;});
}
function drawLineChart(canvasId, arrs){let cv=$(canvasId),ctx=cv.getContext('2d');ctx.clearRect(0,0,cv.width,cv.height);ctx.fillStyle='#020617';ctx.fillRect(0,0,cv.width,cv.height);let vals=arrs.flatMap(a=>a.data).filter(Number.isFinite); if(!vals.length)return; let min=Math.min(...vals),max=Math.max(...vals); if(min===max){min-=1;max+=1;} ctx.strokeStyle='#334155';ctx.beginPath();ctx.moveTo(35,10);ctx.lineTo(35,170);ctx.lineTo(590,170);ctx.stroke(); arrs.forEach(a=>{ctx.strokeStyle=a.color;ctx.beginPath();a.data.forEach((v,i)=>{let x=35+i/Math.max(1,a.data.length-1)*555;let y=170-(v-min)/(max-min)*150;if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y);});ctx.stroke();ctx.fillStyle=a.color;ctx.fillText(a.name,42,20+14*arrs.indexOf(a));});}
function drawCharts(){let last=history.slice(-100);drawLineChart('rewardChart',[{name:'reward',color:'#38bdf8',data:last.map(x=>x.reward)}]);drawLineChart('metricChart',[{name:'makespan',color:'#22c55e',data:last.map(x=>x.makespan)},{name:'soft exposure×20',color:'#f59e0b',data:last.map(x=>(x.softInter||0)*20)}]);}
setupCanvasEditing();
generateScenario();
loadCurrentScenarioToEditor();
