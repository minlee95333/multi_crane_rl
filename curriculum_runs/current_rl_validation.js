const fs=require('fs'), vm=require('vm');
const ids={};
function ctx(){return {clearRect(){},fillRect(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},arc(){},fill(){},ellipse(){},rect(){},fillText(){},save(){},restore(){},setLineDash(){},closePath(){},strokeStyle:'',fillStyle:'',font:'',globalAlpha:1,lineWidth:1};}
function elem(id){if(!ids[id])ids[id]={id,value:'',textContent:'',innerHTML:'',disabled:false,checked:false,scrollTop:0,scrollHeight:0,width:900,height:430,style:{},className:'',getContext:ctx,appendChild(){},remove(){},click(){},addEventListener(){},setAttribute(){},querySelectorAll(){return []}};return ids[id];}
const cfgVals={
  numCranes:3,numLifts:24,craneType:'50톤 모바일 크레인',fixedDuration:25,setupTime:10,craneRadius:18,
  episodes:Number(process.argv[2]||150),maxSteps:220,alpha:0.18,gamma:0.92,epsStart:0.35,epsEnd:0.05,
  seedRuns:10,baseSeed:101,learningMode:'multi',learningAlgorithm:'mappo',actorShareMode:'typeShared',candidateK:5,
  actorLr:0.001,criticLr:0.002,gaeLambda:0.95,ppoClip:0.20,
  rSingle:10,rAll:100,rSame:3,pIdle:-0.5,pInterSoft:-3,pInterHard:-15,liftingRadius:8,safetyRadius:14,pTime:-0.1,pMove:-0.02,
  planningSeed:501,planningPolicy:'mappo',planningCranes:3,planningLifts:24,planningK:5,planningMaxSteps:260
};
for(const [k,v] of Object.entries(cfgVals)) elem(k).value=String(v);
['log','metricLog','interpretLog','baselineRows','repeatRows','agentRows','scheduleLog','mEpisode','mReward','mBest','mDone','mEvalMakespan','mInter','mTravel','mSetup','mMove','trainBtn','sim','rewardChart','metricChart','replaySlider','replayInfo','replaySpeed','planOutput','planRows','planEvents','siteText'].forEach(elem); elem('replaySpeed').value='1';
const context={console,Math,Date,setInterval,clearInterval,setTimeout,clearTimeout,document:{getElementById:elem,createElement:(tag)=>elem('created_'+tag+'_'+Math.random()),querySelectorAll:()=>[],addEventListener(){}},window:{addEventListener(){},removeEventListener(){}},body:{appendChild(){}}};
context.Blob=function(){}; context.URL={createObjectURL:()=>'',revokeObjectURL(){}}; context.global=context; vm.createContext(context);
vm.runInContext(fs.readFileSync(__dirname+'/crane_script.js','utf8'),context);
vm.runInContext('log=function(){}; drawAll=function(){}; drawSim=function(){}; drawChart=function(){}; updateAll=function(){};', context);
function run(code){return vm.runInContext(code,context);}
const result = run(`(()=>{
  function avg(a){return a.reduce((x,y)=>x+y,0)/Math.max(1,a.length)}
  function sd(a){const m=avg(a);return Math.sqrt(avg(a.map(x=>(x-m)*(x-m))))}
  function pctComplete(r){return r.total?100*r.done/r.total:0}
  function summ(samples){
    const v=k=>samples.map(s=>s[k]??0);
    return {n:samples.length,completeRate:avg(samples.map(pctComplete)),makespan:avg(v('makespan')),makespanSd:sd(v('makespan')),reward:avg(v('reward')),soft:avg(v('softInter')),hardMask:avg(v('hardMask')),travel:avg(v('travelTotal')),setup:avg(v('setupTotal')),move:avg(v('moveTotal')),actualLiftRadius:avg(v('actualLiftRadiusAvg')),actualDangerRadius:avg(v('actualDangerRadiusAvg'))};
  }
  function evalPolicyRange(policy, seedStart, seedCount, model){
    const samples=[];
    for(let i=0;i<seedCount;i++){
      const seed=seedStart+i;
      generateScenario(seed,true,true);
      if(policy==='mappo' && model) restoreMappoModel(model);
      const r=runEpisode(0,false,0,policy);
      samples.push(r);
    }
    return summ(samples);
  }
  const c=cfg();
  initMappoModel();
  const beforeModel=cloneMappoModel();
  const beforeSeen=evalPolicyRange('mappo',101,10,beforeModel);
  const beforeUnseen=evalPolicyRange('mappo',201,30,beforeModel);
  initMappoModel();
  const t0=Date.now();
  let train=[]; let best=null;
  for(let ep=1; ep<=c.episodes; ep++){
    const trainSeed=(c.baseSeed||101)+((ep-1)%Math.max(1,c.seedRuns||1));
    generateScenario(trainSeed,true,true);
    setSeed(trainSeed*3000+ep);
    const eps=c.epsStart+(c.epsEnd-c.epsStart)*(ep/c.episodes);
    const r=runEpisode(ep,true,eps,'mappo');
    train.push({ep,seed:trainSeed,done:r.done,total:r.total,makespan:r.makespan,reward:r.reward,soft:r.softInter||0,hardMask:r.hardMask||0,move:r.moveTotal||0});
    if(r.done===r.total && (!best || r.makespan<best.makespan)) best={ep,seed:trainSeed,makespan:r.makespan,reward:r.reward,soft:r.softInter||0,hardMask:r.hardMask||0,move:r.moveTotal||0};
  }
  const model=cloneMappoModel();
  const seen={
    'RL-before':beforeSeen,
    'MAPPO-trained':evalPolicyRange('mappo',101,10,model),
    'Nearest':evalPolicyRange('nearest',101,10),
    'Same-radius':evalPolicyRange('radiusPriority',101,10),
    'Random':evalPolicyRange('random',101,10)
  };
  const unseen={
    'RL-before':beforeUnseen,
    'MAPPO-trained':evalPolicyRange('mappo',201,30,model),
    'Nearest':evalPolicyRange('nearest',201,30),
    'Same-radius':evalPolicyRange('radiusPriority',201,30),
    'Random':evalPolicyRange('random',201,30)
  };
  const recent=train.slice(-20);
  return {config:c, elapsedSec:(Date.now()-t0)/1000, trainEpisodes:train.length, trainSummary:{best, first10:summ(train.slice(0,10).map(x=>({done:x.done,total:x.total,makespan:x.makespan,reward:x.reward,softInter:x.soft,hardMask:x.hardMask,moveTotal:x.move}))), last20:summ(recent.map(x=>({done:x.done,total:x.total,makespan:x.makespan,reward:x.reward,softInter:x.soft,hardMask:x.hardMask,moveTotal:x.move}))), modelStats:model.stats}, seen, unseen};
})()`);
const out=process.argv[3]||(__dirname+'/current_rl_validation_result.json');
fs.writeFileSync(out, JSON.stringify(result,null,2));
console.log(JSON.stringify(result,null,2));
