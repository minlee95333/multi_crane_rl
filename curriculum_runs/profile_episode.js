const fs=require('fs'), vm=require('vm');
const ids={}; function ctx(){return {clearRect(){},fillRect(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},arc(){},fill(){},ellipse(){},rect(){},fillText(){},save(){},restore(){},setLineDash(){}};}
function elem(id){if(!ids[id])ids[id]={id,value:'',textContent:'',innerHTML:'',disabled:false,scrollTop:0,scrollHeight:0,width:900,height:430,getContext:ctx};return ids[id];}
const vals={numCranes:4,numLifts:36,craneType:'50톤 모바일 크레인',fixedDuration:25,setupTime:10,craneRadius:18,episodes:1,maxSteps:320,alpha:0.18,gamma:0.92,epsStart:0.35,epsEnd:0.05,seedRuns:15,baseSeed:301,learningMode:'multi',learningAlgorithm:'mappo',actorShareMode:'typeShared',candidateK:6,actorLr:0.001,criticLr:0.002,gaeLambda:0.95,ppoClip:0.20,rSingle:10,rAll:100,rSame:3,pIdle:-0.5,pInterSoft:-3,pInterHard:-15,liftingRadius:8,safetyRadius:14,pTime:-0.1,pMove:-0.02};
for(const [k,v] of Object.entries(vals)) elem(k).value=String(v);
['log','metricLog','interpretLog','baselineRows','repeatRows','agentRows','scheduleLog','mEpisode','mReward','mBest','mDone','mEvalMakespan','mInter','mTravel','mSetup','mMove','trainBtn','sim','rewardChart','metricChart','replaySlider','replayInfo','replaySpeed'].forEach(elem); elem('replaySpeed').value='1';
const context={console,Math,Date,setInterval,clearInterval,setTimeout,document:{getElementById:elem,createElement:()=>({click(){},remove(){}}),querySelectorAll:()=>[]},body:{appendChild(){}}}; context.Blob=function(){}; context.URL={createObjectURL:()=>'',revokeObjectURL(){}}; context.global=context; vm.createContext(context); vm.runInContext(fs.readFileSync('/tmp/crane_script_check.js','utf8'),context);
const code=`(()=>{
 const names=['candidateActions','candidateOutcome','candidateCircleRisk','candidateFeatureVector','stateKey','mappoDistribution','enforceMaskedChoices','globalFeatures','makePlan','hasHardOverlap'];
 const stat={}; for(const n of names){const old=globalThis[n]; stat[n]={n:0,t:0}; globalThis[n]=function(...args){const s=Date.now(); try{return old.apply(this,args);} finally{stat[n].n++; stat[n].t+=Date.now()-s;}}}
 initMappoModel(); generateScenario(301,true,true); const t=Date.now(); const r=runEpisode(1,true,0.1,'mappo'); return {total:Date.now()-t, done:r.done, stat};
})()`;
console.log(JSON.stringify(vm.runInContext(code,context),null,2));
