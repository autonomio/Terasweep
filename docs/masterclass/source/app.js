'use strict';
const DATA=JSON.parse(document.getElementById('course-data').textContent);
const $=(q,r=document)=>r.querySelector(q), $$=(q,r=document)=>Array.from(r.querySelectorAll(q));
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(x,d=0)=>Number(x).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const sizes=[100000,1000000,10000000,100000000,1000000000], sizeNames=['100k','1m','10m','100m','1b'];
const levels=['original','bounds','compact','intrinsics','assembly'];
const labels={limen:'Native Limen CLI / UEL',naive:'Rust · no cross-trial reuse',stats:'+ Shared Gram statistics',train:'+ Fitted predictions',score:'+ Original score cache',original:'Original factorized Rust',bounds:'+ Constant-signal shortcut',compact:'+ Compact scoring cache',intrinsics:'+ Rust AVX-512 intrinsics',assembly:'+ Handwritten AVX-512'};
const colors={original:'var(--off)',bounds:'var(--gold)',compact:'var(--blue)',intrinsics:'var(--green)',assembly:'var(--purple)',limen:'var(--gold)',naive:'var(--off)',stats:'var(--blue)',train:'var(--green)',score:'var(--purple)'};
const storageKey='terasweep-masterclass-v1';
let storageOK=true,state={chapter:'brief',done:[],notes:{},quiz:{},claims:{},theme:'dark',capstone:''};
try{let old=JSON.parse(localStorage.getItem(storageKey)||'null');if(old&&typeof old==='object')state={...state,...old};}catch(e){storageOK=false;}
state.done=Array.isArray(state.done)?state.done:[];state.notes=state.notes||{};state.quiz=state.quiz||{};state.claims=state.claims||{};
function save(){try{localStorage.setItem(storageKey,JSON.stringify(state));}catch(e){storageOK=false;}$('#note-status').textContent=storageOK?'Saved in this browser only.':'Local storage unavailable. Export your learning record to keep it.';}
function toast(text){const t=$('#toast');t.textContent=text;t.hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>t.hidden=true,2800);}
function download(name,text,type='text/plain'){let a=document.createElement('a'),u=URL.createObjectURL(new Blob([text],{type}));a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}
function strip(s){const d=document.createElement('div');d.innerHTML=s.replace(/<br\s*\/?>/gi,' ');return d.textContent||'';}
function metric(value,label,qual='',cls=''){return `<div class="metric ${cls}">${qual?`<small>${qual}</small>`:''}<strong>${value}</strong><span>${label}</span></div>`;}
function progress(){
 $$('.nav-item').forEach(b=>{$('.nav-done',b).textContent=state.done.includes(b.dataset.go)?'✓':'';b.classList.toggle('active',b.dataset.go===state.chapter);b.setAttribute('aria-current',b.dataset.go===state.chapter?'step':'false');});
 $('#progress-count').textContent=`${state.done.length} / ${DATA.chapters.length}`;$('#progress-fill').style.width=`${state.done.length/DATA.chapters.length*100}%`;
 const done=state.done.includes(state.chapter);$('#mastered').classList.toggle('done',done);$('#mastered').textContent=done?'✓ Marked understood':'Mark this chapter understood';$('#mastered').setAttribute('aria-pressed',String(done));
}
let laneTimer=null;
function stopLanes(){if(laneTimer){clearInterval(laneTimer);laneTimer=null;}$('#lane-play').textContent='Play timeline';}
function navigate(id,scroll=true){
 if(!DATA.chapters.some(c=>c.id===id))id='brief';
 stopLanes();state.chapter=id;save();
 $$('.lesson').forEach(s=>s.classList.toggle('active',s.id===id));
 const idx=DATA.chapters.findIndex(c=>c.id===id);$('#prev-lesson').disabled=idx===0;$('#next-lesson').disabled=idx===DATA.chapters.length-1;
 $('#next-lesson').textContent=idx===DATA.chapters.length-1?'Course complete':'Next chapter →';
 $('#lesson-note').value=state.notes[id]||'';$('#note-label').textContent=`YOUR FIELD NOTES / ${String(idx).padStart(2,'0')}`;
 $('#chapter-counter').textContent=`${String(idx).padStart(2,'0')} / 12`;document.title=`${strip(DATA.chapters[idx].title)} — Terasweep Masterclass`;
 try{history.replaceState(null,'','#'+id);}catch(e){}
 $('#sidebar').classList.remove('open');$('#menu-btn').setAttribute('aria-expanded','false');
 progress();if(scroll){window.scrollTo({top:0,behavior:'instant'});$('.chapter-header h1',$('#'+id)).focus({preventScroll:true});}
 for(const dialog of $$('dialog'))if(dialog.open)dialog.close();
}
document.documentElement.dataset.theme=state.theme;
$('#theme-btn').addEventListener('click',()=>{state.theme=state.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=state.theme;save();$('#theme-btn').setAttribute('aria-label','Switch to '+(state.theme==='dark'?'light':'dark')+' theme');});
$('#menu-btn').addEventListener('click',()=>{const open=$('#sidebar').classList.toggle('open');$('#menu-btn').setAttribute('aria-expanded',String(open));});
$('#mastered').addEventListener('click',()=>{if(state.done.includes(state.chapter))state.done=state.done.filter(x=>x!==state.chapter);else state.done.push(state.chapter);save();progress();});
$('#prev-lesson').addEventListener('click',()=>navigate(DATA.chapters[Math.max(0,DATA.chapters.findIndex(c=>c.id===state.chapter)-1)].id));
$('#next-lesson').addEventListener('click',()=>navigate(DATA.chapters[Math.min(DATA.chapters.length-1,DATA.chapters.findIndex(c=>c.id===state.chapter)+1)].id));
$('#lesson-note').addEventListener('input',e=>{state.notes[state.chapter]=e.target.value;save();});
$('#reset-progress').addEventListener('click',()=>{if(confirm('Reset chapter checkmarks and quiz choices? Your written notes will be kept.')){state.done=[];state.quiz={};state.claims={};save();progress();$$('.quiz-options button').forEach(b=>b.classList.remove('correct','incorrect'));$$('.quiz-feedback').forEach(p=>p.hidden=true);renderClaims();toast('Progress reset. Written notes retained.');}});
window.addEventListener('hashchange',()=>navigate(location.hash.slice(1)));
document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT','BUTTON','SUMMARY','A'].includes(document.activeElement.tagName)||$$('dialog[open]').length)return;if(e.key==='ArrowRight')$('#next-lesson').click();if(e.key==='ArrowLeft')$('#prev-lesson').click();if(e.key==='/'){e.preventDefault();openDialog('search');}});
document.addEventListener('click',e=>{
 const go=e.target.closest('[data-go]');if(go){navigate(go.dataset.go);return;}
 const ref=e.target.closest('[data-source]');if(ref){openSource(ref.dataset.source,ref.dataset.lines||'');return;}
 const op=e.target.closest('[data-open]');if(op){openDialog(op.dataset.open);return;}
 const close=e.target.closest('[data-close]');if(close){close.closest('dialog').close();return;}
 const cb=e.target.closest('.copy-code');if(cb){const text=$('pre',cb.closest('.code-card')).textContent;copyText(text);}
});
async function copyText(text){try{await navigator.clipboard.writeText(text);toast('Commands copied.');}catch(e){const t=document.createElement('textarea');t.value=text;document.body.appendChild(t);t.select();const ok=document.execCommand('copy');t.remove();toast(ok?'Commands copied.':'Select and copy the command block manually.');}}
function openDialog(id){const d=$('#'+id+'-dialog');if(!d.open)d.showModal();if(id==='evidence'){renderSourceList();if(!currentSource)selectSource('scaling');}if(id==='search'){$('#course-search').focus();renderSearch();}}
for(const d of $$('dialog'))d.addEventListener('click',e=>{if(e.target===d){const r=d.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)d.close();}});
let currentSource=null;
function renderSourceList(){const q=$('#source-search').value.toLowerCase();$('#source-buttons').innerHTML=Object.entries(DATA.sources).filter(([k,v])=>(v.title+' '+v.path).toLowerCase().includes(q)).map(([k,v])=>`<button class="source-btn ${k===currentSource?'active':''}" data-source="${k}">${esc(v.title)}<small>${esc(v.kind)}</small></button>`).join('')||'<p class="fine">No matching evidence.</p>';}
function selectSource(id,lines=''){
 const s=DATA.sources[id];if(!s)return;currentSource=id;$('#source-title').textContent=s.title;$('#source-meta').textContent=s.path+' · SHA-256 '+s.sha256;
 $('#source-git').href=s.url;$('#source-git').hidden=!s.url;$('#source-note').textContent=s.note||'Retained source snapshot. Line numbers below refer to this embedded file. No external request is needed to read it.';
 let [a,b]=lines.split('-').map(Number);b=b||a;
 $('#source-text').innerHTML=s.text.split('\n').map((line,i)=>`<div class="source-line ${a&&i+1>=a&&i+1<=b?'highlight':''}" id="source-line-${i+1}"><span class="line-no">${i+1}</span><span>${esc(line)||' '}</span></div>`).join('');
 renderSourceList();requestAnimationFrame(()=>{if(a&&$('#source-line-'+a))$('#source-line-'+a).scrollIntoView({block:'start',behavior:'instant'});else $('.source-main').scrollTop=0;});
}
function openSource(id,lines=''){openDialog('evidence');selectSource(id,lines);}
$('#source-search').addEventListener('input',renderSourceList);
$('#source-export').addEventListener('click',()=>{const s=DATA.sources[currentSource];if(s)download(s.path.split('/').pop(),s.text);});
$('#course-search').addEventListener('input',renderSearch);
function renderSearch(){const q=$('#course-search').value.toLowerCase();const match=DATA.chapters.filter(c=>(strip(c.title+' '+c.lead+' '+c.body)).toLowerCase().includes(q));$('#search-results').innerHTML=match.map(c=>`<button class="search-result" data-go="${c.id}"><b>${strip(c.title)}</b><span>${esc(c.lead)}</span></button>`).join('')||'<p>No lesson matches that phrase.</p>';}
for(const q of $$('.checkpoint')){
 const id=q.closest('.lesson').id,explanation=$('.quiz-feedback',q).textContent;
 function answer(n){const correct=Number(q.dataset.correct);$$('[data-choice]',q).forEach(b=>{b.classList.toggle('correct',Number(b.dataset.choice)===correct);b.classList.toggle('incorrect',Number(b.dataset.choice)===n&&n!==correct);});const p=$('.quiz-feedback',q);p.hidden=false;p.textContent=(n===correct?'Yes. ':'Not quite. ')+explanation;state.quiz[id]=n;save();}
 q.addEventListener('click',e=>{const b=e.target.closest('[data-choice]');if(b)answer(Number(b.dataset.choice));});if(id in state.quiz)answer(state.quiz[id]);
}
const workloads={
 six:{title:'Initial matched execution · M1 Max',dims:['6,521 × 22','719 × 22','6'],labels:['Training matrix','Test matrix','Requested configurations'],text:'8,760 frozen BTCUSDT hourly observations from 2025. Three alphas × two intercept settings; one positive-prediction rule, threshold zero, 5 bps fee + 5 bps slippage. Five separate-process repetitions. Both outputs repeated reliably; full economic equivalence was not established.',source:'initial'},
 grid:{title:'Native-grid factorization study · Intel Xeon 8370C',dims:['1,903 × 13','1,431 × 13','1,014'],labels:['Training matrix','Test matrix','Requested configurations'],text:'Retained Limen real-market fixture: 4,585 raw observations. January–February 2026 date splits. Three alphas × two intercept settings × 13 fees × 13 slippages. An exact native GridStrategy plan is replayed in Rust. Six unique model configurations do not imply six native fitting calls.',source:'funnel'},
 original:{title:'Original frozen sweep · multiple studies, each independently timed',dims:['4,136 × 55','1,136 × 55','100k → 1b'],labels:['Training matrix','Test matrix','Scaling study sizes'],text:'Original 15-minute prepared workload, seed 20260514. 384 alphas × two intercept settings × three pivot cutoffs; six signal rules; signed and absolute threshold grids; 13 × 13 cost pairs. The AVX-512 and scaling work uses this richer sweep, not the six-case or 1,014-row workload.',source:'scaling'}
};
function renderWorkload(k){const w=workloads[k];$$('[data-workload]').forEach(b=>b.classList.toggle('selected',b.dataset.workload===k));$('#workload-content').innerHTML=`<h3>${w.title}</h3><div class="stat-strip">${w.dims.map((x,i)=>`<div><strong>${x}</strong><span>${w.labels[i]}</span></div>`).join('')}</div><p>${w.text}</p><button class="ref" data-source="${w.source}">Open this experiment’s record ↗</button>`;}
$('#workload-tabs').addEventListener('click',e=>{let b=e.target.closest('[data-workload]');if(b)renderWorkload(b.dataset.workload);});renderWorkload('six');
function svgBox(content,w=700,h=300,label='Interactive teaching diagram'){return `<svg class="chart-svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}">${content}</svg>`;}
function renderRidge(){
 const x=[0,1,2,3,4,5,6,7],y=[1.4,2.1,2.7,4.6,4.9,6.2,7.5,7.9],a=Math.pow(10,Number($('#ridge-alpha').value)),inter=$('#ridge-intercept').checked;
 const xm=inter?x.reduce((p,v)=>p+v,0)/x.length:0,ym=inter?y.reduce((p,v)=>p+v,0)/y.length:0;
 const g=x.reduce((p,v)=>p+(v-xm)**2,0),rhs=x.reduce((p,v,i)=>p+(v-xm)*(y[i]-ym),0),w=rhs/(g+a),b=inter?ym-xm*w:0;
 const W=420,H=275,l=35,r=18,t=16,bot=32,px=v=>l+v/7*(W-l-r),py=v=>H-bot-v/9*(H-t-bot);let s='';
 for(let v=0;v<=8;v+=2)s+=`<line class="grid" x1="${l}" x2="${W-r}" y1="${py(v)}" y2="${py(v)}"/><text x="${l-9}" y="${py(v)+4}" text-anchor="end">${v}</text>`;
 s+=`<line x1="${px(0)}" x2="${px(7)}" y1="${py(b)}" y2="${py(7*w+b)}" stroke="var(--green)" stroke-width="2.5"/>`;
 for(let i=0;i<x.length;i++)s+=`<circle cx="${px(x[i])}" cy="${py(y[i])}" r="4.5" fill="var(--blue)"><title>Teaching point: x=${x[i]}, y=${y[i]}</title></circle><text x="${px(i)}" y="${H-11}" text-anchor="middle">${i}</text>`;
 s+=`<text class="axis-label" x="${W-15}" y="${H-11}">x</text>`;
 $('#ridge-chart').innerHTML=svgBox(s,W,H,'Illustrative ridge regression points and fitted line');
 $('#ridge-metrics').innerHTML=metric(a<.01?a.toExponential(2):fmt(a,3),'Regularization α')+metric(fmt(g,3),'Reusable G = Σ(x − x̄)²')+metric(fmt(w,4),'Fitted coefficient w')+metric(fmt(b,4),'Recovered intercept b');
 $('#ridge-explanation').textContent=`h = ${fmt(rhs,4)}; solve w = h / (G + α). Moving α changes the denominator, not G or h. ${inter?'Centered':'Raw'} statistics selected. The displayed line is computed live, not a stored benchmark output.`;
}
$('#ridge-alpha').addEventListener('input',renderRidge);$('#ridge-intercept').addEventListener('change',renderRidge);renderRidge();
const depNotes={alpha:'A model-parameter change invalidates the fitted predictions and downstream scores. With fixed input data, both raw and centered training statistics remain reusable.',threshold:'A rule or threshold change invalidates the signal path and its base score, but cannot change already fitted coefficients or predictions.',cost:'A fee or slippage change affects only downstream cost-adjusted metrics and selection under the retained cost-independent signal convention.',data:'Changing features, target, split, scaler or training window changes the input identity. All subsequent stages must be reconsidered.'};
function renderDep(key){const start={data:0,alpha:1,threshold:2,cost:3}[key],names=['Statistics','Fit & predict','Signal / base score','Apply costs','Select & report'];$('#dep-flow').innerHTML=names.map((n,i)=>`<div class="dep-node ${i>=start?'hot':''}"><b>${n}</b><span>${i>=start?'Reconsider':'Reuse'}</span></div>`).join('');$('#dep-explain').textContent=depNotes[key];$$('[data-dep]').forEach(b=>b.classList.toggle('selected',b.dataset.dep===key));}
$('#dep-controls').addEventListener('click',e=>{const b=e.target.closest('[data-dep]');if(b)renderDep(b.dataset.dep);});renderDep('alpha');
const toy={open:[100,101,101.5,100.5,102,101.5,103,102,103.5,102,104,104.5],close:[101,101.5,100.5,102,101.5,103,102,103.5,102,104,104.5,105],pred:[-.20,.12,.25,-.05,.31,.10,-.13,.18,-.06,.22,.15,-.08],trad:[true,true,true,true,true,true,false,true,true,true,true,true]};
function active(p,t,r){switch(r){case'gt':return p>t;case'gte':return p>=t;case'lt':return p<t;case'lte':return p<=t;case'abs_gt':return Math.abs(p)>t;case'abs_gte':return Math.abs(p)>=t;default:throw new Error('Unknown rule '+r);}}
function toyPath(t=0,r='gt'){
 let prev=false,gross=1,entryCount=0,exitCount=0,eligible=0;const cats=Array.from({length:4},()=>({n:0,s:0,q:0})),path=[],sig=toy.pred.map(p=>active(p,t,r));
 for(let i=0;i<toy.pred.length;i++){
  const pos=i>0&&toy.trad[i]?sig[i-1]:false,next=i+1<toy.pred.length&&toy.trad[i+1]?sig[i]:false,en=pos&&!prev,co=pos&&prev,ex=pos&&!next;
  let ret=0;if(en){ret+=(toy.close[i]-toy.open[i])/toy.open[i];entryCount++;}if(co)ret+=toy.close[i]/toy.close[i-1]-1;if(ex)exitCount++;const a=1+ret,c=(en?1:0)|(ex?2:0);gross*=a;
  if(i>0&&toy.trad[i]){cats[c].n++;cats[c].s+=a;cats[c].q+=a*a;eligible++;}
  path.push({i,pos,next,en,co,ex,a,c,eq:gross,signal:sig[i]});prev=pos;
 }
 return{gross,entryCount,exitCount,eligible,cats,path,sig};
}
function renderCosts(){
 const fee=Number($('#fee').value),slip=Number($('#slip').value);$('#fee-out').value=fee;$('#slip-out').value=slip;const f=fee/10000,s=slip/10000,u=(1-f)/(1+s),v=(1-f)*(1-s),ks=[1,u,v,u*v],p=toyPath();
 const net=p.gross*Math.pow(u,p.entryCount)*Math.pow(v,p.exitCount);let direct=1,sum=0,ss=0,ds=0,dss=0;
 for(const a of p.path){const ret=a.a*ks[a.c]-1;direct*=1+ret;if(a.i>0&&toy.trad[a.i]){ds+=ret;dss+=ret*ret;}}
 p.cats.forEach((c,i)=>{sum+=ks[i]*c.s-c.n;ss+=ks[i]*ks[i]*c.q-2*ks[i]*c.s+c.n;});
 const mean=sum/p.eligible,variance=Math.max(0,(ss-p.eligible*mean*mean)/(p.eligible-1)),sharpe=variance>0?mean/Math.sqrt(variance):NaN;
 $('#path-strip').innerHTML=p.path.map(a=>`<div class="path-bar ${a.pos?'held':''}"><b>${a.i+1}</b><span>${toy.trad[a.i]?(a.pos?'LONG':'FLAT'):'GAP'}</span><br><em>${a.en&&a.ex?'E+X':a.en?'ENTRY':a.ex?'EXIT':'·'}</em></div>`).join('');
 const names=['Neither entry nor exit','Entry only','Exit only','Entry and exit'];$('#cost-categories').innerHTML=p.cats.map((c,i)=>`<tr><td>${names[i]}</td><td>${c.n}</td><td>${fmt(c.s,6)}</td><td>${fmt(c.q,6)}</td><td>${fmt(ks[i],7)}</td></tr>`).join('');
 $('#cost-output').innerHTML=metric(fmt((net-1)*100,5)+'%','Factored net return')+metric(fmt((direct-1)*100,5)+'%','Direct per-bar compounding')+metric(fmt(sharpe,5),'Factored per-bar Sharpe')+metric(`${p.entryCount} / ${p.exitCount}`,'Entries / exits');
 $('#cost-proof').textContent=`Absolute equity difference: ${Math.abs(net-direct).toExponential(2)}. First-moment difference: ${Math.abs(sum-ds).toExponential(2)}; second-moment difference: ${Math.abs(ss-dss).toExponential(2)}. Algebraically equivalent paths can differ by floating-point roundoff; this teaching lab does not claim bitwise equivalence between a regrouped formula and direct compounding. Production parity tests compare each optimized result with the original factored scorer.`;
}
$('#fee').addEventListener('input',renderCosts);$('#slip').addEventListener('input',renderCosts);renderCosts();
const boundPred=[-.30,-.10,-.02,0,.04,.10,.18,.28];
function renderBounds(){
 const r=$('#bounds-rule').value,t=Number($('#threshold').value),arr=boundPred.slice();if($('#nonfinite').checked)arr[3]=NaN;$('#threshold-out').value=t.toFixed(2);const finite=arr.every(Number.isFinite),lo=Math.min(...arr),hi=Math.max(...arr),alo=Math.min(...arr.map(Math.abs)),ahi=Math.max(...arr.map(Math.abs));let off=false,on=false,proof='';
 switch(r){case'gt':off=hi<=t;on=lo>t;proof=off?'max(p) ≤ t':on?'min(p) > t':'';break;case'gte':off=hi<t;on=lo>=t;proof=off?'max(p) < t':on?'min(p) ≥ t':'';break;case'lt':off=lo>=t;on=hi<t;proof=off?'min(p) ≥ t':on?'max(p) < t':'';break;case'lte':off=lo>t;on=hi<=t;proof=off?'min(p) > t':on?'max(p) ≤ t':'';break;case'abs_gt':off=ahi<=t;on=alo>t;proof=off?'max(|p|) ≤ t':on?'min(|p|) > t':'';break;case'abs_gte':off=ahi<t;on=alo>=t;proof=off?'max(|p|) < t':on?'min(|p|) ≥ t':'';}
 const mask=arr.map(p=>active(p,t,r));$('#prediction-chips').innerHTML=arr.map((p,i)=>`<div class="prediction-chip ${mask[i]?'active':''}">${Number.isFinite(p)?p.toFixed(2):'NaN'}<small>${mask[i]?'ON':'OFF'}</small></div>`).join('');
 $('#bounds-answer').innerHTML=!finite?'<strong>Scalar fallback.</strong> The production bounds shortcut is disabled for a non-finite vector. NaN comparisons above are false, but the implementation does not use the finite-vector shortcut proof.':off?`<strong>All off — reuse constant slot 0.</strong> Proven by ${esc(proof)}. No full score scan is needed after the first occurrence.`:on?`<strong>All on — reuse constant slot 1.</strong> Proven by ${esc(proof)}. No full score scan is needed after the first occurrence.`:`<strong>Mixed — ${mask.filter(Boolean).length} of 8 signals active.</strong> This path needs its own base score unless its key is already cached.`;
}
$('#threshold').addEventListener('input',renderBounds);$('#bounds-rule').addEventListener('change',renderBounds);$('#nonfinite').addEventListener('change',renderBounds);
$('#bounds-min').onclick=()=>{$('#threshold').value=$('#bounds-rule').value.startsWith('abs')?0:-.3;renderBounds();};$('#bounds-max').onclick=()=>{$('#threshold').value=$('#bounds-rule').value.startsWith('abs')?.3:.28;renderBounds();};$('#bounds-zero').onclick=()=>{$('#threshold').value=0;renderBounds();};renderBounds();
function memoryText(b){if(b>=1024**3)return fmt(b/1024**3,2)+' GiB';if(b>=1024**2)return fmt(b/1024**2,2)+' MiB';return fmt(b/1024,1)+' KiB';}
function renderMemory(){const i=Number($('#memory-n').value),n=sizes[i],cap=Math.min(n,1048576),blocks=Math.ceil(n/1048576);$('#memory-size').value=sizeNames[i];$('#memory-metrics').innerHTML=metric(memoryText(n*8),'One reference for every requested row','HYPOTHETICAL')+metric(memoryText(cap*8),'Bounded row-reference buffer','DESIGN CALCULATION')+metric(fmt(blocks),'Scheduler blocks to process')+metric(fmt(cap),'Maximum row references held at once');$('#block-animation').innerHTML=`<span>Block 1</span><i>→</i><span>score cache survives</span>${blocks>1?`<i>→</i><span>Block ${fmt(blocks)}</span>`:''}`;}
$('#memory-n').addEventListener('input',renderMemory);renderMemory();
const laneJobs=[['gt',0],['gte',.12],['lt',0],['lte',-.05],['abs_gt',.10],['abs_gte',.18],['gt',.20],['lte',.25]];
function renderLanes(){const i=Number($('#lane-time').value),paths=laneJobs.map(([r,t])=>toyPath(t,r)),rows=[['Rule',laneJobs.map(x=>x[0])],['Threshold',laneJobs.map(x=>x[1].toFixed(2))],['Signal at t',paths.map(p=>p.path[i].signal)],['Position at t',paths.map(p=>p.path[i].pos)],['Entry',paths.map(p=>p.path[i].en)],['Exit',paths.map(p=>p.path[i].ex)],['Gross equity',paths.map(p=>fmt(p.path[i].eq,6))]];
 $('#lane-board').innerHTML=`<table><thead><tr><th>Independent jobs</th>${laneJobs.map((_,j)=>`<th>Lane ${j}</th>`).join('')}</tr></thead><tbody>${rows.map(([name,values])=>`<tr><th>${name}</th>${values.map(x=>`<td class="${typeof x==='boolean'?(x?'bit-on':'bit-off'):''}">${typeof x==='boolean'?(x?'1':'0'):x}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
 $('#lane-status').innerHTML=`<strong>Bar ${i+1} / 12</strong> · Broadcast prediction ${toy.pred[i].toFixed(2)} · ${toy.trad[i]?'tradable':'not tradable'}. ${i===0?'No prior prediction exists; all positions start flat.':i===11?'Last bar: every remaining position receives its terminal exit flag.':'Each lane advances one chronological step; no lane combines different times.'}`;
 $('#lane-back').disabled=i===0;$('#lane-next').disabled=i===11;
}
$('#lane-time').addEventListener('input',()=>{stopLanes();renderLanes();});$('#lane-back').onclick=()=>{stopLanes();$('#lane-time').value=Math.max(0,Number($('#lane-time').value)-1);renderLanes();};$('#lane-next').onclick=()=>{stopLanes();$('#lane-time').value=Math.min(11,Number($('#lane-time').value)+1);renderLanes();};
$('#lane-play').onclick=()=>{if(laneTimer){stopLanes();return;}if(Number($('#lane-time').value)===11)$('#lane-time').value=0;$('#lane-play').textContent='Pause timeline';laneTimer=setInterval(()=>{let i=Number($('#lane-time').value);if(i>=11){stopLanes();return;}$('#lane-time').value=i+1;renderLanes();},900);};renderLanes();
function sourceBetween(source,a,b){const t=DATA.sources[source].text;const start=t.indexOf(a),end=t.indexOf(b,start);return start<0?'Source marker not found.':t.slice(start,end>=0?end+b.length:start+700).trim();}
const asmNotes={
 registers:['zmm0','zmm0 = signal counts; zmm1 = equity. zmm2–5 hold category counts, zmm6–9 sums, and zmm10–13 squared sums. The inputs and output are defined by repr(C) Rust structures with checked offsets. Explicit register choices are part of the authored schedule.'],
 predicate:['Packed comparisons','One broadcast prediction becomes eight rule-specific comparisons. The ternary bitwise instruction performs the per-lane sign and absolute-value transform. Greater-than and masked equality implement strict and inclusive rules.'],
 masks:['Separate Boolean work from arithmetic','Entry = current position AND NOT previous position. Continuation = both positions. Exit = current position AND NOT next position. The schedule uses general-purpose registers for this Boolean state, then kmovw supplies arithmetic masks.'],
 accumulate:['Preserve the arithmetic contract','Masked increments, sums and squared sums update only the relevant lane/category. Vector lanes never perform a horizontal reduction across bars. Ordered products stay lane-local. Final stores write the exact shared Raw8 layout.']
};
function renderAssembly(k){let text='';if(k==='registers')text=sourceBetween('asm-source','/* Handwritten','k7=inclusive.')+'\n*/\n'+sourceBetween('asm-source','    mov r8, [rsi]','    kmovw k7, [rdx+192]');if(k==='predicate')text=sourceBetween('asm-source','    vbroadcastsd zmm20','    mov r14d, eax');if(k==='masks')text=sourceBetween('asm-source','    andn esi, r15d, edx','    mov ebx, edx');if(k==='accumulate')text=sourceBetween('asm-source','    andn edx, ebx, esi','    vaddpd zmm11{k2}, zmm11, zmm22')+'\n\n'+sourceBetween('asm-source','    vmovdqu64 [rcx], zmm0','    vmovdqu64 [rcx+192], zmm3');
 $('#asm-code').textContent=text;$('#asm-notes').innerHTML=`<h3>${asmNotes[k][0]}</h3><p>${asmNotes[k][1]}</p><p class="fine">This is an excerpt from the committed integrated assembly source, not an instruction-count performance prediction. The standalone and integrated files have separately retained identities.</p>`;$$('[data-asm]').forEach(b=>b.classList.toggle('selected',b.dataset.asm===k));}
$('#asm-tabs').addEventListener('click',e=>{const b=e.target.closest('[data-asm]');if(b)renderAssembly(b.dataset.asm);});renderAssembly('registers');
const claims=[
 {s:'“We trained one billion independent ridge models in 98 seconds.”',ok:false,why:'The process evaluated sampled rows, with replacement, using 2,304 configured training keys and cached fits/failures. The independent-fit interpretation is unsupported.'},
 {s:'“The original Rust implementation was already factorized before this investigation.”',ok:true,why:'Shared ridge statistics, fit/prediction reuse, base-score caching and factored cost arithmetic are present in the preserved original source.'},
 {s:'“63,622× is a directly timed billion-row native Limen versus Rust result.”',ok:false,why:'Only Rust was run at 1b. The numerator is a linear reference-cost projection from short prepared-native calibrations, with scoring-contract differences disclosed.'},
 {s:'“The integrated Rust pipeline contains no C scoring kernel.”',ok:true,why:'The final intrinsics are written in Rust; handwritten assembly is linked through global_asm!. Historical C prototypes remain evidence, not the deployed runtime.'},
 {s:'“Matching all 17 BaseScore fields proves current Limen’s backtest is economically equivalent.”',ok:false,why:'Those checks establish agreement with the original scalar Rust scorer for tested inputs. Current native Limen retains different economic conventions.'},
 {s:'“On the scaling host, the original and assembly pipelines actually ran 1b rows in 396.03 and 98.29 seconds.”',ok:true,why:'Both are complete-process measurements on the same AMD scaling guest. Their roughly 4.03× ratio needs no native extrapolation. Each cell has only one timed run.'}
];
function renderClaims(){ $('#claim-clinic').innerHTML=claims.map((c,i)=>{const a=state.claims[i],answered=a!==undefined;return `<article class="claim-card ${answered?'answered':''}"><blockquote>${esc(c.s)}</blockquote><div class="claim-buttons"><button data-claim="${i}" data-verdict="true">Supported</button><button data-claim="${i}" data-verdict="false">Unsupported</button></div><p class="claim-feedback" ${answered?'':'hidden'}>${answered?`${a===c.ok?'Correct.':'Revisit the boundary.'} ${c.why}`:''}</p></article>`;}).join('');}
$('#claim-clinic').addEventListener('click',e=>{const b=e.target.closest('[data-claim]');if(!b)return;state.claims[b.dataset.claim]=b.dataset.verdict==='true';save();renderClaims();});renderClaims();
function logTicks(min,max){let out=[];for(let k=Math.floor(Math.log10(min));k<=Math.ceil(Math.log10(max));k++)for(let q of[1,2,5]){const x=q*10**k;if(x>=min&&x<=max)out.push(x);}if(out.length>10)out=out.filter(x=>Math.abs(Math.log10(x)-Math.round(Math.log10(x)))<1e-8);return out;}
function timeText(s){return s<1?fmt(s*1000,2)+' ms':fmt(s,3)+' s';}
let funnelMode='common';
function renderFunnel(mode){
 funnelMode=mode;const F=DATA.funnel.summary,keys=mode==='common'?['limen','naive','stats','train','score','bounds','compact','intrinsics','assembly']:mode==='large'?levels:['intrinsics','assembly'],dat=F[mode],base=mode==='common'?dat.limen.wall_seconds.median:mode==='large'?dat.original.wall_seconds.median:dat.intrinsics.wall_seconds.median;
 const max=Math.max(...keys.map(k=>dat[k].wall_seconds.max))*1.3,min=Math.min(...keys.map(k=>dat[k].wall_seconds.min))*.75,W=900,L=230,R=75,T=20,rowH=41,H=keys.length*rowH+72,plot=W-L-R,x=v=>L+(Math.log10(v)-Math.log10(min))/(Math.log10(max)-Math.log10(min))*plot;
 let s='';for(const v of logTicks(min,max))s+=`<line class="grid" x1="${x(v)}" x2="${x(v)}" y1="${T-5}" y2="${H-45}"/><text x="${x(v)}" y="${H-24}" text-anchor="middle">${v<1?fmt(v*1000)+'ms':fmt(v,1)+'s'}</text>`;
 keys.forEach((k,i)=>{const y=T+rowH*i+15,d=dat[k].wall_seconds,c=colors[k]||'var(--green)';s+=`<text x="${L-14}" y="${y+4}" text-anchor="end" class="plot-label">${esc(labels[k])}</text><line x1="${x(d.min)}" x2="${x(d.max)}" y1="${y}" y2="${y}" stroke="${c}" stroke-width="3"/><circle cx="${x(d.median)}" cy="${y}" r="5" fill="${c}"><title>${labels[k]}: median ${d.median.toFixed(6)}s; min–max ${d.min.toFixed(6)}–${d.max.toFixed(6)}s</title></circle><text x="${W-3}" y="${y+4}" text-anchor="end">${fmt(base/d.median,mode==='common'?0:2)}×</text>`;});
 $('#funnel-chart').innerHTML=svgBox(s,W,H,'Log-scaled measured process timing medians and minimum–maximum ranges')+'<p class="chart-caption">Time axis is logarithmic. Dot = median; horizontal segment = recorded min–max, not a confidence interval.</p>';
 $('#funnel-context').textContent=mode==='common'?'Exactly 1,014 requested rows. Native: 3 complete repetitions; each Rust level: 9. Ratio to full native CLI includes preparation/artifact differences. Prepared matrices are 1,903 × 13 training and 1,431 × 13 test.':mode==='large'?'Original 1m sampled rows, 55 features. Five complete-process repeats per stage on this Intel guest. Same Rust calculation/output contract. Ratios use original Rust, not projected Limen.':'Dedicated 16-pair experiment, changing only the kernel inside one Rust executable. Complete-process ratio of medians: 1.0406×. A descriptive paired interval reaches effectively no gain.';
 $('#funnel-table').innerHTML=`<table><thead><tr><th>Stage</th><th>Process median</th><th>${mode==='common'?'Vs native CLI':mode==='large'?'Vs original Rust':'Vs Rust intrinsics'}</th><th>Repeats</th><th>Peak RSS median</th></tr></thead><tbody>${keys.map(k=>{const d=dat[k];return `<tr><td>${labels[k]}</td><td>${timeText(d.wall_seconds.median)}</td><td>${fmt(base/d.wall_seconds.median,2)}×</td><td>${d.n}</td><td>${fmt(d.rss_kib.median/1024,1)} MiB</td></tr>`;}).join('')}</tbody></table>`;
 $$('[data-funnel]').forEach(b=>b.classList.toggle('selected',b.dataset.funnel===mode));
}
$('#funnel-tabs').addEventListener('click',e=>{const b=e.target.closest('[data-funnel]');if(b)renderFunnel(b.dataset.funnel);});renderFunnel('common');
let visibleLevels=new Set(levels);
const dataRow=(n,l)=>DATA.scaling.rows.find(r=>r.n===n&&r.level===l);
function chartValue(r,view,ref){if(view==='wall')return r.wall_seconds;if(view==='gain')return r.measured_gain_vs_original_rust;if(view==='memory')return r.peak_rss_kib/1024;return r.multipliers_vs_projected_native[ref];}
function renderScaling(){
 const view=$('#scale-view').value,ref=$('#scale-reference').value,ni=Number($('#scale-n').value),n=sizes[ni],sel=$('#scale-level').value,r=dataRow(n,sel);$('#scale-n-out').value=sizeNames[ni];$('#scale-reference').disabled=view!=='multiplier';
 const warns={
 'limen-prepared':'PROJECTED numerator: uncached native Ridge evaluation on the same 55-feature matrices. New sklearn fit + native metrics per row; excludes raw preparation, Python startup and artifacts. Native scoring/pivot contracts differ. All Rust times are measured.',
 sklearn:'PROJECTED numerator: bare sklearn fit/predict/MAE on the same 55-feature matrices. Fewer outputs than either pipeline; excludes Python startup and native artifact work. All Rust times are measured.',
 'limen-cli':'SEPARATE REFERENCE-RATE PROJECTION: full native CLI calibration uses a different, 13-feature raw-data workflow. This is NOT a matched-input end-to-end speedup. All Rust times are measured.'};
 const w=$('#scale-warning');w.textContent=view==='multiplier'?warns[ref]:view==='gain'?'DIRECT MEASURED RATIO: original Rust process time divided by the selected stage on the same AMD host and original workload. No native projection. One run per point.':view==='wall'?'MEASURED: complete-process elapsed seconds, including startup, inputs, computation, outputs and exit. One fresh process per stage/size.':'MEASURED: peak resident set size recorded for each complete process. Values include the complete pipeline, not just payload estimates.';w.classList.toggle('warning',view==='multiplier');
 $('#scale-series').innerHTML=levels.map(l=>`<button class="${visibleLevels.has(l)?'active':''}" data-series="${l}" aria-pressed="${visibleLevels.has(l)}"><i class="swatch" style="--series:${colors[l]}"></i>${labels[l]}</button>`).join('');
 const shown=levels.filter(l=>visibleLevels.has(l)),vals=shown.flatMap(l=>sizes.map(n=>chartValue(dataRow(n,l),view,ref))),lo=Math.min(...vals)*.8,hi=Math.max(...vals)*1.25,W=920,H=380,L=90,R=22,T=25,B=53,px=i=>L+i*(W-L-R)/4,py=v=>T+(Math.log10(hi)-Math.log10(v))/(Math.log10(hi)-Math.log10(lo))*(H-T-B);let s='';
 for(const v of logTicks(lo,hi)){const text=view==='multiplier'?fmt(v)+'×':view==='gain'?fmt(v,1)+'×':view==='memory'?fmt(v)+' MiB':v<1?fmt(v,2)+' s':fmt(v)+' s';s+=`<line class="grid" x1="${L}" x2="${W-R}" y1="${py(v)}" y2="${py(v)}"/><text x="${L-13}" y="${py(v)+4}" text-anchor="end">${text}</text>`;}
 for(let i=0;i<5;i++)s+=`<line class="grid" x1="${px(i)}" x2="${px(i)}" y1="${T}" y2="${H-B}"/><text x="${px(i)}" y="${H-B+22}" text-anchor="middle">${sizeNames[i]}</text>`;
 s+=`<line x1="${px(ni)}" x2="${px(ni)}" y1="${T}" y2="${H-B}" stroke="var(--subtle)" stroke-width="1.3" stroke-dasharray="5 5"/>`;
 shown.forEach(l=>{const points=sizes.map((n,i)=>`${px(i)},${py(chartValue(dataRow(n,l),view,ref))}`).join(' ');s+=`<polyline points="${points}" fill="none" stroke="${colors[l]}" stroke-width="${l===sel?3:1.7}" opacity="${l===sel?1:.75}"/>`;sizes.forEach((n,i)=>{const val=chartValue(dataRow(n,l),view,ref);s+=`<circle class="chart-point" data-plot-level="${l}" data-plot-size="${i}" tabindex="0" role="button" aria-label="${esc(labels[l])}, ${sizeNames[i]}, ${fmt(val,3)}" cx="${px(i)}" cy="${py(val)}" r="${i===ni&&l===sel?6:3.8}" fill="${colors[l]}" stroke="var(--panel)" stroke-width="1"><title>${labels[l]} · ${sizeNames[i]} · ${fmt(val,4)}${view==='multiplier'||view==='gain'?'×':view==='wall'?' seconds':' MiB'}</title></circle>`;});});
 s+=`<text class="axis-label" x="${(L+W-R)/2}" y="${H-5}" text-anchor="middle">Sampled permutations · logarithmic axes</text>`;
 $('#scale-chart').innerHTML=svgBox(s,W,H,'Scaling curves with actual Rust timings and explicitly selected reference boundary');
 const native=r.native_estimates_seconds[ref],mul=r.multipliers_vs_projected_native[ref];
 $('#scale-readout').innerHTML=metric(fmt(r.wall_seconds,r.wall_seconds<1?4:2)+' s','Selected stage, complete process','MEASURED')+metric(fmt(r.measured_gain_vs_original_rust,2)+'×','Same-host gain over original Rust','MEASURED')+metric(fmt(mul)+'×','Relative to the selected native reference','PROJECTED REFERENCE COST','projected')+metric(fmt(r.sampled_rows_per_second/1e6,2)+'m/s','Sampled rows, including failed-fit rows','MEASURED');
 $('#scale-table').innerHTML=`<table><thead><tr><th>${sizeNames[ni]} rows · stage</th><th>Wall time</th><th>Vs original</th><th>Peak RSS</th><th>Projected native-relative</th></tr></thead><tbody>${levels.map(l=>{const a=dataRow(n,l);return `<tr class="${l===sel?'selected-row':''}"><td>${labels[l]}</td><td>${timeText(a.wall_seconds)}</td><td>${fmt(a.measured_gain_vs_original_rust,2)}×</td><td>${fmt(a.peak_rss_kib/1024,1)} MiB</td><td>${fmt(a.multipliers_vs_projected_native[ref])}×</td></tr>`;}).join('')}</tbody></table><p class="fine" style="padding:0 14px">Selected native model at ${sizeNames[ni]}: ${fmt(native,2)} seconds (${fmt(native/86400,2)} days), inferred. Valid / failed sampled rows: ${fmt(r.valid_rows)} / ${fmt(r.failed_rows)}. Distinct valid score keys: ${fmt(r.score_cache_entries)}.</p>`;
}
for(const id of['scale-view','scale-reference','scale-level'])$('#'+id).addEventListener('change',renderScaling);$('#scale-n').addEventListener('input',renderScaling);
$('#scale-series').addEventListener('click',e=>{const b=e.target.closest('[data-series]');if(!b)return;if(visibleLevels.has(b.dataset.series)){if(visibleLevels.size===1){toast('Keep at least one measured series visible.');return;}visibleLevels.delete(b.dataset.series);}else visibleLevels.add(b.dataset.series);renderScaling();});
function selectPlot(e){const p=e.target.closest('[data-plot-level]');if(p){$('#scale-level').value=p.dataset.plotLevel;$('#scale-n').value=p.dataset.plotSize;renderScaling();}}
$('#scale-chart').addEventListener('click',selectPlot);$('#scale-chart').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectPlot(e);}});renderScaling();
$('#download-scaling').addEventListener('click',()=>download('terasweep_scaling_results.json',JSON.stringify(DATA.scaling,null,2),'application/json'));
$('#coverage-table').innerHTML=`<table><thead><tr><th>Sampled rows</th><th>Unique valid score keys</th><th>Key-space coverage</th><th>Valid rows reusing a base score</th></tr></thead><tbody>${sizes.map((n,i)=>{const r=dataRow(n,'assembly');return `<tr><td>${sizeNames[i]}</td><td>${fmt(r.score_cache_entries)}</td><td>${fmt(r.score_cache_entries/3364062*100,2)}%</td><td>${fmt((1-r.score_cache_entries/r.valid_rows)*100,2)}%</td></tr>`;}).join('')}</tbody></table>`;
function renderAmdahl(){const pp=Number($('#amdahl-p').value),s=Number($('#amdahl-s').value),p=pp/100,gain=1/((1-p)+p/s),cap=p===1?Infinity:1/(1-p);$('#amdahl-p-out').value=fmt(pp,3)+'%';$('#amdahl-s-out').value=fmt(s,1)+'×';$('#amdahl-output').innerHTML=metric(fmt(gain,4)+'×','Calculated whole-program throughput')+metric(fmt((1-1/gain)*100,4)+'%','Calculated complete-runtime reduction')+metric(Number.isFinite(cap)?fmt(cap,4)+'×':'Unbounded','Limit with an infinitely fast target kernel')+metric(fmt(100-pp,3)+'%','Unaffected fraction of original runtime');}
$('#amdahl-p').addEventListener('input',renderAmdahl);$('#amdahl-s').addEventListener('input',renderAmdahl);$$('[data-amdahl]').forEach(b=>b.onclick=()=>{const vals={half:[50,2],billion:[.342,8],near:[95,8]}[b.dataset.amdahl];$('#amdahl-p').value=vals[0];$('#amdahl-s').value=vals[1];renderAmdahl();});renderAmdahl();
$('#capstone-text').value=state.capstone||'';$('#capstone-text').addEventListener('input',e=>{state.capstone=e.target.value;save();});$('#capstone-check').onclick=()=>{const t=$('#capstone-text').value.toLowerCase(),checks=[['Recorded time and scale',/98[.,]2|98 seconds/.test(t)&&/billion|1b|1,000,000,000/.test(t)],['Sampled rows, not independent models',/sampl|independent/.test(t)],['Projected native denominator',/project|infer|linear/.test(t)&&/native|limen/.test(t)],['Assembly increment qualified',/assembly/.test(t)&&/noise|noisy|single|pair|limit|not establish|uncertain|small/.test(t)],['Reproducibility detail',/git|hash|commit|replay|frozen|snapshot/.test(t)]];$('#capstone-feedback').innerHTML='<p class="fine">Keyword checklist, not an automated fact-check or grade. Inspect your wording against the evidence and the example.</p><div class="capstone-checks">'+checks.map(([name,ok])=>`<div class="${ok?'found':'missing'}">${ok?'✓ Mention detected:':'○ Review:'} ${name}</div>`).join('')+'</div>';};
function exportLearning(){let text='# Terasweep masterclass — personal learning record\n\nExported '+new Date().toISOString()+'\nCourse basis: benchmark-handoff-2026-09-24 / 2e549d4.\n\n';for(const c of DATA.chapters)text+=`## ${strip(c.title)}\n\nUnderstood: ${state.done.includes(c.id)?'yes':'not marked'}\n\n${state.notes[c.id]||'(No notes.)'}\n\n`;text+='## Capstone statement\n\n'+(state.capstone||'(Not written.)')+'\n';download('terasweep_learning_record.md',text,'text/markdown');}
$('#export-learning').addEventListener('click',exportLearning);$('#export-notes-top').addEventListener('click',exportLearning);
const glossary={
 'Ablation':'An experiment that disables a reuse or execution layer while retaining the relevant input and output contract, so its contribution can be inspected.',
 'BaseScore':'The inherited score summary: eight integer fields and nine FP64 fields encoding signal count, gross equity, transitions, eligible bars and four categories of moments.',
 'Categorical identity':'An index into a finite configured parameter set. The compact scheduler uses these ordinals while retaining the original value-based deduplication semantics.',
 'Cholesky':'The in-place lower-triangular factorization used by the ridge solver, followed by forward and backward substitution. It is not an explicit matrix inverse.',
 'Constant signal':'A prediction/rule/threshold combination for which every signal is off or every signal is on. Finite min/max bounds can establish this without another full scan.',
 'FP64':'Double-precision floating-point storage used by the pipeline. The SIMD work retained it rather than switching to a cheaper reduced-precision model.',
 'Gram matrix':'G = XᵀX, or its centered equivalent. With data fixed, it can be reused across regularization strengths.',
 'Golden result':'The retained canonical stable output for the original one-million-row sweep. The two timing fields are excluded from comparison.',
 'Handwritten assembly':'An independently authored instruction sequence with explicit register, mask, control-flow and output-store choices; not compiler output renamed as a source file.',
 'Hot/internal timer':'A specific code-delimited interval. The original runner includes ridge precomputation and winner details but excludes input loading and price-statistics preparation.',
 'Intrinsics':'In this final runtime, Rust std::arch::x86_64 calls expressing SIMD operations. The compiler still chooses much of the instruction scheduling and register allocation.',
 'Kernel':'The focused eight-job base-scoring function. A kernel timing is not automatically a full-process timing.',
 'Limen Ridge SFD':'The native Limen definition and reference-model integration backed by a fresh sklearn Ridge estimator, with manifest preparation and native evaluation.',
 'Logical key / physical payload':'A key identifies a requested score state; several keys may point to the same constant payload without dropping any logical requests.',
 'Masked operation':'An operation applied only to selected vector lanes. The scorer uses masks to maintain independent per-job category accumulators.',
 'Projection':'An inferred native-reference runtime using a short-run linear calibration. At large N it is not an executed native run.',
 'Sampled permutation':'A requested parameter combination drawn with replacement. Counted rows are not guaranteed to be distinct combinations or distinct fitted models.',
 'solver_eps':'Terasweep’s Cholesky-pivot cutoff. It is not sklearn’s iterative-solver tol parameter.',
 'UEL':'Limen’s Universal Experiment Loop: the native orchestration path that prepares data, calls the model, and records experiment outputs.',
 'Useful-lane utilization':'The fraction of SIMD lanes occupied by requested jobs in the tested batching scheme, rather than padding or duplicated tail jobs.',
 'Wall time':'Complete-process elapsed time at the parent boundary, including startup, reading, computation, writing and exit.',
 'Working-tree cleanliness':'No uncommitted tracked or untracked project changes. It does not imply ignored build outputs, historical runs, or redundant scratch directories do not exist.'
};
$('#glossary-terms').innerHTML=Object.entries(glossary).map(([a,b])=>`<dt>${esc(a)}</dt><dd>${esc(b)}</dd>`).join('');
const first=location.hash.slice(1)||state.chapter;navigate(first,false);
window.__MASTERCLASS_READY__=true;
