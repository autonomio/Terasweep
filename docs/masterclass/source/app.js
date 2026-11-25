'use strict';
const DATA=JSON.parse(document.getElementById('course-data').textContent);
const $=(q,r=document)=>r.querySelector(q), $$=(q,r=document)=>Array.from(r.querySelectorAll(q));
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(x,d=0)=>Number(x).toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});
const storageKey='terasweep-masterclass-v2';
let storageOK=true,state={chapter:'brief',done:[],notes:{},quiz:{},theme:'dark'};
try{let old=JSON.parse(localStorage.getItem(storageKey)||'null');if(old&&typeof old==='object')state={...state,...old};}catch(e){storageOK=false;}
state.done=Array.isArray(state.done)?state.done:[];state.notes=state.notes||{};state.quiz=state.quiz||{};
function save(){try{localStorage.setItem(storageKey,JSON.stringify(state));}catch(e){storageOK=false;}$('#note-status').textContent=storageOK?'Saved in this browser only.':'Local storage unavailable. Export your learning record to keep it.';}
function toast(text){const t=$('#toast');t.textContent=text;t.hidden=false;clearTimeout(toast.timer);toast.timer=setTimeout(()=>t.hidden=true,2800);}
function download(name,text,type='text/plain'){let a=document.createElement('a'),u=URL.createObjectURL(new Blob([text],{type}));a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}
const lessonTitle=id=>$('#'+id+'-title').textContent;
function metric(value,label,qual='',cls=''){return `<div class="metric ${cls}">${qual?`<small>${qual}</small>`:''}<strong>${value}</strong><span>${label}</span></div>`;}
function progress(){
 $$('.nav-item').forEach(b=>{$('.nav-done',b).textContent=state.done.includes(b.dataset.go)?'✓':'';b.classList.toggle('active',b.dataset.go===state.chapter);b.setAttribute('aria-current',b.dataset.go===state.chapter?'step':'false');});
 $('#progress-count').textContent=`${state.done.length} / ${DATA.chapters.length}`;$('#progress-fill').style.width=`${state.done.length/DATA.chapters.length*100}%`;
 const done=state.done.includes(state.chapter);$('#mastered').classList.toggle('done',done);$('#mastered').textContent=done?'✓ Marked understood':'Mark this chapter understood';$('#mastered').setAttribute('aria-pressed',String(done));
}
function navigate(id,scroll=true){
 if(!DATA.chapters.some(c=>c.id===id))id='brief';
 state.chapter=id;save();
 $$('.lesson').forEach(s=>s.classList.toggle('active',s.id===id));
 const idx=DATA.chapters.findIndex(c=>c.id===id);$('#prev-lesson').disabled=idx===0;$('#next-lesson').disabled=idx===DATA.chapters.length-1;
 $('#next-lesson').textContent=idx===DATA.chapters.length-1?'Course complete':'Next chapter →';
 $('#lesson-note').value=state.notes[id]||'';
 $('#chapter-counter').textContent=`${idx+1} / ${DATA.chapters.length}`;document.title=`${lessonTitle(id)} — Terasweep performance learning`;
 if(scroll){try{history.replaceState(null,'','#'+id);}catch(e){}}
 $('#sidebar').classList.remove('open');$('#menu-btn').setAttribute('aria-expanded','false');
 progress();if(scroll){window.scrollTo({top:0,behavior:'instant'});$('.chapter-header h1',$('#'+id)).focus({preventScroll:true});}
 for(const dialog of $$('dialog'))if(dialog.open)dialog.close();
}
function renderTheme(){
 document.documentElement.dataset.theme=state.theme;
 const nextTheme=state.theme==='light'?'Dark':'Light';
 $('#theme-btn').textContent=nextTheme;
 $('#theme-btn').setAttribute('aria-label',`Switch to ${nextTheme.toLowerCase()} mode`);
}
renderTheme();
$('#theme-btn').addEventListener('click',()=>{state.theme=state.theme==='dark'?'light':'dark';renderTheme();save();});
$('#menu-btn').addEventListener('click',()=>{const open=$('#sidebar').classList.toggle('open');$('#menu-btn').setAttribute('aria-expanded',String(open));});
$('#mastered').addEventListener('click',()=>{if(state.done.includes(state.chapter))state.done=state.done.filter(x=>x!==state.chapter);else state.done.push(state.chapter);save();progress();});
$('#prev-lesson').addEventListener('click',()=>navigate(DATA.chapters[Math.max(0,DATA.chapters.findIndex(c=>c.id===state.chapter)-1)].id));
$('#next-lesson').addEventListener('click',()=>navigate(DATA.chapters[Math.min(DATA.chapters.length-1,DATA.chapters.findIndex(c=>c.id===state.chapter)+1)].id));
$('#lesson-note').addEventListener('input',e=>{state.notes[state.chapter]=e.target.value;save();});
$('#reset-progress').addEventListener('click',()=>{if(confirm('Reset chapter checkmarks and quiz choices? Your written notes will be kept.')){state.done=[];state.quiz={};save();progress();$$('.quiz-options button').forEach(b=>b.classList.remove('correct','incorrect'));$$('.quiz-feedback').forEach(p=>p.hidden=true);toast('Progress reset. Written notes retained.');}});
window.addEventListener('hashchange',()=>navigate(location.hash.slice(1)));
document.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT','BUTTON','SUMMARY','A'].includes(document.activeElement.tagName)||$$('dialog[open]').length)return;if(e.key==='/'){e.preventDefault();openDialog('search');return;}if(e.target.closest('.table-wrap'))return;if(e.key==='ArrowRight')$('#next-lesson').click();if(e.key==='ArrowLeft')$('#prev-lesson').click();});
document.addEventListener('click',e=>{
 const go=e.target.closest('[data-go]');if(go){if(go.tagName==='A'&&(e.metaKey||e.ctrlKey||e.shiftKey||e.altKey))return;e.preventDefault();navigate(go.dataset.go);return;}
 const ref=e.target.closest('[data-source]');if(ref){openSource(ref.dataset.source,ref.dataset.lines||'');return;}
 const op=e.target.closest('[data-open]');if(op){openDialog(op.dataset.open);return;}
 const close=e.target.closest('[data-close]');if(close){close.closest('dialog').close();return;}
 const cb=e.target.closest('.copy-code');if(cb){const text=$('pre',cb.closest('.code-card')).textContent;copyText(text);}
});
async function copyText(text){try{await navigator.clipboard.writeText(text);toast('Commands copied.');}catch(e){const t=document.createElement('textarea');t.value=text;document.body.appendChild(t);t.select();const ok=document.execCommand('copy');t.remove();toast(ok?'Commands copied.':'Select and copy the command block manually.');}}
function openDialog(id){const d=$('#'+id+'-dialog');if(!d.open)d.showModal();if(id==='evidence'){renderSourceList();if(!currentSource)selectSource('sweep');}if(id==='search'){$('#course-search').focus();renderSearch();}}
for(const d of $$('dialog'))d.addEventListener('click',e=>{if(e.target===d){const r=d.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)d.close();}});
let currentSource=null;
function renderSourceList(){const q=$('#source-search').value.toLowerCase();$('#source-buttons').innerHTML=Object.entries(DATA.sources).filter(([k,v])=>(v.title+' '+v.path).toLowerCase().includes(q)).map(([k,v])=>`<button class="source-btn ${k===currentSource?'active':''}" data-source="${k}">${esc(v.title)}<small>${esc(v.kind)}</small></button>`).join('')||'<p class="fine">No matching evidence.</p>';}
function selectSource(id,lines=''){
 const s=DATA.sources[id];if(!s)return;currentSource=id;$('#source-title').textContent=s.title;$('#source-meta').textContent=s.path+' · SHA-256 '+s.sha256;
 $('#source-git').href=s.url;$('#source-git').hidden=!s.url;$('#source-note').textContent=s.note||'Line numbers and SHA-256 identify the copy embedded in this build.';
 let [a,b]=lines.split('-').map(Number);b=b||a;
 $('#source-text').innerHTML=s.text.split('\n').map((line,i)=>`<div class="source-line ${a&&i+1>=a&&i+1<=b?'highlight':''}" id="source-line-${i+1}"><span class="line-no">${i+1}</span><span>${esc(line)||' '}</span></div>`).join('');
 renderSourceList();requestAnimationFrame(()=>{if(a&&$('#source-line-'+a))$('#source-line-'+a).scrollIntoView({block:'start',behavior:'instant'});else $('.source-main').scrollTop=0;});
}
function openSource(id,lines=''){openDialog('evidence');selectSource(id,lines);}
$('#source-search').addEventListener('input',renderSourceList);
$('#source-export').addEventListener('click',()=>{const s=DATA.sources[currentSource];if(s)download(s.path.split('/').pop(),s.text);});
$('#course-search').addEventListener('input',renderSearch);
function renderSearch(){const q=$('#course-search').value.toLowerCase();const match=DATA.chapters.filter(c=>$(`#${c.id}`).textContent.toLowerCase().includes(q));$('#search-results').innerHTML=match.map(c=>`<button class="search-result" data-go="${c.id}"><b>${esc(lessonTitle(c.id))}</b><span>${esc($(`#${c.id} .lead`).textContent)}</span></button>`).join('')||'<p>No lesson matches that phrase.</p>';}
for(const q of $$('.checkpoint')){
 const id=q.closest('.lesson').id,explanation=$('.quiz-feedback',q).textContent;
 function answer(n){const correct=Number(q.dataset.correct);$$('[data-choice]',q).forEach(b=>{b.classList.toggle('correct',Number(b.dataset.choice)===correct);b.classList.toggle('incorrect',Number(b.dataset.choice)===n&&n!==correct);});const p=$('.quiz-feedback',q);p.hidden=false;p.textContent=(n===correct?'Yes. ':'Incorrect. ')+explanation;state.quiz[id]=n;save();}
 q.addEventListener('click',e=>{const b=e.target.closest('[data-choice]');if(b)answer(Number(b.dataset.choice));});if(id in state.quiz)answer(state.quiz[id]);
}
function svgBox(content,w=700,h=300,label='Interactive learning diagram'){return `<svg class="chart-svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}">${content}</svg>`;}
function renderRidge(){
 const x=[0,1,2,3,4,5,6,7],y=[1.4,2.1,2.7,4.6,4.9,6.2,7.5,7.9],a=Math.pow(10,Number($('#ridge-alpha').value)),inter=$('#ridge-intercept').checked;
 const xm=inter?x.reduce((p,v)=>p+v,0)/x.length:0,ym=inter?y.reduce((p,v)=>p+v,0)/y.length:0;
 const g=x.reduce((p,v)=>p+(v-xm)**2,0),rhs=x.reduce((p,v,i)=>p+(v-xm)*(y[i]-ym),0),w=rhs/(g+a),b=inter?ym-xm*w:0;
 const W=380,H=260,l=35,r=18,t=16,bot=47,px=v=>l+v/7*(W-l-r),py=v=>H-bot-v/9*(H-t-bot);let s='';
 for(let v=0;v<=8;v+=2)s+=`<line class="grid" x1="${l}" x2="${W-r}" y1="${py(v)}" y2="${py(v)}"/><text x="${l-9}" y="${py(v)+4}" text-anchor="end">${v}</text>`;
 s+=`<line x1="${px(0)}" x2="${px(7)}" y1="${py(b)}" y2="${py(7*w+b)}" stroke="var(--green)" stroke-width="2.5"/>`;
 for(let i=0;i<x.length;i++)s+=`<circle cx="${px(x[i])}" cy="${py(y[i])}" r="4.5" fill="var(--blue)"><title>Illustrative point: x=${x[i]}, y=${y[i]}</title></circle><text x="${px(i)}" y="${H-bot+17}" text-anchor="middle">${i}</text>`;
 s+=`<text class="axis-label" x="${px(3.5)}" y="${H-7}" text-anchor="middle">x</text>`;
 $('#ridge-chart').innerHTML=svgBox(s,W,H,'Illustrative ridge regression points and fitted line');
 $('#ridge-metrics').innerHTML=metric(a<.01?a.toExponential(2):fmt(a,3),'Regularization α')+metric(fmt(g,3),'Reusable G = Σ(x − x̄)²')+metric(fmt(w,4),'Fitted coefficient w')+metric(fmt(b,4),'Recovered intercept b');
 $('#ridge-explanation').textContent=`With the ${inter?'centered':'raw'} statistics, h = ${fmt(rhs,4)} and w = h / (G + α). The value of α enters only through G + α.`;
}
$('#ridge-alpha').addEventListener('input',renderRidge);$('#ridge-intercept').addEventListener('change',renderRidge);renderRidge();
function renderFitReuse(){
 const unique=Number($('#fit-settings').value),repeats=Number($('#fit-repeats').value),total=unique*repeats;
 $('#fit-settings-value').value=unique;$('#fit-repeats-value').value=repeats;
 $('#reuse-metrics').innerHTML=metric(fmt(total),'Plan entries')+
  metric(fmt(unique),'Distinct fits')+metric(fmt(total-unique),'Repeated fits avoided');
}
$('#fit-settings').addEventListener('input',renderFitReuse);
$('#fit-repeats').addEventListener('input',renderFitReuse);
renderFitReuse();
function exportLearning(){
 let text='# Terasweep performance learning — personal record\n\nExported '+new Date().toISOString()+'\n\n';
 for(const chapter of DATA.chapters)text+=`## ${lessonTitle(chapter.id)}\n\nUnderstood: ${state.done.includes(chapter.id)?'yes':'not marked'}\n\n${state.notes[chapter.id]||'(No notes.)'}\n\n`;
 download('terasweep_learning_record.md',text,'text/markdown');
}
$('#export-notes-top').addEventListener('click',exportLearning);
const glossary={
 'Cholesky':'A lower-triangular matrix factorization used to solve the Ridge system.',
 'Gram matrix':'G = XᵀX, or its centered equivalent. With data fixed, it is reusable across regularization strengths.',
 'Holdout':'Labeled data held back from both fitting and validation to estimate how the selected model performs on unseen data.',
 'MAE':'Mean absolute error. Average the absolute difference between each prediction and its corresponding target.',
 'Numerical parity':'Agreement with a reference at a stated tolerance, on matched inputs and settings.',
 'Trial':'One ordered model entry in the sweep plan. Multiple entries can specify the same model settings.',
 'Validation':'Labeled data used to choose a model or compare its settings.'
};
$('#glossary-terms').innerHTML=Object.entries(glossary).map(([term,definition])=>`<dt>${esc(term)}</dt><dd>${esc(definition)}</dd>`).join('');
navigate(location.hash.slice(1)||state.chapter,false);
window.__MASTERCLASS_READY__=true;
