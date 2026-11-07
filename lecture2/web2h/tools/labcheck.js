const fs=require('fs'),vm=require('vm');
const src=fs.readFileSync(__dirname+'/../js/core.js','utf8')+fs.readFileSync(__dirname+'/../js/lab.js','utf8')+`
;console.log('FEED w,b',FEED.w.toFixed(3),FEED.b.toFixed(3),'Lmin',FEED.Lmin.toFixed(3),'lmax',FEED.lmax.toFixed(2),'lmin',FEED.lmin.toFixed(3),'L(start)',FEED.L(...FEED.start).toFixed(2), 'ymax', Math.max(...FEED.y).toFixed(2), Math.min(...FEED.y).toFixed(2));
console.log('ARR Ea',ARR.Ea_fit.toFixed(1),'CI',ARR.ci.map(v=>v.toFixed(1)),'A',ARR.A_fit.toExponential(2), 'k', ARR.k.map(v=>v.toExponential(1)).join(' '));
console.log('COND kappa raw',COND.raw.kappa.toFixed(0),'std',COND.std.kappa.toFixed(2));
console.log('LAC w b',LAC.w.toFixed(2),LAC.b.toFixed(2),'thr',LAC.thresh.toFixed(2),'ce',LAC.ce(LAC.w,LAC.b).toFixed(3),'line',LAC.line.w.toFixed(3),LAC.line.b.toFixed(3),'ones',sum(LAC.y));
console.log('WIN ones',sum(WIN.y),'of',WIN.n);
console.log('WARP acc',WARP.acc);
const t0=Date.now(); const net=new MLP([2,16,1],'relu','sigmoid',4); let L; for(let e=0;e<3000;e++){L=net.step(WIN.Z,WIN.y,0.5,0.9); if([0,10,30,100,300,1000,2999].includes(e)) console.log('ep',e,L.toFixed(4),net.acc(WIN.Z,WIN.y).toFixed(3));} console.log('ms',Date.now()-t0, 'params', net.nParams);
console.log('HAL polys', HAL.polys.slice(1).map(p=>p.d+':'+p.tr.toExponential(1)+'/'+p.va.toExponential(1)).join(' '));
for (const k of [1,2,4,8,16]) {const f=HAL.fitRelu(k); console.log('relu',k, mean(HAL.S.map((s,i)=>(f.f(s)-HAL.Y[i])**2)).toExponential(2), 'f(40)', f.f(40).toFixed(3));}
console.log('BP', JSON.stringify(BP));
`;
vm.runInNewContext(src,{console,Math,Date,Array,Float32Array,Set});
