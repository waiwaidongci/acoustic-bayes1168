import {domain} from '../constants/domain.js';
import {research} from '../stores/research.js';
import {offsetStudiesApi} from '../api/research.js';
import {status} from './common/status.js';
import {esc,fmt} from '../utils/format.js';
import {action,toast} from '../hooks/actions.js';

function rangeText(ranges){
  if(!ranges||!ranges.length)return '—';
  return ranges.map(([a,b])=>a===b?fmt(a):`${fmt(a)} ~ ${fmt(b)}`).join('；');
}

function studyCard(s){
  const done=s.shard_done||0,total=s.shard_total||0;
  const pct=total?Math.round(done/total*100):0;
  const busy=['queued','running'].includes(s.state);
  const body=busy
    ? `<div class="offset-progress"><span class="spinner"></span><p>分片处理中 ${done}/${total}（${pct}%）。全批算齐后才显示判读数上下限，旧研究照常读取。</p></div>`
    : s.state==='failed'
      ? `<div class="failure"><p>${esc(s.error||'零偏研究失败')}</p></div>`
      : (()=>{const r=s.result;return `
        <dl class="metrics offset-metrics">
          <div><dt>判读数下限</dt><dd>${fmt(r.minimum.count)}</dd></div>
          <div><dt>判读数上限</dt><dd>${fmt(r.maximum.count)}</dd></div>
          <div><dt>零偏判读数</dt><dd>${fmt(r.count_at_zero_offset??'—')}</dd></div>
          <div><dt>验证样本</dt><dd>${fmt(r.validation_samples)}</dd></div>
        </dl>
        <table class="offset-table">
          <thead><tr><th>极值</th><th>判读数</th><th>达到该判读数的偏移（Hz）</th></tr></thead>
          <tbody>
            <tr><td>下限</td><td>${fmt(r.minimum.count)}</td><td>${esc(rangeText(r.minimum.offsets))}</td></tr>
            <tr><td>上限</td><td>${fmt(r.maximum.count)}</td><td>${esc(rangeText(r.maximum.offsets))}</td></tr>
          </tbody>
        </table>
        <p class="fine">目标故障「${esc(r.target_label)}」· 偏移区间 ${fmt(r.offset_min)} ~ ${fmt(r.offset_max)} Hz · 临界偏移 ${fmt(r.critical_offsets)} 个 · ${fmt(r.shards)} 片归并</p>`;})();
  return `<article class="offset-card" data-offset-id="${s.id}">
    <header><strong>${esc(s.name)}</strong>${status(s.state)}<small>#${s.id} · 任务 #${s.run_id}</small></header>
    ${body}
  </article>`;
}

export function offsetPanel(){
  const runId=research.runId;
  const list=research.offsetStudies.filter(s=>s.run_id===runId);
  const labels=(research.activeRun?.result?.labels)||[];
  const labelOptions=labels.map(l=>`<option value="${esc(l)}">${esc(l)}</option>`).join('');
  return `<section class="offset-panel">
    <div class="page-heading compact-heading"><p class="eyebrow">ZERO-BIAS SWEEP</p>
    <h2>零偏偏移区间分析</h2>
    <p>冻结本研究的交叉验证折模型，只给同批验证样本的频谱质心加同一偏移；分片可恢复，全批算齐后给出目标故障判读数的精确上下限与对应偏移。</p></div>
    ${research.activeRun?.state==='completed'?`
    <form id="offset-form" class="offset-form">
      <label>目标故障判读<select name="target_label" required>${labelOptions}</select></label>
      <label>偏移下限 Hz<input name="offset_min" type="number" step="any" value="-200" required></label>
      <label>偏移上限 Hz<input name="offset_max" type="number" step="any" value="200" required></label>
      <label>单片样本数（有限内存）<input name="shard_size" type="number" min="1" max="100000" value="1000"></label>
      <button type="submit">发起零偏分析 →</button>
    </form>`:'<p class="fine">仅已完成的分析任务可发起零偏偏移分析。</p>'}
    <div class="offset-list">${list.length?list.map(studyCard).join(''):'<p class="empty">还没有零偏研究。</p>'}</div>
  </section>`;
}

export function bindOffset(refresh){
  action('#offset-form',async form=>{
    const fd=new FormData(form);
    const payload={
      run_id:Number(research.runId),
      target_label:fd.get('target_label'),
      offset_min:Number(fd.get('offset_min')),
      offset_max:Number(fd.get('offset_max')),
      shard_size:Number(fd.get('shard_size')||1000),
    };
    try{
      const created=await offsetStudiesApi.create(payload);
      research.offsetStudies=[created,...research.offsetStudies];
      toast('零偏研究已入队，分片可恢复地处理中');
      await refresh();
    }catch(e){toast(e.message,true);}
  });
}
