import json,re,hashlib,collections
from pathlib import Path
BASE=Path(__file__).resolve().parent
FIELDS=['doc_id','title','source_name','org_name','source_type','priority','role','country','language','route','section_type','version','effective_date','updated_date','collected_at','source_url','raw_file','content']
def date(s): return str(s or '')[:10]
def tidy(s):
 s=str(s or '').replace('\u200b','').replace('\ufeff','').strip()
 return re.sub(r'[ \t]+',' ',s)
def build(r, country):
 src=r['source']; rid=r['record_id']; body=tidy(r.get('content',''))
 out={k:'' for k in FIELDS}
 out.update(doc_id=rid.lower().replace('_','.'), country=country,language='en',content=body,source_url=r.get('source_url',''),collected_at=date(r.get('collected_at')),raw_file=f'all_{"uk" if country=="UK" else "china"}_final.json',effective_date='',updated_date='',version='',route='all',section_type='page',role='evidence')
 if country=='UK':
  if src=='legislation.gov.uk':
   out.update(source_name='Aviation Security Act 1982',org_name='UK Legislation',source_type='law',priority=1)
   typ=r.get('section_type',''); schedule=r.get('schedule_number'); sec=r.get('section_number'); heading=r.get('section_title') or ''
   out['section_type']='appendix' if schedule is not None or str(typ).lower() in ('schedule','appendix') else 'body'
   out['title']='Aviation Security Act 1982'+(f' Schedule {schedule}' if schedule is not None else f' Section {sec}' if sec else '')+(f' — {heading}' if heading and not set(heading)<=set('. ') else '')
   # Current law version/effective date cannot be inferred from act title or scrape.
  elif src=='UK Civil Aviation Authority':
   out.update(title='CAA Baggage Restrictions — '+r.get('item_name',''),source_name='CAA Safety advice on what to pack',org_name='UK Civil Aviation Authority',source_type='government',priority=2,route='all')
  elif src=='GOV.UK':
   out.update(title='Hand luggage restrictions at UK airports — '+r.get('page_category','overview'),source_name='GOV.UK Hand luggage restrictions',org_name='UK Government',source_type='government',priority=3)
  else: raise ValueError(src)
 else:
  if src=='China Southern Airlines':
   p=r.get('page_number');out.update(title=f'{r.get("document_name", "China Southern Conditions")} — p. {p}',source_name='China Southern International Conditions of Carriage',org_name=src,source_type='airline',priority=2,route='international',section_type='body')
  elif src=='China Eastern Airlines':
   out.update(title=r.get('title') or r.get('document_name'),source_name='China Eastern Safety instructions for lithium batteries and dangerous goods',org_name=src,source_type='airline',priority=2,route='all')
  elif src=='Beijing Municipal Government':
   out.update(title=r.get('title',''),source_name='Beijing Government 3C power bank notice',org_name=src,source_type='press',priority=9,role='notice',route='domestic',updated_date=r.get('published_date',''))
  elif src=='Shanghai Municipal Government':
   out.update(title=r.get('title',''),source_name='Shanghai Government power bank guidance',org_name=src,source_type='government',priority=3,role='notice',route='all',updated_date=r.get('published_date',''))
  else: raise ValueError(src)
 # Unknown dates are empty, not fabricated; web snapshot version uses collection date.
 if out['source_type']!='law':out['version']=out['collected_at']
 # Preserve original attributes useful to later preprocessing and review.
 out['original_record_id']=rid
 for key in ('section_number','section_title','schedule_number','page_number','item_name','carry_on_baggage','checked_baggage','airline_approval','restrictions','page_category','flight_scope','scope','published_date'):
  if key in r:out[key]=r[key]
 return out

def main():
 all_out=[]
 for country in ('UK','CN'):
  slug='uk' if country=='UK' else 'china'
  raw=json.loads((BASE/f'all_{slug}_final.json').read_text(encoding='utf-8'))
  records=[build(r,country) for r in raw]
  assert len(records)==len(raw)
  assert all(x['content'] and x['doc_id'] and x['source_url'] for x in records)
  assert len(set(x['doc_id'] for x in records))==len(records)
  outpath=BASE/f'all_{slug}_preprocessed.json'
  outpath.write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
  print(outpath.name,len(records),collections.Counter(x['source_type'] for x in records))
  all_out.extend(records)
 assert len(set(x['doc_id'] for x in all_out))==len(all_out)
 (BASE/'all_uk_china_preprocessed.json').write_text(json.dumps(all_out,ensure_ascii=False,indent=2),encoding='utf-8')
 print('combined',len(all_out))
if __name__=='__main__':main()
