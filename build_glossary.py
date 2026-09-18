#!/usr/bin/env python3
"""Classify proper-noun candidates into a series glossary using the NVIDIA free lane."""
import os,json,re,sys,time,urllib.request
key=[l.split('=',1)[1].strip() for l in open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'.env')) if l.startswith('NVIDIA_API_KEY_1=')][0]
series=sys.argv[1]
c=json.load(open(f'glossaries/{series}_candidates.json'))
items=[(w,d) for w,d in c.items()]
SYS='''You classify words from an English romance series. For each candidate you get the word, its count, and one context snippet. Return JSON array of objects: {"word":..., "type": one of "character","place","team_or_brand","nickname_or_petname","title_or_role","other_proper","not_a_name"}. "not_a_name" = ordinary words that were capitalised (contractions, sentence starters, common nouns). Be strict: a first/last name of a person = character. Output JSON only.'''
def call(batch):
    body={"model":"google/gemma-4-31b-it","max_tokens":6000,"temperature":0.0,
          "messages":[{"role":"system","content":SYS},{"role":"user","content":json.dumps([{"word":w,"count":d["count"],"ctx":d["ctx"]} for w,d in batch],ensure_ascii=False)}]}
    for attempt in range(4):
        try:
            req=urllib.request.Request("https://integrate.api.nvidia.com/v1/chat/completions",data=json.dumps(body).encode(),headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
            r=json.load(urllib.request.urlopen(req,timeout=600)); out=r['choices'][0]['message']['content']
            m=re.search(r'\[.*\]',out,re.S); return json.loads(m.group(0))
        except Exception as e:
            print("retry",attempt,str(e)[:80],file=sys.stderr); time.sleep(5*(attempt+1))
    return []
gl={}
for i in range(0,len(items),60):
    for o in call(items[i:i+60]):
        w=o.get("word"); t=o.get("type")
        if w in c and t and t!="not_a_name":
            gl[w]={"type":t,"count":c[w]["count"],"books":c[w]["books"]}
    print(f"batch {i//60+1}: {len(gl)} names so far",file=sys.stderr)
out={"series":series,"rule":"Names, places, teams and pet names keep their English spelling in every language unless the style sheet says otherwise. Never translate, never decline into a different spelling, never invent a new name.","entries":dict(sorted(gl.items(),key=lambda kv:-kv[1]["count"]))}
json.dump(out,open(f'glossaries/{series}.json','w'),ensure_ascii=False,indent=1)
print(json.dumps({t:sum(1 for v in gl.values() if v["type"]==t) for t in set(v["type"] for v in gl.values())}))
print("characters:",[w for w,v in out["entries"].items() if v["type"]=="character"][:60])
print("places/teams:",[w for w,v in out["entries"].items() if v["type"] in("place","team_or_brand")][:40])
print("nicknames:",[w for w,v in out["entries"].items() if v["type"]=="nickname_or_petname"])
