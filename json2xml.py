# -*- coding: utf-8 -*-

from __future__ import print_function

import io
import os
import json
import argparse
from utils import *
import operator
import time
import logging

def datainstance2file(inst, xml_gold, xml_sys):
  linkslist={}
  id = inst['id']
  xmlfile = open(xml_gold+id, 'r')
  lines = xmlfile.readlines()
  n = len(inst['entities'])
  writefile = open(xml_sys+id, 'w')
  for line in lines:
    if "<TAGS>" in line:
      writefile.write(line)
      
      for ent in inst['entities']:
        etype = ent['label']
        eid = ent['entity_id']
        entity = ent['name']
        etext = entity.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')
        start = ent['start']
        end = ent['end']
        emod = "NA"
        epol = "NA"
        if etype in ['ADMISSION', 'DISCHARGE']:
          val =""
          writefile.write('<SECTIME id="S{}" start="{}" end="{}" text="{}" type="{}" dvalue="{}" />'.format(str(eid), str(start), str(end), etext, etype, val) + '\n')
          writefile.write('<TIMEX3 id="T{}" start="{}" end="{}" text="{}" type="{}" val="{}" mod="NA" />'.format(str(eid), str(start), str(end), etext, 'DATE', val)+ '\n')
        elif etype in ['TREATMENT', 'OCCURRENCE', 'TEST', 'CLINICAL_DEPT', 'PROBLEM', 'EVIDENTIAL']:
          writefile.write('<EVENT id="E{}" start="{}" end="{}" text="{}" modality="{}" polarity="{}" type="{}"/>'.format(str(eid), str(start),str(end),etext,emod, epol, etype)+ '\n')
        else:
          val = ""
          writefile.write('<TIMEX3 id="T{}" start="{}" end="{}" text="{}" type="{}" val="{}" mod="NA" />'.format(str(eid), str(start), str(end), etext, etype, val)+ '\n')
      
      tlno = 0
      for rel in inst['interactions']:
        p1, p2 = rel['participants']
        reltype = "TL"
        e1, e2 = None, None
        for ent in inst['entities']:
          eid = ent['entity_id']
          if eid == p1:
            entity1 = ent['name']
            entity1 = entity1.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')
            if ent['label'] in ['ADMISSION', 'DISCHARGE']:
              reltype = "SECTIME"
            if ent['label'] in ['TREATMENT', 'OCCURRENCE', 'TEST', 'CLINICAL_DEPT', 'PROBLEM', 'EVIDENTIAL']:
              e1 = "E"+str(p1)
            else:
              e1 = "T"+str(p1)
          elif eid==p2:
            entity2 = ent['name']
            entity2 = entity2.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')
            
            if ent['label'] in ['TREATMENT', 'OCCURRENCE', 'TEST', 'CLINICAL_DEPT', 'PROBLEM', 'EVIDENTIAL']:
              e2 = "E"+str(p2)
            else:
              e2 = "T"+str(p2)
          if e1 and e2:
            break
        
        rtype = rel['label']
        rprob = rel["probability"]
        relno=tlno
        tlno+=1
        try:
          linkslist['<TLINK id="{}{}" fromID="{}" fromText="{}" toID="{}" toText="{}" type="{}" />'.format(reltype, str(relno), e1, entity1, e2, entity2, rtype)+ '\n']= rprob#0 if rtype=="AFTER" else 1 if rtype=="BEFORE" else 2# rprob
        except:
          print(rel)
          raise
        
      for line, prob in linkslist.items():  
        writefile.write(line)

    elif "<EVENT" in line: continue
    elif "<TLINK" in line: continue
    elif "<TIMEX3" in line: continue
    elif "<SECTIME" in line: continue
    else: writefile.write(line)
  writefile.close()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-s', '--split', default='test', type =str, choices = SPLITS)
  parser.add_argument('-n', '--modelname', default='model_basic', type =str)
  args = parser.parse_args()
  xmlpath = f'output/xml/{args.split}/system{args.modelname.replace("model","")}/'
    
  create_dir_if_not_exist(xmlpath)
  print(f"Evaluating {args.modelname} on {args.split}")
  output_json_path = os.path.join(f'output/json/', args.split+'_'+args.modelname+'_predictions.json')
    
  with open(output_json_path, 'r') as f:
    predictions = json.load(f)
    
  for id in predictions.keys():
      datainstance2file(predictions[id], f'output/xml/{args.split}/gold/', xmlpath)#, noTT=False, sort=False, TTfirst=False)

if __name__ == "__main__":
  main()

