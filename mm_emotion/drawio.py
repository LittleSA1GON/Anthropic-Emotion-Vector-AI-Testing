"""Generate native editable diagrams.net (.drawio) documents without a web dependency."""
from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET


def new_document():
    return ET.Element("mxfile",{"host":"app.diagrams.net","modified":"2026-10-29T00:00:00.000Z", "agent":"Model Mountain CMPE187", "version":"24.7.0","type":"device"})


def page(doc,title):
    dia=ET.SubElement(doc,"diagram",{"name":title,"id":title.lower().replace(' ','-')})
    graph=ET.SubElement(dia,"mxGraphModel",{"dx":"1600","dy":"1000","grid":"1","gridSize":"10","guides":"1","tooltips":"1","connect":"1","arrows":"1","fold":"1","page":"1","pageScale":"1","pageWidth":"1450","pageHeight":"1000","math":"0","shadow":"0"})
    root=ET.SubElement(graph,"root")
    ET.SubElement(root,"mxCell",{"id":"0"}); ET.SubElement(root,"mxCell",{"id":"1","parent":"0"})
    return root


def box(root,uid,text,x,y,w=210,h=50,style=None):
    style=style or "rounded=1;whiteSpace=wrap;html=0;fillColor=#F3F6FB;strokeColor=#64748B;fontColor=#172554;fontSize=15;spacing=8;"
    obj=ET.SubElement(root,"mxCell",{"id":uid,"value":text,"style":style,"vertex":"1","parent":"1"})
    ET.SubElement(obj,"mxGeometry",{"x":str(x),"y":str(y),"width":str(w),"height":str(h),"as":"geometry"})


def edge(root,uid,src,target,label=""):
    obj=ET.SubElement(root,"mxCell",{"id":uid,"value":label,"style":"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=0;endArrow=block;endFill=1;strokeColor=#64748B;strokeWidth=2;","edge":"1","parent":"1","source":src,"target":target})
    ET.SubElement(obj,"mxGeometry",{"relative":"1","as":"geometry"})


def make_diagrams(outdir):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    doc=new_document()
    # 2.1 context model
    rt=page(doc,"2.1 Context Model")
    box(rt,"context","Input context",45,395,170,60)
    box(rt,"abstract","Abstraction",280,230);box(rt,"diction","Diction",280,580)
    edge(rt,"e1","context","abstract");edge(rt,"e2","context","diction")
    for i,t in enumerate(("Objective","Stated","Enacted")):
        uid=f"a{i}";box(rt,uid,t,570,135+i*105);edge(rt,f"ea{i}","abstract",uid)
    for i,t in enumerate(("Standard","Expressive")):
        uid=f"d{i}";box(rt,uid,t,570,535+i*105);edge(rt,f"ed{i}","diction",uid)
    box(rt,"contextnote","Abstraction: how emotion is expressed. Diction: word-choice style. Source: Deliverable #2A, p.3.",855,350,370,105, "rounded=0;whiteSpace=wrap;fillColor=#EFF6FF;strokeColor=#93C5FD;fontSize=14;spacing=12;")
    # 2.2 input
    rt=page(doc,"2.2 Input Classification")
    box(rt,"start","Text prompt",40,355,180,60)
    for i,(e,x) in enumerate((("Happiness",160),("Sadness",365),("Anger",570))):
        uid=f"e{i}";box(rt,uid,e,290,x,205,60);edge(rt,f"p{i}","start",uid)
        for j,intensity in enumerate(("Neutral","Moderate","High")):
            child=f"{uid}-{j}";box(rt,child,intensity,620,x-54+j*58,190,48)
            edge(rt,f"{uid}-link{j}",uid,child)
    box(rt,"footnote","3 emotions x 3 intensities. Full test matrix also crosses 3 abstraction x 2 diction = 18 primary cases.",925,310,330,110,"rounded=0;whiteSpace=wrap;fillColor=#EFF6FF;strokeColor=#93C5FD;fontSize=14;spacing=12;")
    # 2.3 output flow
    rt=page(doc,"2.3 Output Event Action")
    cells=[("capture","Capture hidden state",50,340), ("delta","Delta from matched neutral",310,340),
           ("cos","Cosine vs trained baseline",600,340), ("classify","Correct emotion?",895,230),
           ("range","Intensity range reviewed?",895,430), ("valid","Valid / Invalid / Ungraded",1135,340)]
    for uid,t,x,y in cells:box(rt,uid,t,x,y,210,68)
    for i,(a,b) in enumerate((("capture","delta"),("delta","cos"),("cos","classify"),("classify","range"),("range","valid"))):edge(rt,f"o{i}",a,b)
    box(rt,"caution","The PDF contains reversed inequality bounds. Proposed corrections are unapproved; retain UNGRADED until reviewed. Zero-norm activation deltas are unscorable.",400,580,670,108,"rounded=0;whiteSpace=wrap;fillColor=#FFF7ED;strokeColor=#FDBA74;fontSize=15;spacing=13;")
    # 2.4 decision table editable
    rt=page(doc,"2.4 Decision Table")
    box(rt,"header","Context classification decision table (all 3 emotions repeat these 6 rows)",40,65,1150,65)
    headers=["Abstraction","Diction","Intensity","Proposed cosine range","Original PDF bound status"]
    widths=[175,170,205,210,270]; xpos=[40,220,395,605,820]
    for j,h in enumerate(headers):
        box(rt,f"h{j}",h,xpos[j],160,widths[j],55,"rounded=0;whiteSpace=wrap;fillColor=#DBEAFE;strokeColor=#93C5FD;fontStyle=1;fontSize=13;")
    rows=[("Objective","Standard","Neutral","<=0.20","Consistent"),
          ("Objective","Expressive","Neutral/Moderate","0.20 to 0.50","Reversed inequality"),
          ("Stated","Standard","Moderate","0.50 to 0.60","Reversed inequality"),
          ("Stated","Expressive","Moderate","0.60 to 0.70","Reversed inequality"),
          ("Enacted","Standard","Moderate/High","0.70 to 0.80","Reversed inequality"),
          ("Enacted","Expressive","High","0.70 to 1.00","Reversed inequality")]
    for i,row in enumerate(rows):
        for j,v in enumerate(row):box(rt,f"r{i}c{j}",v,xpos[j],220+i*77,widths[j],68,
            "rounded=0;whiteSpace=wrap;fillColor=#FFFFFF;strokeColor=#CBD5E1;fontSize=13;spacing=7;")
    box(rt,"note","Operational thresholds are proposed, not validated; explicit approval is required to assign pass/fail. Neutral rows are intentionally labeled neutral even though original tables show the target emotion.",50,730,1100,85,"rounded=0;whiteSpace=wrap;fillColor=#FFF7ED;strokeColor=#FDBA74;fontSize=14;spacing=10;")
    multi=out/"CMPE187_All_Editable_DrawIO.drawio"
    ET.indent(doc,space="  ")
    ET.ElementTree(doc).write(multi,encoding="utf-8",xml_declaration=True)
    for d in list(doc):
        separate=new_document()
        separate.append(ET.fromstring(ET.tostring(d)))
        ET.indent(separate,space="  ")
        ET.ElementTree(separate).write(out/(d.attrib["name"].replace(" ","_")+".drawio"),encoding="utf-8",xml_declaration=True)
    return multi

if __name__=="__main__":
    import sys
    print(make_diagrams(sys.argv[1] if len(sys.argv)>1 else "course/drawio"))
