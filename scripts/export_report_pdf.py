"""Portable content-equivalent DOCX report PDF; pagination differs from Word.

Requires python-docx, reportlab and Pillow. Preserves body order and diagrams.
"""
from pathlib import Path
import io
import sys
from xml.sax.saxutils import escape


def export_pdf(source, target):
    from docx import Document
    from docx.oxml.ns import qn
    from docx.table import Table as WordTable
    from docx.text.paragraph import Paragraph as WordParagraph
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image, KeepTogether
    from PIL import Image as PILImage
    doc=Document(source);styles=getSampleStyleSheet()
    for name in ('Heading1','Heading2','Heading3'):
        styles[name].keepWithNext=True
    styles.add(ParagraphStyle('ReportBody',fontName='Helvetica',fontSize=10,leading=13,spaceAfter=4))
    styles.add(ParagraphStyle('ReportCell',fontName='Helvetica',fontSize=7,leading=9,wordWrap='CJK'))
    story=[];width=468
    for elem in doc.element.body:
        if elem.tag==qn('w:p'):
            p=WordParagraph(elem,doc)
            if p.text:
                style='Title' if p.style.name=='Title' else ('Heading'+p.style.name.split()[-1] if p.style.name.startswith('Heading ') else 'ReportBody')
                paragraph=Paragraph(escape(p.text),styles[style])
                paragraph._case_start=p.text.startswith(('hap-','sad-','ang-')) and style=='Heading2'
                story.append(paragraph)
            for blip in elem.findall('.//'+qn('a:blip')):
                part=doc.part.related_parts[blip.get(qn('r:embed'))]
                stream=io.BytesIO(part.blob)
                with PILImage.open(stream) as im:w,h=im.size
                stream.seek(0)
                picture=Image(stream,width=width,height=width*h/w)
                story.append(picture)
                story.append(Spacer(1,8))
        elif elem.tag==qn('w:tbl'):
            tbl=WordTable(elem,doc)
            rows=[[Paragraph(escape(c.text),styles['ReportCell']) for c in r.cells] for r in tbl.rows]
            widths=[53,35,50,90,64,55,45,76] if len(rows[0])==8 else [width/len(rows[0])]*len(rows[0])
            table=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E9EDF2')),
                ('GRID',(0,0),(-1,-1),.3,colors.grey),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            story.extend([table,Spacer(1,8)])
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    packed=[];case_group=None
    for item in story:
        if getattr(item,'_case_start',False):
            if case_group:packed.append(KeepTogether(case_group))
            case_group=[item]
        elif case_group is not None:case_group.append(item)
        else:packed.append(item)
    if case_group:packed.append(KeepTogether(case_group))
    def footer(canvas,document):
        canvas.setFont('Helvetica',8);canvas.drawString(72,30,'Model Mountain Emotion Lab | '+str(document.page))
    SimpleDocTemplate(str(target),pagesize=letter,leftMargin=72,rightMargin=72,topMargin=54,bottomMargin=54).build(packed,onFirstPage=footer,onLaterPages=footer)
    return target


if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit('Usage: python scripts/export_report_pdf.py INPUT.docx OUTPUT.pdf')
    print(export_pdf(sys.argv[1],sys.argv[2]))
