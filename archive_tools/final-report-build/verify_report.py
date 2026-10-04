"""Final automated PDF checks; visual inspection is recorded separately."""
from pathlib import Path
import hashlib,json,re
from pypdf import PdfReader
import pdfplumber

ROOT=Path(r'E:\Codex\2026-09-27\yo')
OUT=ROOT/'outputs/elec4240-final-report'
RESULTS=ROOT/'outputs/marigold-depth/results/final_extension_v5'

def main():
    source=(OUT/'final_report.tex').read_text(encoding='utf-8')
    body=source.split('\\end{filecontents*}\n',1)[1]
    assert '@@' not in body and 'Pending' not in body
    style=source.split('\\begin{filecontents*}{cvpr.sty}\n',1)[1].split('\n\\end{filecontents*}',1)[0]
    assert style.encode()==(ROOT/'work/milestone-build/cvpr.sty').read_bytes()
    abstract=body.split('\\begin{abstract}')[1].split('\\end{abstract}')[0]
    assert len(abstract.split())<=300
    pdf=OUT/'final_report.pdf';reader=PdfReader(pdf)
    assert 6<=len(reader.pages)<=8
    text='\n'.join(p.extract_text() for p in reader.pages)
    for field in ['HO, Chun Wai','Tong, Man Hung','MA, Shenhan','21053878','21064669','21041382']:
        assert field in text,field
    for value in ['0.11106','0.09632','0.07224','0.07038','0.37886','0.94306','0.18231','0.47903']:
        assert value in text or value[1:] in text,value
    ext=json.loads((RESULTS/'external_summary.json').read_text())
    for row in ext:
        value=f"{row['abs_rel']:.5f}"
        assert value in text or value[1:] in text,value
    fonts=set();chars=0
    with pdfplumber.open(pdf) as doc:
        for page in doc.pages:
            assert page.width==612 and page.height==792
            for c in page.chars:
                chars+=1;fonts.add(c['fontname'])
                assert c['x0']>=-0.5 and c['x1']<=612.5 and c['top']>=-0.5 and c['bottom']<=792.5,c
    assert any('NimbusRom' in name for name in fonts),sorted(fonts)
    log=(OUT/'final_report.log').read_text(encoding='utf-8',errors='replace')
    assert not re.search(r'Overfull \\[hv]box|undefined references|Citation .* undefined|Reference .* undefined',log)
    result={'status':'passed','pages':len(reader.pages),'abstract_whitespace_words':len(abstract.split()),
            'all_author_fields_verified':True,'all_external_aligned_means_verified':True,
            'original_style_exact':True,'page_format':'Letter; CVPR2022 10pt two-column',
            'fonts':sorted(fonts),'characters_within_page':chars,
            'no_overfull_boxes_or_undefined_references':True,
            'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256((OUT/'final_report.tex').read_bytes()).hexdigest()}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
