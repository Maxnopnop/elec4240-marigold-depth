"""Create submission packages, enforcing the uncompressed 10 MB code limit."""
from pathlib import Path
import hashlib, json, zipfile

ROOT=Path(r'E:\Codex\2026-09-27\yo')
REPO=ROOT/'outputs/marigold-depth'
OUT=ROOT/'outputs/elec4240-final-report'

def digest(data): return hashlib.sha256(data).hexdigest()

def archive(path, files):
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(files.items()): z.writestr(name,data)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert {i.filename: digest(z.read(i.filename)) for i in z.infolist()} == {n:digest(b) for n,b in files.items()}
    return {'file':path.name,'bytes':path.stat().st_size,'uncompressed_bytes':sum(map(len,files.values())),
            'entries':len(files),'sha256':digest(path.read_bytes())}

def main():
    check=json.loads((REPO/'results/final_extension_v5/delivery_checks.json').read_text())
    assert check['status']=='passed'
    pdf=OUT/'final_report.pdf';assert pdf.is_file()
    text=(OUT/'final_report.tex').read_text(encoding='utf-8');assert '@@' not in text.split('\\end{filecontents*}\n',1)[1]
    overleaf={p.relative_to(OUT).as_posix():p.read_bytes() for p in [OUT/'final_report.tex',*sorted((OUT/'figures').glob('*.png'))]}
    overleaf['README.txt']=b'Open final_report.tex as the main document and use pdfLaTeX. The unchanged official CVPR2022 style is embedded. The figures directory must remain beside the source.\n'
    a=archive(OUT/'overleaf_source.zip',overleaf)
    selected=[]
    for p in REPO.iterdir():
        if p.is_file() and p.suffix in {'.py','.ps1','.txt','.md'}: selected.append(p)
    # Summary/manifest files only, not per-image matrices or run histories.
    for folder in [REPO/'results',*sorted((REPO/'results').glob('*'))]:
        if not folder.is_dir(): continue
        selected.extend(p for p in folder.iterdir() if p.is_file() and p.suffix in {'.json','.csv','.md'})
    final=REPO/'results/final_extension_v5'
    selected.extend([final/'cost/summary.json',final/'cost/protocol.json',final/'cost/verification.json'])
    selected.extend(sorted((final/'failure').glob('*.json')))
    selected.extend(sorted((final/'figures').glob('*.png')))
    files={p.relative_to(REPO).as_posix():p.read_bytes() for p in selected}
    assert not any(Path(n).suffix.lower() in {'.pt','.pth','.npy','.npz','.safetensors','.mat'} for n in files)
    files['SUPPLEMENT_README.txt']=(
        'ELEC4240 final-report source supplement\n'
        'Authors: HO, Chun Wai 21053878; Tong, Man Hung 21064669; MA, Shenhan 21041382.\n'
        'Start with FINAL_REPRODUCE.md, REFERENCES.md and CORRECTIONS.md.\n'
        'Included: project source, protocols, selected manifests/summaries, final diagnostic figures.\n'
        'Excluded: datasets, weights, raw predictions, installed packages, full per-image result matrices and Git history.\n'
        'Completion markers are historical evidence, not substitutes for excluded raw artifacts.\n'
        'Published result links in README.md may refer to files kept only in the full repository.\n'
        'Exact replay requires separately prepared assets and upstream runs; a fresh reproduction must use separate result directories.\n'
        'Private repository: https://github.com/Maxnopnop/elec4240-marigold-depth\n'
        'Original six-test conclusions remain unchanged; the final extension is exploratory.\n'
    ).encode()
    files['SUPPLEMENT_MANIFEST.json']=json.dumps({n:{'bytes':len(b),'sha256':digest(b)} for n,b in sorted(files.items())},indent=2).encode()
    assert sum(map(len,files.values())) < 10_000_000, 'Uncompressed supplement exceeds 10 MB'
    b=archive(OUT/'supplementary_code.zip',files)
    assert b['bytes'] < 10_000_000
    report={'overleaf':a,'supplement':b,'pdf_sha256':digest(pdf.read_bytes())}
    (OUT/'package_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    qa=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    (OUT/'README.txt').write_text(
        'ELEC4240 final project deliverables\n\n'
        f"final_report.pdf: {qa['pages']} pages including references, official CVPR2022 format.\n"
        'final_report.tex and figures/: editable LaTeX source.\n'
        'overleaf_source.zip: upload as an Overleaf project; select final_report.tex.\n'
        f"supplementary_code.zip: {b['bytes']:,} compressed bytes; {b['uncompressed_bytes']:,} uncompressed bytes.\n"
        'The supplement excludes model checkpoints, datasets, raw prediction arrays and installed frameworks.\n'
        'verification.json and package_checks.json: automated PDF/archive checks and SHA256 values.\n\n'
        'Authors: HO, Chun Wai 21053878; Tong, Man Hung 21064669; MA, Shenhan 21041382.\n'
        'The final extension adds 15 matched low-resolution trainings and 9 fixed-time diagnostic trainings.\n'
        'It audits 7,890 control predictions, 576 budget-validation predictions, and 320 timed reference matches.\n'
        'Original prospective results are preserved; all later comparisons are explicitly exploratory.\n'
        'Read FINAL_REPRODUCE.md inside the code archive before running any script.\n'
        'No Canvas submission has been performed.\n'
        'Private GitHub repository: https://github.com/Maxnopnop/elec4240-marigold-depth\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
