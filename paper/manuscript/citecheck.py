import re, urllib.request, urllib.parse, difflib, xml.etree.ElementTree as ET, time
bib = open('references.bib').read()
cited = set(open('/tmp/claude-1000/-home-mohhakami-Documents-paper-quworld/af4d80b6-5cb2-4675-b69d-81a19e2a7c31/scratchpad/cited.txt').read().split())
entries = {}
for m in re.finditer(r'@(\w+)\{([^,]+),(.*?)\n\}', bib, re.S):
    entries[m.group(2).strip()] = m.group(3)
def field(body, name):
    m = re.search(name + r'\s*=\s*[{"](.*?)[}"]\s*,?\s*\n', body, re.S | re.I)
    return re.sub(r'[{}\\]', '', m.group(1)).strip() if m else ''
missing = [k for k in cited if k not in entries]
ids, noid = {}, []
for k in sorted(cited & set(entries)):
    b = entries[k]
    m = re.search(r'(\d{4}\.\d{4,5})', field(b, 'eprint') + ' ' + field(b, 'journal') + ' ' + field(b, 'url') + ' ' + field(b, 'note') + ' ' + field(b, 'howpublished'))
    if m: ids[k] = m.group(1)
    else: noid.append(k)
got = {}
for k, i in ids.items():
    try:
        req = urllib.request.Request('https://arxiv.org/abs/' + i, headers={'User-Agent': 'Mozilla/5.0 citecheck'})
        h = urllib.request.urlopen(req, timeout=30).read().decode('utf8', 'ignore')
        m = re.search(r'<meta name="citation_title" content="([^"]*)"', h)
        got[i] = (m.group(1) if m else None)
    except Exception as ex:
        got[i] = None; print('ERR', k, i, ex)
    time.sleep(0.7)
import html as H
got = {i: (H.unescape(t) if t else t) for i, t in got.items()}
norm = lambda s: re.sub(r'[^a-z0-9 ]', '', s.lower())
print('cited keys missing from bib:', missing)
bad = []
for k, i in ids.items():
    t = field(entries[k], 'title'); g = got.get(i)
    r = difflib.SequenceMatcher(None, norm(t), norm(g or '')).ratio()
    if g is None or r < 0.85: bad.append((k, i, round(r, 2), t[:70], (g or 'NOT FOUND')[:70]))
print(f'arXiv-checked {len(ids)}; mismatches {len(bad)}')
for b in bad: print(' ', b)
print('no arXiv id (check by hand):', noid)
