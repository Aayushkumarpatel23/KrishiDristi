"""Add Dataset Files tab button to nav."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HTML_PATH = os.path.join(ROOT, "frontend", "templates", "index.html")

with open(HTML_PATH, "r", encoding="utf-8") as f:
    h = f.read()

# Find exact bytes at the cleaning button
idx = h.find("tab-cleaning")
print("Context:", repr(h[idx-70:idx+100]))

# Replace
old = 'data-target="tab-cleaning">Data Cleaning & Quality</button>'
new_part = ('data-target="tab-datasets">\U0001f4c2 Dataset Files</button>\n'
            '    <button class="tab-btn px-4 py-2 rounded-lg text-slate-400 hover:text-white transition" '
            'data-target="tab-cleaning">\U0001f9f9 Data Cleaning & Quality</button>')
if old in h:
    h = h.replace(old, new_part, 1)  # only replace first occurrence (in nav)
    print("REPLACED")
else:
    print("NOT FOUND, trying alt...")
    # Some editors store &amp; 
    old2 = 'data-target="tab-cleaning">Data Cleaning &amp; Quality</button>'
    if old2 in h:
        h = h.replace(old2, new_part.replace("& Quality", "&amp; Quality"), 1)
        print("REPLACED with &amp; version")
    else:
        print("STILL NOT FOUND")

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(h)
print("Done")
