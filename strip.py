import os
import re
import glob

# Find all python files in current directory and subdirectories
files = glob.glob('**/*.py', recursive=True)

for f in files:
    # Skip this script
    if f == 'strip.py':
        continue
        
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Remove docstrings (anything between """)
    content = re.sub(r'"""[\s\S]*?"""\n?', '', content)
    
    # Remove fancy lines
    content = re.sub(r'# ═+[\r\n]+#\s*(.*?)[\r\n]+# ═+', lambda m: '# ' + m.group(1).lower(), content)
    content = re.sub(r'# ─+\s*(.*?)\s*─+', lambda m: '# ' + m.group(1).lower(), content)
    
    # Make all comments lowercase and remove extra spaces
    def simplify_comment(m):
        c = m.group(1).strip().lower()
        # strip out typical AI phrasing
        c = c.replace('parameters', '').replace('returns', '').replace('examples', '')
        if not c:
            return ''
        return '# ' + c

    # Replace comments except those in strings
    # A simple regex for comments
    content = re.sub(r'#(.*)', simplify_comment, content)
    
    # Remove multiple blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content.strip() + '\n')

print("All comments simplified and docstrings removed.")
