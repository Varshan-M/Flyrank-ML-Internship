import os
import re

with open('work/capstone_report.md', 'r', encoding='utf-8') as f:
    markdown_content = f.read()

html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Capstone Report: FlyRank</title>
    <style>
        :root {{
            --bg-color: #0d1117;
            --text-color: #c9d1d9;
            --accent-color: #58a6ff;
            --border-color: #30363d;
            --card-bg: #161b22;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--bg-color);
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
        }}
        h1, h2, h3 {{
            color: #ffffff;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.3em;
        }}
        a {{
            color: var(--accent-color);
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        ul {{
            padding-left: 20px;
        }}
        code {{
            background: rgba(110,118,129,0.4);
            padding: 0.2em 0.4em;
            border-radius: 6px;
            font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 85%;
        }}
        pre {{
            background: var(--card-bg);
            padding: 16px;
            border-radius: 6px;
            overflow: auto;
            border: 1px solid var(--border-color);
        }}
    </style>
    <!-- Use marked.js to render Markdown -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
    <div id="content" class="card"></div>
    <script>
        const markdown = `{0}`;
        document.getElementById('content').innerHTML = marked.parse(markdown);
    </script>
</body>
</html>
""".format(markdown_content.replace('`', '\\`').replace('$', '\\$'))

os.makedirs('docs', exist_ok=True)
with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(html_template)

print("Generated docs/index.html")
