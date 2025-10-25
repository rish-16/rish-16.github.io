import json
import os
from pathlib import Path

# Read all markdown files from blogposts directory
blogposts_dir = Path('essays')
index_path = blogposts_dir / 'index.json'

# Read the index.json file
with open(index_path, 'r', encoding='utf-8') as f:
    index = json.load(f)

# Read all markdown files
posts = []
for post in index:
    posts.append({
        **post,
    })

# Generate the HTML file with embedded posts
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Essays</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Funnel+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/11.1.1/marked.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
  <style>
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}

    body {{
      font-family: 'Funnel Sans', sans-serif;
      background: #fff;
      color: #000;
    }}

    nav {{
      border-bottom: 1px solid #e0e0e0;
      padding: 1.5rem 0;
    }}

    nav ul {{
      list-style: none;
      display: flex;
      gap: 2rem;
      max-width: 800px;
      margin: 0 auto;
      padding: 0 2rem;
    }}

    nav a {{
      text-decoration: none;
      color: #000;
      font-size: 1rem;
      font-weight: 400;
    }}

    nav a:hover {{
      text-decoration: underline;
    }}

    .container {{
      max-width: 800px;
      margin: 0 auto;
      padding: 3rem 2rem;
    }}

    h1 {{
      font-size: 2.5rem;
      font-weight: 700;
      margin-bottom: 2rem;
    }}

    .substack-notice {{
      margin-bottom: 2rem;
      padding-bottom: 2rem;
      border-bottom: 1px solid #e0e0e0;
    }}

    .substack-notice a {{
      color: #00b894;
      text-decoration: none;
    }}

    .substack-notice a:hover {{
      text-decoration: underline;
    }}

    .blog-list {{
      display: flex;
      flex-direction: column;
      gap: 3pt;
    }}

    .blog-post {{
      padding-bottom: 0;
    }}

    .blog-post:last-child {{
      border-bottom: none;
    }}

    .post-date {{
      color: #666;
      font-size: 0.9rem;
      margin-bottom: 0.5rem;
    }}

    .post-title {{
      font-size: 16;
      font-weight: 600;
      margin: 0;
      margin-bottom: 0;
    }}

    .post-title a {{
      color: #00b894;
      text-decoration: none;
    }}

    .post-title a:hover {{
      text-decoration: underline;
    }}

    .post-excerpt {{
      color: #333;
      line-height: 1.6;
    }}

    .blog-content {{
      line-height: 1.8;
    }}

    .blog-content h1 {{
      margin-top: 2rem;
      margin-bottom: 1rem;
      font-size: 2rem;
    }}

    .blog-content h2 {{
      margin-top: 2rem;
      margin-bottom: 1rem;
      font-size: 1.8rem;
    }}

    .blog-content h3 {{
      margin-top: 1.5rem;
      margin-bottom: 0.75rem;
      font-size: 1.4rem;
    }}

    .blog-content p {{
      margin-bottom: 1rem;
    }}

    .blog-content img {{
      max-width: 100%;
      height: auto;
      margin: 1.5rem 0;
    }}

    .blog-content pre {{
      background: #f6f8fa;
      padding: 1rem;
      border-radius: 6px;
      overflow-x: auto;
      margin: 1.5rem 0;
    }}

    .blog-content code {{
      font-family: 'Monaco', 'Courier New', monospace;
      font-size: 0.9rem;
    }}

    .blog-content p code {{
      background: #f6f8fa;
      padding: 0.2rem 0.4rem;
      border-radius: 3px;
    }}

    .blog-content ul, .blog-content ol {{
      margin-left: 2rem;
      margin-bottom: 1rem;
    }}

    .blog-content li {{
      margin-bottom: 0.5rem;
    }}

    .back-link {{
      display: inline-block;
      margin-bottom: 2rem;
      color: #00b894;
      text-decoration: none;
    }}

    .back-link:hover {{
      text-decoration: underline;
    }}

    .error {{
      color: #d32f2f;
      padding: 1rem;
      background: #ffebee;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <nav>
    <ul>
      <li><a href="/">Home</a></li>
      <li><a href="/blog.html">Blog</a></li>
      <li><a href="/essays.html">Essays</a></li>
    </ul>
  </nav>

  <div class="container" id="app">
    <div id="content"></div>
  </div>

  <script>
    // Embedded blog posts data
    const blogPosts = {json.dumps(posts, indent=2, ensure_ascii=False)};

    // Configure marked for syntax highlighting
    marked.setOptions({{
      highlight: function(code, lang) {{
        if (lang && hljs.getLanguage(lang)) {{
          return hljs.highlight(code, {{ language: lang }}).value;
        }}
        return hljs.highlightAuto(code).value;
      }}
    }});

    // Parse frontmatter from markdown
    function parseFrontmatter(content) {{
      const frontmatterRegex = /^---\\s*\\n([\\s\\S]*?)\\n---\\s*\\n([\\s\\S]*)$/;
      const match = content.match(frontmatterRegex);
      
      if (!match) {{
        return {{ metadata: {{}}, content: content }};
      }}
      
      const frontmatter = match[1];
      const markdown = match[2];
      const metadata = {{}};
      
      frontmatter.split('\\n').forEach(line => {{
        const [key, ...valueParts] = line.split(':');
        if (key && valueParts.length) {{
          metadata[key.trim()] = valueParts.join(':').trim();
        }}
      }});
      
      return {{ metadata, content: markdown }};
    }}

    function renderBlogList() {{
      const content = document.getElementById('content');
      content.innerHTML = `
        <h1>Essays</h1>
        <div class="blog-list">
          ${{blogPosts.map(post => `
            <div class="blog-post">
              <div class="post-title">
                <a href="${{post.filename}}">${{post.title}}</a>
              </div>
            </div>
          `).join('')}}
        </div>
      `;
    }}

    function renderBlogPost(filename) {{
      const post = blogPosts.find(p => p.filename === filename);
      const content = document.getElementById('content');
      
      if (!post) {{
        content.innerHTML = `
          <a href="#/" class="back-link">← Back to Essays</a>
          <div class="error">Essay not found.</div>
        `;
        return;
      }}

      const {{ metadata, content: markdownContent }} = parseFrontmatter(post.content);
      const htmlContent = marked.parse(markdownContent);
      
      content.innerHTML = `
        <a href="#/" class="back-link">← Back to Essays</a>
        ${{metadata.date || post.date ? `<div class="post-date">${{metadata.date || post.date}}</div>` : ''}}
        <div class="blog-content">${{htmlContent}}</div>
      `;

      // Render LaTeX equations
      renderMathInElement(document.getElementById('content'), {{
        delimiters: [
          {{left: '$$', right: '$$', display: true}},
          {{left: '$', right: '$', display: false}}
        ],
        throwOnError: false
      }});
    }}

    function router() {{
      const hash = window.location.hash.slice(1) || '/';
      
      if (hash === '/') {{
        renderBlogList();
      }} else if (hash.startsWith('/blog/')) {{
        const filename = hash.replace('/blog/', '');
        renderBlogPost(filename);
      }}
    }}

    window.addEventListener('hashchange', router);
    window.addEventListener('load', router);
  </script>
</body>
</html>"""

# Write the generated HTML file
with open('essays.html', 'w', encoding='utf-8') as f:
    f.write(html)

print('✅ Blog generated successfully! blog.html is ready for deployment.')
