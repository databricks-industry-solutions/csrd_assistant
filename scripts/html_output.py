import re

def article_html(label, content):
  content = re.sub('\n', '<br>', content)
  return f'''<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css?family=DM Sans" rel="stylesheet"/>
    <style>
    h1, h2, h3, h4, h5, h6, .h1, .h2, .h3, .h4, .h5, .h6 {{ margin-bottom: 0.5rem; font-family: "DM Sans"; color: inherit; }}
    body {{ font-family: "DM Sans";}}
    .jumbotron {{ background-color: #f8fafc; }}
    </style>
</head>
<body>
<div class="container jumbotron">
  <h3>{label}</h3><br>
  <p><small>{content}</small></p><br>
</div>
</body>
</html>'''

def references_html(label, content, references):
  content = re.sub('\n', '<br>', content)
  reference_list = ['<ul class="list-group">']
  reference_list.extend([f'<li class="list-group-item"><small>CSRD §{ref}</small></li>' for ref in references])
  reference_list.append('</ul>')
  reference_list = '\n'.join(reference_list)
  return f'''<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css?family=DM Sans" rel="stylesheet"/>
    <style>
    h1, h2, h3, h4, h5, h6, .h1, .h2, .h3, .h4, .h5, .h6 {{ margin-bottom: 0.5rem; font-family: "DM Sans"; color: inherit; }}
    body {{ font-family: "DM Sans";}}
    .jumbotron {{ background-color: #f8fafc; }}
    </style>
</head>
<body>
<div class="container jumbotron">
  <h3>{label}</h3><br>
  <p><small>{content}</small></p><br>
  <h3>References</h3><br>
  <p>{reference_list}</p><br>
</div>
</body>
</html>'''

def vector_html(label, content, relevance):
  content = re.sub('\n', '<br>', content)
  return f'''<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css?family=DM Sans" rel="stylesheet"/>
    <style>
    h1, h2, h3, h4, h5, h6, .h1, .h2, .h3, .h4, .h5, .h6 {{ margin-bottom: 0.5rem; font-family: "DM Sans"; color: inherit; }}
    body {{ font-family: "DM Sans";}}
    .jumbotron {{ background-color: #f8fafc; }}
    </style>
</head>
<body>
<div class="container jumbotron">
  <h3>{label}</h3><br>
  <h5 class="card-subtitle text-muted">relevance: {relevance}</h5><br>
  <p><small>{content}</small></p><br>
</div>
</body>
</html>'''

def llm_html(question, response):
  response = re.sub('\n', '<br>', response)
  return f'''<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css?family=DM Sans" rel="stylesheet"/>
    <style>
    h1, h2, h3, h4, h5, h6, .h1, .h2, .h3, .h4, .h5, .h6 {{ margin-bottom: 0.5rem; font-family: "DM Sans"; color: inherit; }}
    body {{ font-family: "DM Sans";}}
    .jumbotron {{ background-color: #f8fafc; }}
    </style>
</head>
<body>
<div class="container jumbotron">
  <h3>Question</h3><br>
  <p><small>{question}</small></p><br>
  <h3>Response</h3><br>
  <p><small>{response}</small></p><br>
</div>
</body>
</html>'''

def rag_html(question, response, supporting_docs):
  response = re.sub('\n', '<br>', response)
  docs = ['<ul class="list-group">']
  for doc in supporting_docs:
    content = re.sub('\n', '<br>', doc.page_content)
    docs.append(f'<li class="list-group-item"><small>{content}</small></li>')
  docs.append('</ul>')
  docs = '\n'.join(docs)
  return f'''<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css?family=DM Sans" rel="stylesheet"/>
    <style>
    h1, h2, h3, h4, h5, h6, .h1, .h2, .h3, .h4, .h5, .h6 {{ margin-bottom: 0.5rem; font-family: "DM Sans"; color: inherit; }}
    body {{ font-family: "DM Sans";}}
    .jumbotron {{ background-color: #f8fafc; }}
    </style>
</head>
<body>
<div class="container jumbotron">
  <h3>Question</h3><br>
  <p><small>{question}</small></p><br>
  <h3>Response</h3><br>
  <p><small>{response}</small></p><br>
  <h3>Facts</h3><br>
  {docs}
</div>
</body>
</html>'''

def rag_kg_html(question, response, supporting_docs):
  response = re.sub('\n', '<br>', response)
  docs = ['<ul class="list-group">']
  for doc in supporting_docs:
    content = re.sub('\n', '<br>', doc.page_content)
    docs.append(f'<li class="list-group-item"><small>{content}</small></li>')
  docs.append('</ul>')
  docs = '\n'.join(docs)
  return f'''<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css?family=DM Sans" rel="stylesheet"/>
    <style>
    h1, h2, h3, h4, h5, h6, .h1, .h2, .h3, .h4, .h5, .h6 {{ margin-bottom: 0.5rem; font-family: "DM Sans"; color: inherit; }}
    body {{ font-family: "DM Sans";}}
    .jumbotron {{ background-color: #f8fafc; }}
    </style>
</head>
<body>
<div class="container jumbotron">
  <h3>Question</h3><br>
  <p><small>{question}</small></p><br>
  <h3>Response</h3><br>
  <p><small>{response}</small></p><br>
  <h3>Facts & References</h3><br>
  {docs}
</div>
</body>
</html>'''