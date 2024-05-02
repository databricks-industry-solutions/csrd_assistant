from pyvis.network import Network
import os
import uuid

def displayGraph(graph):

  net = Network(
    height="750px", 
    width="100%", 
    directed=True, 
    cdn_resources='remote',
    notebook=True
  )

  net.options.groups = {
      "DIRECTIVE": {
        "icon": {
            "face": 'FontAwesome',
            "code": '\uf19c',
        }
      },
      "CHAPTER": {
          "icon": {
              "face": 'FontAwesome',
              "code": '\uf02d',
          }
      },
      "ARTICLE": {                 
        "icon": {
            "face": 'FontAwesome',
            "code": '\uf07c',
          }
      },
      "PARAGRAPH": {                 
        "icon": {
            "face": 'FontAwesome',
            "code": '\uf15b',
          }
      }
  }

  net.from_nx(graph)
  net.show(f"/tmp/{uuid.uuid4().hex}.html")
  return net.html.replace(
    '<head>',
    '<head><link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" type="text/css"/>'
  )