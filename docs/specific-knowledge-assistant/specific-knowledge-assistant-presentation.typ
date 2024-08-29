#import "@preview/polylux:0.3.1": *
#import themes.university: *

#show: university-theme.with(
  short-author: "Thanos Apostolou",
  short-title: "Development of an AI-Based System for Knowledge Specific Assistance",
  short-date: "2024-09",
  aspect-ratio: "16-9",
)
#set text(font: "Arial", weight: "regular", size: 20pt)

#title-slide(
  authors: ("Thanos Apostolou"),
  title: "Development of an AI-Based System for Knowledge Specific Assistance",
  // subtitle: "AI-Based Model for Knowledge Specific Assistance",
  date: "September 2024",
  institution-name: "University of Piraeus\n Department of Informatics",
  logo: image("images/unipi.png", width: 120pt),
)

#slide()[
  = Knowledge Specific Assistance

  Define the problem domain.

  - Documents in different formats about a specific knowledge field.

  - Investigation of Knowledge Assistance Approaches

  - Specific Knowledge Assistant (SKA) Application
]

#slide()[
  = Specific Knowledge Assistance Approaches
  Two different Methods

  == 1. Custom Text Generation Model Method
  Create custom text generation model from scratch

  == 2. Retrieval Augmented Generation (RAG) Method
  Use pre-trained LLMs together with RAG technique

]

#slide(
  )[
  == 1. Custom Text Generation Model Method
  - Create custom text generation model based on LSTM
  - Train this model with users' Documents
  - Invoke model by predicting the next word each time

  #table(columns: (1fr, 1fr), inset: 0pt, align: horizon, stroke: none, [
    #set align(start + top)
    === Advantages
    - Independence
    - Flexibility
  ], [
    #set align(start + top)
    === Disadvantages
    - Needs many sources
    - Highly consuming
    - Hard to implement
    - Hard to return sources
  ])
]

#slide(
  )[
  == 2. Retrieval Augmented Generation (RAG) Method
  - Split Documents in chunks and save in vector store.
  - Use pre-trained LLMs
  - Invoke LLM and instruct to answer only based on context from vector store.

  #table(columns: (1fr, 1fr), inset: 0pt, align: horizon, stroke: none, [
    #set align(start + top)
    === Advantages
    - Needs only relative sources
    - Adaptable
    - Easy to implement
    - Able to return sources
    - Prompting
  ], [
    #set align(start + top)
    === Disadvantages
    - Dependency on external LLMs
    - Inflexibility
  ])
]

#slide()[
= System Architecture

#table(columns: (1fr, 1fr), inset: 0pt, align: horizon, stroke: none, [
#set align(start + top)
== Our Components
- `ska_llm`
- `ska_cli`
- `ska_server`
- `ska_frontend`
], [
#set align(start + top)
== Extra Components
- `PostgreSQL`
- `Keycloak`
- `api_gateway`
])

#figure(
  image("images/component.drawio.png", height: 200pt),
  supplement: [IMAGE],
) <img_ska_system_components>

]

#slide(
  )[
  == Our Components

  #table(
    columns: (1fr, 1fr),
    inset: 0pt,
    align: horizon,
    stroke: none,
    [
      #figure(image("images/ska_llm.drawio.png", height: 165pt), supplement: [IMAGE]) <img_ska_llm_operations>

    ],
    [
      #figure(image("images/ska_cli.drawio.png", height: 165pt), supplement: [IMAGE]) <img_ska_cli_operations>

    ],
    [
      #figure(
        image("images/ska_server.drawio.png", height: 165pt),
        supplement: [IMAGE],
      ) <img_ska_server_operations>

    ],
    [
      #figure(
        image("images/ska_frontend.drawio.png", height: 165pt),
        supplement: [IMAGE],
      ) <img_ska_frontend_operations>

    ],
  )
]

#slide(
  )[
== Extra Components

#table(
  columns: (1fr, 1fr),
  inset: 0pt,
  align: horizon,
  stroke: none,
  [
    #set align(start + top)
    === PostgreSQL
    #figure(image("images/ska_schema.png", height: 180pt), supplement: [IMAGE]) <img_ska_schema>

  ],
  table.cell(rowspan: 2)[
  #set align(start + top)
  === Keycloak
  - `SKA_ADMIN`
  - `SKA_USER`
  - `SKA_GUEST`

  #figure(
    image("images/keyclaok_oauth2_pkce.png", height: 180pt),
    supplement: [IMAGE],
  ) <img_keyclaok_oauth2_pkce>

  ],
  [
  #set align(start + top)
  === API Gateway
  - "/app" to `ska_frontend`
  - "/backend" to `ska_server`
  - "/iam" to `Keycloak`
  ],
)
]

#slide()[
= SKA Application

#table(columns: (1fr, 1fr), inset: 0pt, align: horizon, stroke: none, [
#set align(start + top)
== CLI

```
Usage: app-cli <COMMAND>

Commands:
  model
  db
  help   Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

], [
  #set align(start + top)
  == GUI

  #figure(
    image("images/screenshot_home_loggedin.png", width: 100%),
    supplement: [IMAGE],
  ) <img_ska_home_loggedin>
])
]

#slide(
  )[
  == Usage skalm

  #table(columns: (1fr, 1fr), inset: 0pt, align: horizon, stroke: none, [
    #set align(start + top)
    === Create skalm chat
    #figure(
      image("images/screenshot_assistant_add_chat.png", width: 100%),
      supplement: [IMAGE],
    ) <img_ska_assistant_add_chat>

  ], [
    #set align(start + top)
    === Ask skalm question
    #set align(start + horizon)
    #figure(
      image("images/screenshot_assistant_skalm.png", width: 100%),
      supplement: [IMAGE],
    ) <img_ska_assistant_skalm>
  ])
]

#slide(
  )[
  == Usage Llama3

  #table(columns: (1fr, 1fr), inset: 0pt, align: horizon, stroke: none, [
    #set align(start + top)
    === Create Llama3 chat
    #figure(
      image("images/screenshot_assistant_add_llama3.png", width: 100%),
      supplement: [IMAGE],
    ) <img_ska_assistant_add_llama3>
  ], [
    #set align(start + top)
    === Ask Llama3 question
    #set align(start + horizon)
    #figure(
      image("images/screenshot_assistant_llama3.png", width: 100%),
      supplement: [IMAGE],
    ) <img_ska_assistant_our_llama3>
  ])
]

#slide(
  )[
  #table(columns: (1fr, 1fr), inset: 0pt, align: horizon, stroke: none, [
    #set align(start + top)
    == Conclusions
    - Custom text generation model was not able to perform well.
    - Custom text generation model method is not suitable for this task.
    - RAG method achieved the desired outcome.
    - System and infrastructure can be self hosted.
    - High resources are needed.
  ], [
    #set align(start + top)
    == Future Work
    - Deprecate support of the custom text generation model.
    - Utilize a supported GPU.
    - Support more generation formats like images, sounds and videos.
    - Add integrations with other applications.
  ])
]
#focus-slide()[
  #set align(center + horizon)
  #set text(size: 60pt)

  *Thank you!*
]

// #matrix-slide[
//   left
// ][
//   middle
// ][
//   right
// ]

// #matrix-slide(columns: 1)[
//   top
// ][
//   bottom
// ]

// #matrix-slide(columns: (1fr, 2fr, 1fr), ..(lorem(8),) * 9)