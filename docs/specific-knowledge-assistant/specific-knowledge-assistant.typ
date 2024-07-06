#set document(
  title: [AI-Based Model for Knowledge Specific Assistance],
  author: "Thanos Apostolou",
  date: datetime(year: 2024, month: 02, day: 1)
)
#set page(
  paper: "a4",
  margin: (x: 3cm, y: 3cm),
  header: locate(
    loc => if loc.page() > 1 {
        [
          #set text(font: "Arial",size: 8pt)
          #grid(columns: (1fr, 1fr), rows: 1,
          align(left)[
            MSc Thesis
          ],
          align(right)[
            #document.author.at(0)
          ])
        ]
    }
  ),
  footer: locate(
    loc => if loc.page() > 1 {
        [
          #set text(font: "Arial",size: 8pt)
          #grid(columns: (1fr, 1fr), rows: 1,
          align(left)[
            #document.title
          ],
          align(right)[
            #loc.page()
          ])
        ]
    }
  ),

)
#set text(
  font: "Arial",
  size: 10pt
)
#set par(
  justify: true,
  leading: 0.52em,
  first-line-indent: 10pt,
)
#set heading(numbering: "1.")

#image("images/unipi.png", width: 40pt)

#pagebreak()
#outline()

#pagebreak()
= Abstract

#pagebreak()
= Introduction

In this report, we will explore the
various factors that influence _fluid @harry
dynamics_ in glaciers and how they
contribute to the formation and
behaviour of these natural structures.

In this report, we will explore the
various factors that influence _fluid
dynamics_ in glaciers and how they
contribute to the formation and
behaviour of these natural structures.

In this report, we will explore the
various factors that influence _fluid
dynamics_ in glaciers and how they
contribute to the formation and
behaviour of these natural structures.

#pagebreak()
= Literature Review



In this report, we will explore the
various factors that influence _fluid
dynamics_ in glaciers and how they
contribute to the formation and
behaviour of these natural structures.



In this report, we will explore the
various factors that influence _fluid
dynamics_ in glaciers and how they
contribute to the formation and
behaviour of these natural structures.

#pagebreak()
= Technologies and Machine Learning Approaches

#pagebreak()
= Experimentation / Execution Examples

#pagebreak()
= Conclustions and Future Work

#pagebreak()
#bibliography("bibliography.yaml")