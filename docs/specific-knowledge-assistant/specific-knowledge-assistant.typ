#let title_english = "AI-Based Model for Knowledge Specific Assistance"
#let title_greek = "Μοντέλο Τεχνητής Νοημοσύνης για Βοήθεια σε Συγκεκριμένη Γνώση"
#let author_engish = "Thanos Apostolou"
#let author_greek = "Θάνος Αποστόλου"

#set document(
  title: title_english,
  author: author_engish,
  date: datetime(year: 2024, month: 02, day: 1)
)
#set page(
  paper: "a4",
  margin: (x: 3cm, y: 3cm),
  header: locate(
    loc => if loc.page() > 2 {
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
    loc => if loc.page() > 2 {
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
#set text(font: "Arial", size: 10pt)
#set heading(numbering: "1.")
#show heading.where(
  level: 1
): it => [
  #set text(font: "Arial", size: 12pt, weight: "black")
  // #v(18pt)
  #it
  #v(6pt)
]
#show heading.where(
  level: 2
): it => [
  #set text(font: "Arial", size: 11pt, weight: "black")
  #v(18pt)
  #it
  #v(6pt)
]
#show heading.where(
  level: 3
): it => [
  #set text(font: "Arial", size: 10pt, weight: "black")
  #v(18pt)
  #it
  #v(6pt)
]
#show figure: it => [
  #set text(font: "Arial", size: 9pt, weight: "bold")
  #it
]

// #figure(
//    image("images/unipi.png"),
//    caption: [here comes text],
//    supplement: [IMAGE],
//  ) <test_img>

// FIRST PAGE

#set align(center)
#image("images/unipi.png", width: 70pt, height: 70pt)
#text(weight: "black", size: 12pt, [UNIVERISTY OF PIRAEUS - DEPARTMENT OF INFORMATICS])

#text(weight: "regular", size: 12pt, [ΠΑΝΕΠΙΣΤΗΜΙΟ ΠΕΙΡΑΙΩΣ - ΤΜΗΜΑ ΠΛΗΡΟΦΟΡΙΚΗΣ])

#v(10pt)

#text(weight: "black", size: 12pt, [MSc #sym.quote.angle.l.double#title_english#sym.quote.angle.r.double])

#text(weight: "regular", size: 12pt, [ΠΜΣ #sym.quote.angle.l.double#title_greek#sym.quote.angle.r.double])

#v(20pt)

#text(weight: "black", size: 12pt, [#underline([MSc Thesis])])

#text(weight: "regular", size: 12pt, [#underline([Μεταπτυχιακή Διατριβή])])

#v(20pt)

#set align(left)
#table(
  columns: (0.55fr, 1fr),
  inset: 5pt,
  align: horizon,
  // table.header(
  //   [], [*Area*], [*Parameters*],
  // ),
  [
    #text(weight: "black", [Thesis Title:])

    #text(weight: "regular", [Τίτλος Διατριβής:])
  ],
  [
    #text(weight: "black", [#title_english])

    #text(weight: "regular", [#title_greek])
  ],
  [
    #text(weight: "black", [Student's name-surname:])

    #text(weight: "regular", [Ονοματεπώνυμο φοιτητή:])
  ],
  [
    #text(weight: "black", [#author_engish])

    #text(weight: "regular", [#author_greek])
  ],
  [
    #text(weight: "black", [Father's name:])

    #text(weight: "regular", [Πατρώνυμο:])
  ],
  [
    #text(weight: "black", [Christos])

    #text(weight: "regular", [Χρήστος])
  ],
  [
    #text(weight: "black", [Student's ID No:])

    #text(weight: "regular", [Αριθμός Μητρώου:])
  ],
  [
    #text(weight: "black", [MPSP2203])

    #text(weight: "regular", [ΜΠΣΠ2203])
  ],
  [
    #text(weight: "black", [Supervisor:])

    #text(weight: "regular", [Επιβλέπων:])
  ],
  [
    #text(weight: "black", [Dionisios Sotiropoulos, Assistant Professor])

    #text(weight: "regular", [Διονύσιος Σωτηρόπουλος, Επίκουρος Καθηγητής])
  ],
)

#align(center + bottom, [September 2024/ Σεπτέμβριος 2024])
#v(20pt)
#line(length: 100%)

#pagebreak()
#align(center + bottom, [
  #table(
    columns: (1fr, 1fr, 1fr),
    inset: 5pt,
    align: horizon,
    table.cell(colspan: 3)[
      #v(20pt)
      #text(weight: "black", [3-Member Examination Committee])

      #text(weight: "regular", [Τριμελής Εξεταστική Επιτροπή])
      #v(20pt)
    ],
    [
      #v(10pt)
      #text(weight: "black", [Dionisios Sotiropoulos])
      #text(weight: "black", [Assistant Professor])

      #v(10pt)

      #text(weight: "regular", [Διονύσιος Σωτηρόπουλος])
      #text(weight: "regular", [Επίκουρος Καθηγητής])
      #v(10pt)
    ],
    [
      #v(10pt)

      #v(10pt)

      #v(10pt)
    ],
    [
      #v(10pt)

      #v(10pt)

      #v(10pt)
    ]
  )
])

#pagebreak()
#outline()

#set par(
  justify: true,
  leading: 7pt,
  first-line-indent: 10pt,
)
#show par: it => [
  #set block(spacing: 10pt)
  #it
  // #v(3pt)
]

#pagebreak()
= Abstract

== test

contribute to the formation and
behaviour of these natural structures.

=== test2

contribute to the formation and
behaviour of these natural structures.

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