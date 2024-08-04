#let title_english = "AI-Based Model for Knowledge Specific Assistance"
#let title_greek = "Μοντέλο Τεχνητής Νοημοσύνης για Βοήθεια σε Συγκεκριμένη Γνώση"
#let author_english = "Thanos Apostolou"
#let author_greek = "Θάνος Αποστόλου"

#set document(
  title: title_english,
  author: author_english,
  date: datetime(year: 2024, month: 02, day: 1),
)
#set page(
  paper: "a4",
  margin: (x: 3cm, y: 3cm),
  header: locate(loc => if loc.page() > 2 {
    [
      #set text(font: "Arial", size: 8pt)
      #grid(columns: (1fr, 1fr), rows: 1, align(left)[
        MSc Thesis
      ], align(right)[
        #document.author.at(0)
      ])
    ]
  }),
  footer: locate(loc => if loc.page() > 2 {
    [
      #set text(font: "Arial", size: 8pt)
      #grid(columns: (1fr, 1fr), rows: 1, align(left)[
        #document.title
      ], align(right)[
        #loc.page()
      ])
    ]
  }),
)
#set text(font: "Arial", size: 10pt)
#show heading.where(level: 1): it => [
  #set text(font: "Arial", size: 12pt, weight: "black")
  // #v(18pt)
  #it
  #v(6pt)
]
#show heading.where(level: 2): it => [
  #set text(font: "Arial", size: 11pt, weight: "black")
  #v(18pt)
  #it
  #v(6pt)
]
#show heading.where(level: 3): it => [
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
#text(
  weight: "black",
  size: 12pt,
  [UNIVERISTY OF PIRAEUS - DEPARTMENT OF INFORMATICS],
)

#text(
  weight: "regular",
  size: 12pt,
  [ΠΑΝΕΠΙΣΤΗΜΙΟ ΠΕΙΡΑΙΩΣ - ΤΜΗΜΑ ΠΛΗΡΟΦΟΡΙΚΗΣ],
)

#v(10pt)

#text(
  weight: "black",
  size: 12pt,
  [MSc #sym.quote.angle.l.double#title_english#sym.quote.angle.r.double],
)

#text(
  weight: "regular",
  size: 12pt,
  [ΠΜΣ #sym.quote.angle.l.double#title_greek#sym.quote.angle.r.double],
)

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
    #text(weight: "black", [#author_english])

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
    ],
  )
])

#pagebreak()
#outline()

#set par(justify: true, leading: 7pt, first-line-indent: 10pt)
#show par: it => [
  #set block(spacing: 7pt)
  #it
  // #v(3pt)
]

// ABASTRACT
#pagebreak()
#set heading(outlined: true)
#grid(
  columns: 1,
  rows: (1fr, 1fr),
  [
    #align(center, [= Abstract])
    This MSc thesis is about utilizing artificial intelligence models in order to
    find specific knowledge. As part of this goal we will develop a complete web
    application, where users will be able to ask questions to artificial
    intelligence models, which will answer them based on a specific context. We will
    follow two different methodologies. For the first methodology we will create our
    own text generation AI model @web_huggingface_text_generation which will be
    trained to understand specific knowledge. For the second methodology, we will
    use existing artificial intelligence models, trying to limit them so that they
    respond only to the specific knowledge context that we have chosen. In the end
    we will be able to come to conclusions about the usefulness of these
    methodologies.
  ],
  [
    #set heading(outlined: false)
    #align(center, [= Περίληψη])
    Η παρούσα μεταπτυχιακή εργασία ασχολείται με την αξιοποίηση μοντέλων τεχνητής
    νοημοσύνης για την υποβοήθηση ανεύρεσης συγκεκριμένης γνώσης. Στα πλαίσια αυτού
    του στόχου θα αναπτύξουμε μια πλήρη διαδικτυακή εφαρμογή, στην οποία οι χρήστες
    θα μπορούν να κάνουν ερωτήσεις σε μοντέλα τεχνητής νοημοσύνης, τα οποία θα τους
    απαντάνε με βάση συγκεκριμένο πλαίσιο. Θα ακολουθήσουμε δύο διαφορετικές
    μεθοδολογίες. Για την πρώτη μεθοδολογία θα δημιουργήσουμε ένα δικό μας μοντέλο
    τεχνητής νοημοσύνης παραγωγής κειμένου @web_huggingface_text_generation το οποίο
    θα εκπαιδευτεί για να κατανοεί συγκεκριμένη γνώση. Για την δεύτερη μεθοδολογία
    θα χρησιμοποιήσουμε υπάρχοντα μοντέλα τεχνητής νοημοσύνης προσπαθώντας να τα
    περιορίσουμε ώστε να απαντάνε μόνο στο συγκεκριμένο πλαίσιο γνώσης που έχουμε
    επιλέξει. Στο τέλος θα μπορέσουμε να καταλήξουμε σε συμπεράσματα. Στο τέλος θα
    μπορέσουμε να καταλήξουμε σε συμπεράσματα για την χρησιμότητα αυτών των
    μεθοδολογιών. @book_artificial_intelligence_a_modern_approach
  ],
)

// INTRODUCTION
#set heading(numbering: "1.", outlined: true)
#pagebreak()
= Introduction

In our era, the knowledge we have acquired is bigger than ever. The number of
books, notes, web pages and other forms of content keeps increasing year by
year. It is impossible for any human being, to be able to read an process all
this available knowledge. Fortunately, technology has been greatly improved and
is being used daily for tasks involving knowledge search and analysis. While
traditional tools like search engines made it easier for us to find existing
knowledge, in the past years we have observed the increasing development of
tools using artificial intelligence. We will study the usage of text generation
machine learning models in specific knowledge search and analysis assistance. We
will use two different methodologies for these tasks and we will develop a full
web application with which users will be able to ask questions

In chapter 2 we will describe and analyze the fundamental theoretical concepts
needed for better understanding of this thesis. We will also describe the
various technologies and their advantage, which we will use for our application
development and deployment.

In chapter 3 we will dive in the details of the two methodologies that we will
use. We will compare them and we will describe their advantages and
disadvantages.

In chapter 4 we will describe the architecture and the implementation of our
application. We will show the components which construct our application, the
tasks each component can perform and how they are connected together.

In chapter 5 we will show the design and execution results of our deployed
application. We will investigate the various ways in which our application can
be used by the users in order to find specific knowledge based on raw data like
documents or web pages.

In chapter 6 we will write our conclusions we reached. We will describe the
problems and limitations we faced. Finally, we will specify future improvements
that can be made as well as future goals about scaling and expand the core idea.

#pagebreak()
= Theory and Literature Review

In this chapter we will talk about the theoretic terms that this thesis is based
upon. We will also describe the main technologies which we will use.

== Theoretic Terms

=== Artificial intelligence
In the general sense, Artificial intelligence (AI) is intelligence exhibited by
machines, particularly computer systems. It is a field of research in computer
science that develops and studies methods and software that enable machines to
perceive their environment and use learning and intelligence to take actions
that maximize their chances of achieving defined goals. Such machines may be
called AIs. @web_wiki_artificial_intelligence

Intelligence can be considered to be a property of internal thought processes
and reasoning, or a property of intelligent behavior, an external
characterization. From these two dimensions (human vs. rational and thought vs.
behavior) there are four possible combinations. The methods used are necessarily
different: the pursuit of human-like intelligence must be in part an empirical
science related to psychology, involving observations and hypotheses about
actual human behavior and thought processes; a rationalist approach, on the
other hand, involves a combination of mathematics and engineering, and connects
to statistics, control theory, and economics. These 4 approaches are the
following:@book_artificial_intelligence_a_modern_approach
- Acting humanly: The Turing test approach

  The Turing test, proposed by Alan Turing (1950) and it consists of 4 core
  principles that a computer would need to follow in order to pass it.
  - natural language processing to communicate successfully in a human language
  - knowledge representation to store what it knows or hears
  - automated reasoning to answer questions and to draw new conclusions
  - machine learning to adapt to new circumstances and to detect and extrapolate
    patterns

  The full turing test is completed with 2 additional characteristics which have
  been added by later researchers:
  - computer vision and speech recognition to perceive the world
  - robotics to manipulate objects and move about

- Thinking humanly: The cognitive modeling approach

  We can determine if a computer or a program thinks like a human by analyzing the
  human thought in 3 main concepts:
  - introspection - trying to catch our own thoughts as they go by
  - psychological experiments - observing a person in action
  - brain imaging - observing the brain in action

- Thinking rationally: The “laws of thought” approach

  Rationally thinking can be achieved by following the rules defined by the "logic"
  study field. When conventional logic requires knowledge that cannot be obtained
  realistically, then the theory of probability helps us define logical thinking.

- Acting rationally: The rational agent approach

  Rational thinking can achieve a construction of a comprehensive model of
  rational thought, but cannot generate intelligent behavior by itself. A rational
  agent is one that acts so as to achieve the best outcome or, when there is
  uncertainty, the best expected outcome.

=== Machine Learning
We described the fundamental concepts with which artificial intelligence is
defined. Machine learning (ML) is a field of study in artificial intelligence
concerned with the development and study of statistical algorithms that can
learn from data and generalize to unseen data and thus perform tasks without
explicit instructions. @web_wiki_machine_learning

Machine learning is a subset of artificial intelligence (AI) focused on
developing algorithms and statistical models that enable computers to perform
tasks without explicit instructions. Instead, these systems learn and improve
from experience by identifying patterns in data. Machine Learning uses
algorithms and statistical models to enable computers to perform specific tasks
without being explicitly programmed to do so. Machine learning systems learn
from and make decisions based on data. The process involves the following steps:
- Data Collection: Gathering relevant data that the model will learn from.
- Data Preparation: Cleaning and organizing data to make it suitable for training.
- Model Selection: Choosing an appropriate algorithm that fits the problem.
- Training: Using data to train the model, allowing it to learn and identify
  patterns.
- Evaluation: Assessing the model's performance using different metrics.
- Optimization: Fine-tuning the model to improve its accuracy and efficiency.
- Deployment: Implementing the model in a real-world scenario for practical use.

There are 4 basic types of Machine Learning: @web_wiki_machine_learning
@web_geeksforgeeks_types_machine_learning
@web_lakefs_machine_learning_components
- Supervised Learning:

  The model is trained on labeled data, meaning the input comes with the correct
  output. The goal is to learn a mapping from inputs to outputs. Examples:
  Regression, classification.
- Unsupervised Learning:

  The model is trained on unlabeled data, and it must find hidden patterns or
  intrinsic structures in the input data. Examples: Clustering, association.
- Semi-Supervised Learning:

  Combines a small amount of labeled data with a large amount of unlabeled data
  during training. It falls between supervised and unsupervised learning.
- Reinforcement Learning:

  The model learns by interacting with an environment, receiving rewards or
  penalties based on its actions, and aims to maximize the cumulative reward.
  Examples: Game playing, robotic control.

Deep learning is a subset of machine learning that uses multilayered neural
networks, called deep neural networks, to simulate the complex decision-making
power of the human brain @web_ibm_deep_learning. Deep learning is being used in
order to teach computers how to process data in a way that is inspired by the
human brain. Deep learning models can recognize complex patterns in pictures,
text, sounds, and other data to produce accurate insights and predictions. Deep
learning methods can be used in order to automate tasks that typically require
human intelligence, such as describing images or transcribing a sound file into
text @web_aws_deep_learning. We can visualize the subsets of Deep Learning,
Machine Learning and Artificial Intelligence with the diagram below:

#figure(
  image("images/Venn-Diagram-for-AI-ML-NLP-DL.png", height: 250pt),
  caption: [ Venn Diagram for AI, ML, Deep Learning @article_handwritten_text_recognition ],
  supplement: [IMAGE],
) <img_venn>

Artificial Intelligence, Machine Learning and Deep Learning are involved in many
applications like Image Recognition, Speech Recognition, Traffic prediction,
Recommender Systems, Self-driving cars, Email Spam and Malware Filtering,
Virtual Personal Assistant, Fraud Detection, Stock Market trading, Medical
Diagnosis, Automatic Language Translation, Chatbots, Generation of text images
and videos. @web_javatpoint_applications_machine_learning
@web_geeksforgeeks_applications_machine_learning
@web_coursera_applications_machine_learning. All these applications required
different artificial intelligence disciplines that can be combined in order to
create a complete artificial intelligence system which produces the required
output.

#figure(
  image("images/ai-systems.png", height: 300pt),
  caption: [ Flowcharts showing how the different parts of an AI system relate to each other
    within different AI disciplines. Shaded boxes indicate components that are able
    to learn from data. @book_deep_learning ],
  supplement: [IMAGE],
) <img_venn>

=== Text Generation Models and RAG

== Technologies

=== Programming Languages

=== Libraries

=== Containers, Docker and Kubernetes

#pagebreak()
= Specific Knowledge Assistance Approaches

== Custom Text Generation Model Method

== Retrieval Augmented Generation Method

#pagebreak()
= System Architecture

#pagebreak()
= Usage and Execution of the Application

#pagebreak()
= Conclusions and Future Work

#pagebreak()
#bibliography("bibliography.bib")