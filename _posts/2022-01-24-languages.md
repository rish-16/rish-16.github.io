---
title: 'Mediums of Computational Communication'
date: 2022-01-16
permalink: /posts/2022/01/languages/
tags:
    - machine languages
    - natural languages
    - linguistics
---

## Preface

I'm currently taking a class on [computers and information technology](https://nusmods.com/modules/UIT2201/computer-science-the-i-t-revolution) and recently covered machine languages and their evolution. An ongoing discussion at the time was the different ways machine languages differed from human languages. A lot of points were brought up about the flexibility of human languages (HLs), the rigidity of machine languages (MLs), the use of synonyms, referenences, metaphors, etc. in HLS, and others.
 
In this post, I wish to highlight three more differences I think are very cool and do a fine job at differentiating the two. They are, to some extent, inspired by the book I'm reading titled "Thinking Forth" by _Leo Brodie_. FYI, `Forth` is a popular programming language created by _Charles H. Moore_ and is claimed to have been designed with a lot of forethought and reflection on what languages should be like. MLs are what the computer understands or is provided with. This includes low-level languages like `Assembly` and high-level languages like `Python` and `Golang`. 

## The differences

1. **Order of Formalisation**

For HLs, the language is created and used first, then the grammar and syntax are invented. A bunch of people (i.e., linguists) come together and formalise a bunch of “rules” that help bucket/quantify the different unit elements of language (eg: words, phrases, the parts of speech, etc.). For MLs, the grammar, semantics, and syntax are devised first, and then the language is built on top of this schematic. Machine language is very derivative in that sense.

<img src="/images/order.jpg" width="100%">

2. **Presence of Design Principles**

MLs have design principles that guide their construction:

<img src="/images/desprinci.jpg" width="100%">

However, HLs do not have such things attached to their creation. No one sat down one day and said “I’m going to create a schematic for a human language that’s so good, everyone’s gonna use it!!!”. It develops naturally the longer a diverse set of people use it. Furthermore, back in the day, languages evolved greatly when they were taken out of the location of origin and spread far and wide. Locals would add their. This isn't widely seen in MLs – it stays the same no matter where in the world it's used. Guess it automatically solves the issue of "but it works on my computer!". This goes back to MLs' rigidity when it comes to their expression and usage.

3. **Generalisability to Higher-levels**

In the book, Brodie talks about how `Forth` was built using the design principles listed above. Conveniently, these design principles can be generalised to the construction of computer software in general ie. software can be modular, readable, writeable, manageable, or ripe with abstraction. It's a whole concept of the "creation" (the software built) being constructed using similar principles or considerations as the "medium of creation" (the ML or programming language used).

<img src="/images/pl.jpg" width="100%">

This doesn’t necessarily work with HLs – the “design principles” of human language cannot be generalised to something higher, broader, or more abstract. One can’t exactly categorise or define these design principles vividly either. As in, what makes one human language “better” than the others? That question is easily answerable when it comes to machine languages (eg: Python > Assembly because it’s more readable and writeable). Also, what constitutes as the creations? Poems, stories, literature in general? In that case, while a poem sounds comprehensive, the same can't be said about the language it is written in.  

## In a nutshell
Languages are fascinating: they have evolved from being simple media of communication to running the world of politics, economics, stock markets, and more. Words are charged with meaning and should never be taken at face value, _especially_ not online. Machine Languages have changed over the decades because humans crave for less-rigid ways of sharing ideas and concepts.

> “Programming is a medium of communication – not between man and machine, but amongst man himself.” ~ my freshman "Intro to Programming" Professor at NUS

I look forward to how communication with computers changes over the next few years. The improvement in SOTA NLP models suggests the possible usage of natural language to instruct computers on what to do. However, I feel it's more worthwhile to teach computers to fully understand the nuance of what these commands mean instead of simply translating between natural language and machine code/language (brings the whole [Chinese Room Argument](https://plato.stanford.edu/entries/chinese-room/) into picture).