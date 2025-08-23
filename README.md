## Courses

### Includes active course directory, with course information, lecture notes, and assignments.

I admit I have committed some LaTeX sins here (specifically to do with my use of \input{template.tex}). I don't care. It works, and I will come up with a better solution in the future.

General structure of my courses:

```
.
|
|-- courses.json
|-- assignment-template.tex
|-- course-template.tex
|--
|-- Course X/
|   |-- Syllabus.pdf
|   |-- Notes/
|   |   |-- Course X.tex
|   |   |-- Course X.pdf
|   |   |-- Assets/
|   |   |   `-- cover.svg
|   |   `-- Chapters/
|   |       |-- acknowledgement.tex
|   |       `-- *.tex
|   `-- Assignments/
|       |-- Homework X/
|       `-- Project X/
`-- Course Y/
    ...
```

Each .tex document is built using pdfTeX in a .tmp directory (hidden in .gitignore), then the PDF is moved to the root directory of the .tex file.

## To Do
1. Improve .tex templates by converting them to .sty files
2. Fetch color information from courses.json
