# Courses

Includes active course directory, with course information, lecture notes, and assignments.

## Directory Structure

```
~/courses
|
|-- styles/
|	|-- assignment.sty
|   `-- notes.sty
|-- crse-xxx_course_name/
|   |-- syllabus.pdf
|   `-- course-info.tex
|   |-- notes/
|   |   |-- course-name.tex
|   |   |-- course-name.pdf
|   |   |-- assets/
|   |   |   `-- cover.svg
|   |   `-- chapters/
|   |       |-- acknowledgement.tex
|   |       `-- *.tex
|   `-- assignments/
|       |-- homework-X/
|       `-- project-X/
`-- crse-yyy_course_name/
    ...
```

Each .tex document is built using pdfTeX in a .tmp directory (hidden in .gitignore), then the PDF is moved to the root directory of the .tex file.

## Assignments

Assignments follow this basic structure, where `course-info.tex` is imported alongside `styles/assignment.sty`.
```
\documentclass{article}

\input{../../course-info}
\usepackage[
	AssignmentName={Homework 1},
	DueDate={January 27, 2026 at 11:59 PM}
]{../../../styles/assignment}

\begin{document}
\MakeAssignmentTitle
\end{document}
```

## Notes

Similar to above, update this later.