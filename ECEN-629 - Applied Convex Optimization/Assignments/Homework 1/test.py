from pdfminer.high_level import extract_text
from pathlib import Path

pdf_path = Path(r"c:\Users\bljor\Sync\Courses\ECEN-629 - Applied Convex Optimization\Assignments\Homework 1\HW1_TAMU_2025.pdf")
print(extract_text(str(pdf_path)))