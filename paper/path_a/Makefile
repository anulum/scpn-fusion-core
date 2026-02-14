# Makefile — Path A paper (Computer Physics Communications)
# Standard LaTeX build: pdflatex -> bibtex -> pdflatex x2

MAIN    = main
LATEX   = pdflatex
BIBTEX  = bibtex
LATEXFLAGS = -interaction=nonstopmode -halt-on-error

.PHONY: all clean distclean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	$(LATEX) $(LATEXFLAGS) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(LATEXFLAGS) $(MAIN)
	$(LATEX) $(LATEXFLAGS) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log \
	      $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot \
	      $(MAIN).fdb_latexmk $(MAIN).fls $(MAIN).synctex.gz \
	      $(MAIN).spl

distclean: clean
	rm -f $(MAIN).pdf

view: $(MAIN).pdf
	@if command -v xdg-open >/dev/null 2>&1; then \
	    xdg-open $(MAIN).pdf; \
	elif command -v open >/dev/null 2>&1; then \
	    open $(MAIN).pdf; \
	else \
	    echo "Open $(MAIN).pdf manually."; \
	fi
