RSCRIPT=/usr/local/bin/Rscript

DOCNAME=lifting

$(DOCNAME).html: $(DOCNAME).Rmd
	$(RSCRIPT) -e "library(knitr); knit2html(\"$^\")"

clean:
	rm -f $(DOCNAME).html $(DOCNAME).md
	rm -rf figure

.PHONY: clean
