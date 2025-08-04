# LaTeX Compilation Makefile for Agent Loop Documentation
# ================================================================

.PHONY: all clean help hrm-paper convert-pdfs setup install-deps
.DEFAULT_GOAL := help

# Variables
LATEX := pdflatex
BIBTEX := bibtex
PDFLATEX_FLAGS := -interaction=nonstopmode -halt-on-error
DOCS_DIR := ..
R_AND_D_DIR := $(DOCS_DIR)/R&D
OUTPUT_DIR := build
SOURCES_DIR := sources

# Monitoring Configuration
MONITORING_ENABLED := $(shell command -v python3 >/dev/null 2>&1 && echo 1 || echo 0)
MONITORING_SCRIPT := scripts/latex_monitoring.py
METRICS_PORT := 9091
CORRELATION_ID := $(shell date +%s)-$(shell whoami)

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m

# ================================================================
# Help and Setup
# ================================================================

help: ## Show this help message
	@echo "$(GREEN)LaTeX Documentation Compilation$(NC)"
	@echo "=================================="
	@echo ""
	@echo "$(YELLOW)Available Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Example Usage:$(NC)"
	@echo "  make setup           # Initial setup and dependency check"
	@echo "  make convert-pdfs    # Convert existing PDFs to LaTeX templates"
	@echo "  make hrm-paper       # Compile HRM research paper"
	@echo "  make all             # Compile all documents"

setup: ## Setup LaTeX compilation environment
	@echo "$(GREEN)[SETUP]$(NC) Creating LaTeX directory structure..."
	@mkdir -p $(OUTPUT_DIR) $(SOURCES_DIR) figures bibliography
	@echo "$(GREEN)[SETUP]$(NC) Checking LaTeX installation..."
	@which pdflatex > /dev/null || (echo "$(RED)[ERROR]$(NC) pdflatex not found. Install texlive-latex-extra" && exit 1)
	@which bibtex > /dev/null || (echo "$(RED)[ERROR]$(NC) bibtex not found. Install texlive-bibtex-extra" && exit 1)
	@echo "$(GREEN)[SUCCESS]$(NC) LaTeX environment ready!"

install-deps: ## Install LaTeX dependencies (Ubuntu/Debian)
	@echo "$(GREEN)[INSTALL]$(NC) Installing LaTeX dependencies..."
	sudo apt-get update
	sudo apt-get install -y texlive-latex-extra texlive-bibtex-extra texlive-fonts-recommended texlive-fonts-extra
	sudo apt-get install -y texlive-science texlive-publishers pandoc
	@echo "$(GREEN)[SUCCESS]$(NC) LaTeX dependencies installed!"

# ================================================================
# Document Compilation
# ================================================================

all: hrm-paper ## Compile all LaTeX documents
	@echo "$(GREEN)[SUCCESS]$(NC) All documents compiled!"

hrm-paper: $(OUTPUT_DIR)/hierarchical_reasoning_model.pdf ## Compile HRM research paper

hrm-paper-monitored: ## Compile HRM research paper with monitoring
ifeq ($(MONITORING_ENABLED),1)
	@echo "$(GREEN)[MONITOR]$(NC) Starting monitored compilation..."
	python3 $(MONITORING_SCRIPT) --document hierarchical_reasoning_model --target hrm-paper --latex-dir . --correlation-id $(CORRELATION_ID) --metrics-port $(METRICS_PORT)
else
	@echo "$(YELLOW)[WARNING]$(NC) Python3 not found, running without monitoring"
	$(MAKE) hrm-paper
endif

$(OUTPUT_DIR)/hierarchical_reasoning_model.pdf: $(SOURCES_DIR)/hierarchical_reasoning_model.tex
	@echo "$(GREEN)[COMPILE]$(NC) Building HRM research paper..."
	@mkdir -p $(OUTPUT_DIR)
	cd $(SOURCES_DIR) && $(LATEX) $(PDFLATEX_FLAGS) -output-directory=../$(OUTPUT_DIR) hierarchical_reasoning_model.tex
	cd $(SOURCES_DIR) && $(BIBTEX) ../$(OUTPUT_DIR)/hierarchical_reasoning_model.aux || true
	cd $(SOURCES_DIR) && $(LATEX) $(PDFLATEX_FLAGS) -output-directory=../$(OUTPUT_DIR) hierarchical_reasoning_model.tex
	cd $(SOURCES_DIR) && $(LATEX) $(PDFLATEX_FLAGS) -output-directory=../$(OUTPUT_DIR) hierarchical_reasoning_model.tex
	@echo "$(GREEN)[SUCCESS]$(NC) HRM paper compiled: $(OUTPUT_DIR)/hierarchical_reasoning_model.pdf"

# ================================================================
# PDF Conversion and Templates
# ================================================================

convert-pdfs: setup ## Convert existing PDFs to LaTeX source templates
	@echo "$(GREEN)[CONVERT]$(NC) Converting existing PDFs to LaTeX templates..."
	@echo "$(YELLOW)[INFO]$(NC) Creating LaTeX template for HRM paper..."
	@$(MAKE) create-hrm-template
	@echo "$(YELLOW)[INFO]$(NC) Extracting figures from PDFs..."
	@$(MAKE) extract-figures
	@echo "$(GREEN)[SUCCESS]$(NC) PDF conversion templates created!"

create-hrm-template: ## Create LaTeX template for HRM paper
	@mkdir -p $(SOURCES_DIR) bibliography figures
	@echo "$(YELLOW)[TEMPLATE]$(NC) Creating hierarchical_reasoning_model.tex..."
	@if [ ! -f $(SOURCES_DIR)/hierarchical_reasoning_model.tex ]; then \
		echo "Creating LaTeX template..."; \
		echo '\documentclass[11pt,twocolumn]{article}' > $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\usepackage[utf8]{inputenc}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\usepackage[T1]{fontenc}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\usepackage{amsmath,amsfonts,amssymb}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\usepackage{graphicx}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\usepackage{booktabs}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\usepackage{algorithm}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\usepackage{algorithmic}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\usepackage{natbib}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\usepackage{url}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\usepackage{hyperref}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\usepackage{geometry}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\usepackage{tikz}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\usepackage{pgfplots}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '% Page geometry' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\geometry{margin=1in}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '% Title and authors' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\title{Hierarchical Reasoning Model}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\author{Guan Wang$$^{1,\dagger}$$, Jin Li$$^1$$, Yuhao Sun$$^1$$, Xing Chen$$^1$$, Changling Liu$$^1$$, \\' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '        Yue Wu$$^1$$, Meng Lu$$^{1,\dagger}$$, Sen Song$$^{2,\dagger}$$, Yasin Abbasi Yadkori$$^{1,\dagger}$$ \\' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '        $$^1$$Sapient Intelligence, Singapore \\' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '        $$^2$$Tsinghua University}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\date{}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\begin{document}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\maketitle' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\begin{abstract}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo 'Reasoning, the process of devising and executing complex goal-oriented action sequences,' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo 'remains a critical challenge in AI. Current large language models (LLMs) primarily employ' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo 'Chain-of-Thought (CoT) techniques, which suffer from brittle task decomposition, extensive' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo 'data requirements, and high latency. Inspired by the hierarchical and multi-timescale' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo 'processing in the human brain, we propose the Hierarchical Reasoning Model (HRM), a novel' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo 'recurrent architecture that attains significant computational depth while maintaining both' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo 'training stability and efficiency.' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\end{abstract}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\section{Introduction}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '% TODO: Add introduction content from PDF' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\section{Hierarchical Reasoning Model}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '% TODO: Add methodology content' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\section{Results}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '% TODO: Add results and analysis' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\section{Brain Correspondence}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '% TODO: Add brain correspondence analysis' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\section{Related Work}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '% TODO: Add related work section' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\section{Conclusion}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '% TODO: Add conclusion' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\bibliographystyle{plain}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\bibliography{references}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
		echo '\end{document}' >> $(SOURCES_DIR)/hierarchical_reasoning_model.tex; \
	else \
		echo "Template already exists"; \
	fi
	@echo "$(GREEN)[SUCCESS]$(NC) Template created: $(SOURCES_DIR)/hierarchical_reasoning_model.tex"

extract-figures: ## Extract figures from existing PDFs
	@echo "$(YELLOW)[EXTRACT]$(NC) Note: Manual figure extraction required"
	@echo "$(YELLOW)[INFO]$(NC) Use tools like 'pdfimages' or 'pdftk' to extract figures:"
	@echo "  pdfimages -all '$(R_AND_D_DIR)/Hierarchical Reasoning Model - 2506.21734v2.pdf' figures/hrm_"
	@echo "$(YELLOW)[INFO]$(NC) Place extracted figures in the 'figures/' directory"

# ================================================================
# Bibliography Management
# ================================================================

create-bibliography: ## Create bibliography file from PDF references
	@echo "$(YELLOW)[BIBLIOGRAPHY]$(NC) Creating bibliography template..."
	@mkdir -p bibliography
	@if [ ! -f bibliography/references.bib ]; then \
		echo "% Bibliography for Hierarchical Reasoning Model" > bibliography/references.bib; \
		echo "% Extracted from: Hierarchical Reasoning Model - 2506.21734v2.pdf" >> bibliography/references.bib; \
		echo "" >> bibliography/references.bib; \
		echo "@book{goodfellow2016deep," >> bibliography/references.bib; \
		echo "  title={Deep Learning}," >> bibliography/references.bib; \
		echo "  author={Goodfellow, Ian and Bengio, Yoshua and Courville, Aaron}," >> bibliography/references.bib; \
		echo "  year={2016}," >> bibliography/references.bib; \
		echo "  publisher={MIT Press}," >> bibliography/references.bib; \
		echo "  url={http://www.deeplearningbook.org}" >> bibliography/references.bib; \
		echo "}" >> bibliography/references.bib; \
		echo "" >> bibliography/references.bib; \
		echo "@inproceedings{he2015deep," >> bibliography/references.bib; \
		echo "  title={Deep residual learning for image recognition}," >> bibliography/references.bib; \
		echo "  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian}," >> bibliography/references.bib; \
		echo "  booktitle={2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}," >> bibliography/references.bib; \
		echo "  pages={770--778}," >> bibliography/references.bib; \
		echo "  year={2015}" >> bibliography/references.bib; \
		echo "}" >> bibliography/references.bib; \
		echo "" >> bibliography/references.bib; \
		echo "% TODO: Add remaining 99 references from the PDF" >> bibliography/references.bib; \
	else \
		echo "Bibliography already exists"; \
	fi
	@echo "$(GREEN)[SUCCESS]$(NC) Bibliography template created: bibliography/references.bib"

# ================================================================
# Maintenance and Cleanup
# ================================================================

clean: ## Clean generated files
	@echo "$(GREEN)[CLEAN]$(NC) Removing generated files..."
	@rm -rf $(OUTPUT_DIR)/*.pdf $(OUTPUT_DIR)/*.aux $(OUTPUT_DIR)/*.log $(OUTPUT_DIR)/*.bbl $(OUTPUT_DIR)/*.blg $(OUTPUT_DIR)/*.toc $(OUTPUT_DIR)/*.out
	@echo "$(GREEN)[SUCCESS]$(NC) Cleanup completed!"

deep-clean: clean ## Deep clean including output directory
	@rm -rf $(OUTPUT_DIR)
	@echo "$(GREEN)[SUCCESS]$(NC) Deep cleanup completed!"

# ================================================================
# Development Helpers
# ================================================================

watch: ## Watch for changes and recompile
	@echo "$(GREEN)[WATCH]$(NC) Watching for changes..."
	@echo "$(YELLOW)[INFO]$(NC) Install 'inotify-tools' for automatic recompilation"
	@echo "  sudo apt-get install inotify-tools"
	@echo "  while inotifywait -e modify $(SOURCES_DIR)/*.tex; do make hrm-paper; done"

validate: ## Validate LaTeX syntax
	@echo "$(GREEN)[VALIDATE]$(NC) Checking LaTeX syntax..."
	@for tex_file in $(SOURCES_DIR)/*.tex; do \
		echo "Checking $$tex_file..."; \
		$(LATEX) -interaction=nonstopmode -halt-on-error -draftmode $$tex_file > /dev/null || \
		(echo "$(RED)[ERROR]$(NC) Syntax error in $$tex_file" && exit 1); \
	done
	@echo "$(GREEN)[SUCCESS]$(NC) All LaTeX files validated!"

info: ## Show compilation environment info
	@echo "$(GREEN)LaTeX Compilation Environment$(NC)"
	@echo "================================="
	@echo "$(YELLOW)LaTeX:$(NC) $(shell which pdflatex 2>/dev/null || echo 'Not found')"
	@echo "$(YELLOW)BibTeX:$(NC) $(shell which bibtex 2>/dev/null || echo 'Not found')"
	@echo "$(YELLOW)Output Directory:$(NC) $(OUTPUT_DIR)"
	@echo "$(YELLOW)Sources Directory:$(NC) $(SOURCES_DIR)"
	@echo "$(YELLOW)R&D PDFs:$(NC) $(R_AND_D_DIR)"