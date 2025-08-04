# LaTeX Documentation Compilation

This directory contains LaTeX compilation setup for the Agent Loop project documentation.

## Overview

The project contains several research papers and technical documents in PDF format that can be compiled and maintained as LaTeX sources for better version control and collaborative editing.

## Current PDFs in Project

Located in `docs/R&D/`:
- `Hierarchical Reasoning Model - 2506.21734v2.pdf` (24 pages, main research paper)
- `StepHint_ Multi-level Stepwise Hints Enhance Reinforcement Learning to Reason - 2507.02841v1.pdf`
- `groupthink.pdf`
- `xnet.pdf`
- `2506.21734v2.pdf` (duplicate)

## Quick Start

### 1. Install Dependencies

On Ubuntu/Debian systems:
```bash
make install-deps
```

Or manually:
```bash
sudo apt-get install texlive-latex-extra texlive-bibtex-extra texlive-fonts-recommended texlive-fonts-extra texlive-science texlive-publishers pandoc
```

### 2. Setup Environment
```bash
make setup
```

### 3. Convert Existing PDFs to LaTeX Templates
```bash
make convert-pdfs
```

### 4. Compile Documents
```bash
# Compile main HRM paper
make hrm-paper

# Compile all documents
make all
```

## Directory Structure

```
docs/latex/
├── Makefile              # Main compilation commands
├── README.md            # This file
├── build/               # Compiled PDF outputs
├── sources/             # LaTeX source files
├── figures/             # Extracted figures and images
└── bibliography/        # BibTeX bibliography files
```

## Available Commands

Run `make help` to see all available commands:

- `make setup` - Initial setup and dependency check
- `make install-deps` - Install LaTeX dependencies (Ubuntu/Debian)
- `make convert-pdfs` - Convert existing PDFs to LaTeX templates
- `make hrm-paper` - Compile HRM research paper
- `make all` - Compile all documents
- `make clean` - Clean generated files
- `make validate` - Validate LaTeX syntax
- `make watch` - Watch for changes and recompile
- `make info` - Show compilation environment info

## Working with the Main Research Paper

The main paper "Hierarchical Reasoning Model" is a comprehensive 24-page academic paper covering:

### Content Structure
1. **Abstract** - Novel recurrent architecture for reasoning
2. **Introduction** - Limitations of current LLMs and CoT techniques
3. **Hierarchical Reasoning Model** - Two-module architecture (high-level/low-level)
4. **Results** - Performance on ARC-AGI, Sudoku-Extreme, and Maze-Hard benchmarks
5. **Brain Correspondence** - Neuroscientific parallels and dimensionality analysis
6. **Related Work** - Comparison with existing approaches
7. **Conclusion** - Transformative advancement toward universal computation

### Key Technical Features
- 27M parameters achieving exceptional performance with only 1000 training samples
- Hierarchical convergence mechanism
- Approximate gradient training (O(1) memory vs BPTT's O(T))
- Adaptive Computation Time (ACT) for dynamic resource allocation
- Brain-inspired multi-timescale processing

## Bibliography Management

The paper contains 101 references that need to be extracted and formatted into BibTeX. Run:

```bash
make create-bibliography
```

This creates a template bibliography file that needs manual completion from the PDF references.

## Figure Extraction

To extract figures from the existing PDFs:

```bash
# Extract images from main HRM paper
pdfimages -all "../R&D/Hierarchical Reasoning Model - 2506.21734v2.pdf" figures/hrm_

# Convert to appropriate formats
for img in figures/hrm_*.ppm; do
    convert "$img" "${img%.ppm}.png"
done
```

## Integration with Main Project

Add this to the main project Makefile:

```makefile
# LaTeX Documentation
docs-latex: ## Compile LaTeX documentation
	@echo "$(BLUE)[DOCS]$(NC) Compiling LaTeX documentation..."
	cd docs/latex && make all
	@echo "$(GREEN)[SUCCESS]$(NC) LaTeX documentation compiled!"

docs-setup: ## Setup LaTeX documentation environment
	cd docs/latex && make setup

docs-clean: ## Clean LaTeX build files
	cd docs/latex && make clean
```

## Best Practices

1. **Version Control**: Keep LaTeX sources in git, exclude build artifacts
2. **Collaborative Editing**: Use LaTeX for collaborative research paper editing
3. **Automated Builds**: Set up CI/CD to automatically compile documents
4. **Bibliography Management**: Use tools like Mendeley/Zotero to export BibTeX
5. **Figure Management**: Keep figures in vector format (PDF/SVG) when possible

## Troubleshooting

### Common Issues

1. **Missing LaTeX packages**: Run `make install-deps` to install required packages
2. **Compilation errors**: Use `make validate` to check syntax
3. **Missing figures**: Ensure figures are properly extracted and referenced
4. **Bibliography errors**: Check BibTeX syntax and file paths

### Debug Mode

For detailed error information:
```bash
cd sources && pdflatex -interaction=errorstopmode hierarchical_reasoning_model.tex
```

## Contributing

When adding new LaTeX documents:

1. Create source file in `sources/` directory
2. Add compilation target to Makefile
3. Update this README with document description
4. Ensure all dependencies are documented

## Future Enhancements

- [ ] Automated PDF-to-LaTeX conversion using ML tools
- [ ] Integration with reference management systems
- [ ] Automated figure extraction and optimization
- [ ] Template creation for different document types
- [ ] Integration with overleaf for collaborative editing