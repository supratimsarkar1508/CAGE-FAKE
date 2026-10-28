"""
CAGE-Fake: PDF & PNG Figure Export Utilities
Guarantees every diagnostic and evaluation chart is exported as both high-resolution PNG
and publication-quality vector PDF under outputs/figures/pdf/.
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

from config import FIGURES_DIR
from src.utils import setup_logger

logger = setup_logger("pdf_figure_utils")

PDF_FIGURES_DIR = FIGURES_DIR / "pdf"
PDF_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_fig_dual(
    fig: plt.Figure,
    base_name: str,
    pdf_number_name: Optional[str] = None,
    png_dir: Path = FIGURES_DIR,
    pdf_dir: Path = PDF_FIGURES_DIR,
) -> Path:
    """
    Saves a Matplotlib figure as:
    1. PNG in png_dir / f"{base_name}.png"
    2. Vector PDF in pdf_dir / f"{pdf_number_name or base_name}.pdf"
    """
    png_dir = Path(png_dir)
    pdf_dir = Path(pdf_dir)
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    png_path = png_dir / f"{base_name}.png"
    pdf_filename = f"{pdf_number_name}.pdf" if pdf_number_name else f"{base_name}.pdf"
    if not pdf_filename.endswith(".pdf"):
        pdf_filename += ".pdf"
    pdf_path = pdf_dir / pdf_filename

    fig.savefig(str(png_path), format="png", dpi=300, bbox_inches="tight")
    fig.savefig(str(pdf_path), format="pdf", bbox_inches="tight")
    logger.info(f"Saved figure: [PNG] {png_path} | [PDF] {pdf_path}")
    return pdf_path
