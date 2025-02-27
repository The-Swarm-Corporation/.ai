from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from datetime import datetime


def generate_invoice():
    # Create the PDF document
    doc = SimpleDocTemplate(
        "swarms_invoice.pdf",
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )

    # Container for the 'Flowable' objects
    elements = []

    # Define styles
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CustomTitle",
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center alignment
        )
    )

    # Header
    elements.append(Paragraph("INVOICE", styles["CustomTitle"]))
    elements.append(Spacer(1, 20))

    # Company Information
    company_info = [
        ["MAKO DYNAMICS S.R.L.", "Invoice #: INV-2024-1114"],
        [
            "STR. ARDEALULUI, NR.7, SC.A, AP.3",
            f"Date: {datetime.now().strftime('%B %d, %Y')}",
        ],
        ["MUN. BACĂU, BACĂU", "Due Date: Upon Receipt"],
        ["ROMANIA", ""],
        ["", ""],
        ["Bill To:", "Payment Details:"],
        [
            "Swarms Platform",
            "ING Bank N.V. Amsterdam Bucharest Branch",
        ],
        ["", "SWIFT: INGBROBU"],
        ["", "54A Aviator Popisteanu Street,"],
        ["", "Building no. 3, district 1"],
        ["", "012095 Bucharest, Romania"],
    ]

    # Create the table for company info
    company_table = Table(
        company_info, colWidths=[4 * inch, 3 * inch]
    )
    company_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                (
                    "FONTNAME",
                    (0, 0),
                    (0, 0),
                    "Helvetica-Bold",
                ),  # Company name in bold
                (
                    "FONTNAME",
                    (0, 5),
                    (0, 5),
                    "Helvetica-Bold",
                ),  # "Bill To:" in bold
                (
                    "FONTNAME",
                    (1, 5),
                    (1, 5),
                    "Helvetica-Bold",
                ),  # "Payment Details:" in bold
            ]
        )
    )

    elements.append(company_table)
    elements.append(Spacer(1, 20))

    # Invoice items
    data = [
        ["Description", "Quantity", "Rate", "Amount"],
        ["Swarms Platform Development", "", "", ""],
        ["- Drag and Drop Implementation", "1", "$400.00", "$400.00"],
        [
            "- Feature Development and Bug Fixes",
            "Included",
            "Included",
            "",
        ],
        ["", "", "", ""],
        ["", "", "Subtotal:", "$400.00"],
        2["", "", "Tax:", "$0.00"],
        ["", "", "Total:", "$400.00"],
    ]

    # Create table for invoice items
    table = Table(
        data, colWidths=[4 * inch, 1 * inch, 1 * inch, 1 * inch]
    )
    table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, 0), 1, colors.black),
                ("LINEBELOW", (0, -3), (-1, -3), 1, colors.black),
                ("FONTNAME", (2, -3), (-1, -1), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ]
        )
    )

    elements.append(table)
    elements.append(Spacer(1, 40))

    # Terms and conditions
    terms = """Terms & Conditions:
    1. Payment is due upon receipt of invoice
    2. Please include invoice number on payment
    3. Make all payments to the bank account listed above"""

    elements.append(Paragraph(terms, styles["Normal"]))

    # Generate the PDF
    doc.build(elements)


if __name__ == "__main__":
    generate_invoice()
