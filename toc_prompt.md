I have to update the structure and include a table of contents in the generated pdf in the src/output_generator.py. I am using fpdf2 to generate a pdf.

I want to create a table of contents.

I want the overall statistics and overall customer statistics to be presented on each their page.

For each customer i want to iteratively include all related projects. So i have Customer Overview: {customer} and then the customer's projects' summary and statistics are generated on new pages after. Only when the customer's projects have been generated it should iterate on to the next customer and generate the new customer's projects summary and statistics. 
Each project should still continue to genereate the current description and tables.
Use the context below to understand how to generate a table of contents and how to iterate over sub-sections:
Table of Contents¶
Quoting Wikipedia, a table of contents is:

a list, usually found on a page before the start of a written work, of its chapter or section titles or brief descriptions with their commencing page numbers.

Inserting a Table of Contents¶
Use the insert_toc_placeholder method to define a placeholder for the ToC. A page break is triggered after inserting the ToC.

Parameters:

render_toc_function: Function called to render the ToC, receiving two parameters: pdf, an FPDF instance, and outline, a list of fpdf.outline.OutlineSection.
pages: The number of pages that the ToC will span, including the current one. A page break occurs for each page specified.
allow_extra_pages: If True, allows unlimited additional pages to be added to the ToC as needed. These extra ToC pages are initially created at the end of the document and then reordered when the final PDF is produced.
Note: Enabling allow_extra_pages may affect page numbering for headers or footers. Since extra ToC pages are added after the document content, they might cause page numbers to appear out of sequence. To maintain consistent numbering, use (Page Labels)[PageLabels.md] to assign a specific numbering style to the ToC pages. When using Page Labels, any extra ToC pages will follow the numbering style of the first ToC page.

Reference Implementation¶
New in  2.8.2

The fpdf.outline.TableOfContents class provides a reference implementation of the ToC, which can be used as-is or subclassed.

from fpdf import FPDF
from fpdf.outline import TableOfContents

pdf = FPDF()
pdf.add_page()
toc = TableOfContents()
pdf.insert_toc_placeholder(toc.render_toc, allow_extra_pages=True)

