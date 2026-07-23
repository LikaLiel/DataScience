from mcp.server.fastmcp import FastMCP
from pydantic import Field
from mcp.server.fastmcp.prompts import base
from typing import List

mcp = FastMCP("DocumentMCP", log_level="ERROR")

DOCS = {
    "deposition.md": "This deposition covers the testimony of Angela Smith, P.E.",
    "report.pdf": "The report details the state of a 20m condenser tower.",
    "financials.docx": "These financials outline the project's budget and expenditures.",
    "outlook.pdf": "This document presents the projected future performance of the system.",
    "plan.md": "The plan outlines the steps for the project's implementation.",
    "spec.txt": "These specifications define the technical requirements for the equipment.",
}

@mcp.tool(
    name="read_doc_contents",
    description="Read the contents of a document given its doc id and return the contents as a string.",
)
def read_doc_contents(
        doc_id: str = Field(description="The id of the document to read.")
) -> str:
    if doc_id not in DOCS:
        raise ValueError(f"Document with id '{doc_id}' not found.")
    return DOCS[doc_id]

@mcp.tool(
    name="edit_doc_contents",
    description="Replace the contents of a document given its doc id the old content and new content.",
)
def edit_doc_contents(
        doc_id: str = Field(description="The id of the document to edit"),
        old_content: str = Field(description="The current content for the document to replace"),
        new_content: str = Field(description="The new content for the document"),
):
    if doc_id not in DOCS:
        raise ValueError(f"Document with id '{doc_id}' not found.")
    DOCS[doc_id] = DOCS[doc_id].replace(old_content, new_content)

@mcp.resource(
    "docs://documents",
    mime_type="application/json",
)
def return_all_doc_ids() -> list[str]:
    # MCP python SDK will turn this to a string which we will need to deserialize
    return list(DOCS.keys())

@mcp.resource(
    "docs://documents/{doc_id}", # the same name needs to be in the function signature
    mime_type='text/plain'
)
def get_doc_content(doc_id: str) -> str: # we can add more parameters the same way as we need them
    if doc_id not in DOCS.keys():
        raise ValueError(f"Key not found: {doc_id}")
    return DOCS[doc_id]

@mcp.prompt(
    name="format",
    description="Rewrites the contents of the docuemnt in Markdown format"
)
def format_document(
    doc_id: str = Field(description="Id of the document to format")
) -> List[base.Message]:
    prompt = f"""
    Your goal is to reformat a document to be written with markdown syntax.

    The id of the document you need to reformat is:
    <codument_id>
    {doc_id}
    </document_id>

    Add in headers, bullet points, tabls, etc as necessary. Feel free to add in structure.
    Use the 'edit_document' tool to edit the document.
    """

    return [
        base.UserMessage(prompt)
    ]

# prompt_rewrite_doc = f"""Replace the content of the document named <doc_id>{doc_id}</doc_id> with 
# new content <new_content>{new_content}</new_content>
# """

# TODO: Write a prompt to summarize a doc
# prompt_summarize_doc = """Summarize the content of the document named <doc_id>{doc_id}</doc_id>
# """

if __name__ == "__main__":
    mcp.run(transport="stdio")
