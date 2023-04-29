"""
Script containing utility functions for parsing.
"""
import json
import operator
import os
import tempfile

import pandas as pd
import fitz
import tabula
from PIL import Image


def save_blocks(blocks, page_num, output_dir="outputs/"):
    """Save extracted blocks as csv."""
    colnames = ["x0", "y0", "x1", "y1", "text", "block_type", "block_no"]
    pd.DataFrame(blocks, columns=colnames).to_csv(
        output_dir + f"page{page_num}.csv", index=False
    )


def save_html(htmltext, page_num, output_dir="outputs/"):
    """Save as html."""
    with open(output_dir + f"page{page_num}.html", "w") as fp:
        fp.write(htmltext)


def save_json(data, filepath):
    """Save as json."""
    with open(filepath, "w") as fp:
        json.dump(data, fp, indent=2)


def save_images(img, doc, page_num, output_dir="outputs/"):
    """Save extracted images."""
    xref = img[0]  # check if this xref was handled already?
    pix = fitz.Pixmap(doc, xref)
    if pix.n < 5:  # this is GRAY or RGB
        pix.writeImage(output_dir + f"page{page_num}-{xref}.png")
    else:  # CMYK needs to be converted to RGB first
        pix1 = fitz.Pixmap(fitz.csRGB, pix)  # make RGB pixmap copy
        pix1.writeImage(output_dir + f"page{page_num}-{xref}.png")


def save_tables(dfs, page_num, output_dir="outputs/"):
    """Save extracted tables as csv."""
    for j, df in enumerate(dfs):
        if len(df) > 0:
            df.to_csv(
                output_dir + f"page{page_num}_table{j + 1}.csv",
                index=False,
            )


def extract_contents(filename, output_dir="outputs/"):
    """Extract contents from PDF."""
    doc = fitz.Document(filename)
    print(f"Total number of pages = {len(doc)}")

    for i, page in enumerate(doc.pages()):
        page_num = i + 1

        blocks = page.getText("blocks")
        save_blocks(blocks, page_num, output_dir)

        htmltext = page.getText("html")
        save_html(htmltext, page_num, os.path.join(output_dir, "html/"))

        for img in doc.getPageImageList(i):
            save_images(img, doc, page_num, os.path.join(output_dir, "figures/"))

        dfs = tabula.read_pdf(filename, pages=[page_num])
        save_tables(dfs, page_num, os.path.join(output_dir, "tables/"))


def bbsearch(
    page: fitz.Page,
    heading: str,
    ending: str,
    double_col: bool = False,
) -> list:
    """Get bounding boxes by heading and ending."""
    if double_col:
        xs = [
            (page.rect.x0, page.rect.x1 / 2),
            (page.rect.x1 / 2, page.rect.x1),
        ]
    else:
        xs = [(page.rect.x0, page.rect.x1)]

    search1 = page.searchFor(heading)
    search2 = page.searchFor(ending)

    bbs = list()
    for xmin, xmax in xs:
        alls = sorted(
            [[rect1.y0, 0] for rect1 in search1 if xmin <= rect1.x1 <= xmax]
            + [[rect2.y1, 1] for rect2 in search2 if xmin <= rect2.x1 <= xmax],
            key=operator.itemgetter(0),
        )
        for (ya, a), (yb, b) in zip(alls[:-1], alls[1:]):
            if b > a:
                bbs.append([xmin, ya, xmax, yb])
    return bbs


def extract_image(
    src: fitz.Document,
    spage: fitz.Page,
    rx: fitz.Rect,
    save: str = None,
) -> None:
    """Copy image from a page to a new PDF."""
    new_doc = fitz.open()
    # new output page with rx dimensions
    new_page = new_doc.newPage(-1, width=rx.width * 2, height=rx.height * 2)
    new_page.showPDFpage(
        new_page.rect,  # fill all new page with the image
        src,  # input document
        spage.number,  # input page number
        clip=rx,  # which part to use of input page
    )
    pix = new_page.getPixmap(alpha=False)

    if save is not None:
        pix.writeImage(save)
    else:
        fh, temp_filename = tempfile.mkstemp()
        try:
            pix.writeImage(temp_filename)
            return Image.open(temp_filename)
        finally:
            os.close(fh)
            os.remove(temp_filename)


def extract_all_images(
    filename: str,
    output_dir: str = "outputs/",
    heading: str = "Figure",
    ending: str = "Source",
    double_col: bool = False,
) -> None:
    """Extract images from PDF."""
    src = fitz.Document(filename)
    for i, spage in enumerate(src.pages()):
        bbs = bbsearch(spage, heading, ending, double_col=double_col)
        for j, bb in enumerate(bbs):
            extract_image(
                src,
                spage,
                fitz.Rect(*bb),
                save=output_dir + f"page{i + 1}_fig{j + 1}.png",
            )


# def extract_rows(doc, page_num, feature, heading, ending):
#     page = doc.loadPage(page_num)
#     df = parse_table(page, heading, ending)
#     out_df = pd.concat(
#         [df.iloc[:3], df[df["0"].str.contains(feature)]], axis=0)
#     return out_df


# def extract_tables_report(doc, dct):
#     results = dict()
#     for feature, v in dct.items():
#         title = v["title"]
#         heading = v.get("heading") or v["title"] or "Group"
#         ending = v.get("ending") or "Page"
#         page_nums = search_for_keyword(doc, title)

#         res = list()
#         for page_num in page_nums.keys():
#             df = extract_rows(doc, page_num, feature, heading, ending)
#             if len(df) > 0:
#                 res.append(df)

#         results[feature] = res
#     return results


# def extract_all_tables_report(filename, key):
#     from analyser.constants import dct

#     doc = fitz.Document(filename)
#     results = extract_tables_report(doc, dct[key])
#     return results
