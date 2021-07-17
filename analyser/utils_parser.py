"""
Script containing utility functions for parsing.
"""
import json
import operator
import os
import tempfile

import numpy as np
import pandas as pd
import fitz
from PIL import Image


def perform(filebytes, func):
    """Wrapper function to perform func for bytes file."""
    fh, temp_filename = tempfile.mkstemp()
    try:
        with open(temp_filename, "wb") as f:
            f.write(filebytes)
            f.flush()
            return func(f.name)
    finally:
        os.close(fh)
        os.remove(temp_filename)


def save_blocks(blocks, page_num, output_dir="outputs/"):
    """Save extracted blocks as csv."""
    colnames = ["x0", "y0", "x1", "y1", "text", "block_type", "block_no"]
    pd.DataFrame(blocks, columns=colnames).to_csv(
        output_dir + f"page{page_num}.csv", index=False)


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
            df.to_csv(output_dir + f"page{page_num}_table{j + 1}.csv", index=False)


def extract_contents(filename, output_dir="outputs/"):
    import tabula

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


def ysearch(page, heading, ending):
    """Get y-coords by heading and ending."""
    search1 = page.searchFor(heading, hit_max=1)
    if not search1:
        raise ValueError("table top delimiter not found")
    ymin = search1[0].y0 - 3  # table starts below this value

    search2 = page.searchFor(ending, hit_max=1)
    if not search2:
        print("warning: table bottom delimiter not found - using end of page")
        ymax = 99999
    else:
        ymax = search2[0].y1 + 3  # table ends above this value

    if not ymin < ymax:  # something was wrong with the search strings
        raise ValueError("table bottom delimiter higher than top")
    return ymin, ymax


def bbsearch(page, heading, ending, double_col=False):
    """Get bounding boxes by heading and ending."""
    if double_col:
        xs = [(page.rect.x0, page.rect.x1 / 2), (page.rect.x1 / 2, page.rect.x1)]
    else:
        xs = [(page.rect.x0, page.rect.x1)]

    search1 = page.searchFor(heading)
    search2 = page.searchFor(ending)

    bbs = list()
    for xmin, xmax in xs:
        alls = sorted(
            [[rect1.y0, 0] for rect1 in search1 if xmin <= rect1.x1 <= xmax] +
            [[rect2.y1, 1] for rect2 in search2 if xmin <= rect2.x1 <= xmax],
            key=operator.itemgetter(0))
        for (ya, a), (yb, b) in zip(alls[:-1], alls[1:]):
            if b > a:
                bbs.append([xmin, ya, xmax, yb])
    return bbs


def extract_image(src, spage, rx, save=None):
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
    filename,
    output_dir="outputs/",
    heading="Figure",
    ending="Source",
    double_col=False,
):
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


def compute_iom(box_a, box_b):
    """Compute ratio of intersection area over min area."""
    # determine the (x, y)-coordinates of the intersection rectangle
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    # compute the area of intersection rectangle
    inter_area = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    # compute the area of both the prediction and ground-truth rectangles
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    return inter_area / min(box_a_area, box_b_area)


def get_line(words_in_line, bb):
    """Get the line with the given bounding box."""
    for line in words_in_line:
        if compute_iom(line[:4], bb) > 0.8:
            return line[4]


def get_closest(words_in_line, bb):
    """Get the line that is closest to the given bounding box."""
    candidate = None
    min_dist = 1e6
    for line in words_in_line:
        if compute_iom(line[:4], bb) > 0.8:
            continue

        dist = np.abs(np.array(bb[:2]) - np.array(line[:2])).sum()
        if dist < min_dist:
            min_dist = dist
            candidate = line[4]
    return candidate


def get_lines(blocks, rect, xbuf=0., ybuf=0.):
    """Get lines that intersect with the given bounding box."""
    new_rect = rect + [-xbuf, -ybuf, xbuf, ybuf]
    lines = list()
    for block in blocks:
        if new_rect.intersects(fitz.Rect(block[:4])):
            lines.append(block[4])
    return lines


def search_for_keyword(doc, keyword, page_nums=None):
    """Search for keyword in a doc."""
    if page_nums is None:
        page_nums = range(len(doc))

    all_instances = dict()
    for i in page_nums:
        instances = doc[i].searchFor(keyword)
        if len(instances) > 0:
            all_instances[i] = instances
    return all_instances


def and_search_for_keyword(doc, keywords, page_nums=None):
    assert isinstance(keywords, list)

    all_instances = search_for_keyword(doc, keywords[0], page_nums=page_nums)
    for keyword in keywords:
        tmp = all_instances.copy()
        all_instances = search_for_keyword(doc, keyword, page_nums=list(tmp.keys()))
        for page_num in all_instances.keys():
            all_instances[page_num].extend(tmp[page_num])
    return all_instances


def or_search_for_keyword(doc, keywords, page_nums=None):
    assert isinstance(keywords, list)

    if page_nums is None:
        page_nums = range(len(doc))

    all_instances = dict()
    for i in page_nums:
        instances = list()
        for keyword in keywords:
            instances.extend(doc[i].searchFor(keyword))
        if len(instances) > 0:
            all_instances[i] = instances
    return all_instances


def extract_numeric(line):
    """Extract numerics from a string."""
    # nums = re.findall(r"\d+", line.replace(",", ""))
    for s in [",", "%", "$", "¢"]:
        line = line.replace(s, "")
    nums = []
    for t in line.split():
        try:
            nums.append(float(t))
        except ValueError:
            pass
    return nums


def extract_line_slides(doc, keyword):
    all_instances = search_for_keyword(doc, keyword)

    results = dict()
    for page_num, rects in all_instances.items():
        page = doc.loadPage(page_num)
        blocks = page.getText("blocks")

        for rect in rects:
            exact = get_line(blocks, list(rect))
            if exact.lower() != keyword.lower():
                nums = extract_numeric(exact)
                if nums:
                    results[nums[0]] = {
                        "line": exact,
                        "first_line": blocks[0][4],
                        "page_num": page_num + 1,
                    }
            closest = get_closest(blocks, list(rect))
            if closest is not None:
                nums = extract_numeric(closest)
                if nums:
                    results[nums[0]] = {
                        "line": closest,
                        "first_line": blocks[0][4],
                        "page_num": page_num + 1,
                    }
    return results


def extract_all_lines_slides(filename):
    dict_keywords = {
        "Net property income": ["Net property income"],
        "Distribution per unit": ["Distribution per unit", "DPU"],
        "Total assets": ["Total assets"],
        "Total liabilities": ["Total liabilities"],
        "Total debts": ["Total debts"],
        "Units": ["Units in issue"],
        "Net asset value": ["Net asset value", "NAV"],
        "Gearing": ["Aggregate Leverage", "Gearing"],
        "Cost of debt": ["Cost of debt"],
        "Interest cover": ["Interest cover"],
        "Average term to maturity": ["Average term to maturity"],
        "WALE": ["WALE", "Weighted average"]
    }
    doc = fitz.Document(filename)
    results = dict()
    for key, keywords in dict_keywords.items():
        res = dict()
        for keyword in keywords:
            dct = extract_line_slides(doc, keyword)
            if dct is not None:
                res.update(dct)
        results[key] = res
    return results


def extract_line_report(doc, keyword, aux_kw):
    res = search_for_keyword(doc, aux_kw)
    all_instances = search_for_keyword(doc, keyword, page_nums=list(res.keys()))

    results = dict()
    for page_num, rects in all_instances.items():
        page = doc.loadPage(page_num)
        blocks = page.getText("blocks")

        for rect in rects:
            for line in get_lines(blocks, rect):
                nums = extract_numeric(line)
                if nums:
                    results[nums[0]] = {
                        "line": line,
                        "first_line": blocks[0][4],
                        "page_num": page_num + 1,
                    }
    return results


def extract_all_lines_report(filename):
    dict_keywords = {
        "Net property income": {
            "keywords": ["Net property income"],
            "aux_kws": [
                "Review of performance",
                "1(a)(i) Statement of total return",
                "1(a)(i) Statements of total return",
            ],
        },
        "Distribution per unit": {
            "keywords": ["Distribution per unit", "DPU"],
            "aux_kws": [
                "Review of performance",
                "6 Earnings per Unit",
                "6. Earnings per Unit",
            ],
        },
        "Investment properties": {
            "keywords": ["Investment properties"],
            "aux_kws": [
                "Statement of Financial Position",
                "Statements of Financial Position",
            ],
        },
        "Total assets": {
            "keywords": ["Total assets"],
            "aux_kws": [
                "Statement of Financial Position",
                "Statements of Financial Position",
            ],
        },
        "Total liabilities": {
            "keywords": ["Total liabilities"],
            "aux_kws": [
                "Statement of Financial Position",
                "Statements of Financial Position",
            ],
        },
        "Perpetual securities": {
            "keywords": ["Perpetual securities"],
            "aux_kws": [
                "Statement of Financial Position",
                "Statements of Financial Position",
            ],
        },
        "Units": {
            "keywords": ["Units issued", "Issued Units", "Total issued and issuable Units"],
            "aux_kws": [
                "1(d)(ii) Details of",
            ],
        },
        "Net asset value": {
            "keywords": ["NAV", "Net asset value"],
            "aux_kws": [
                "7 Net Asset Value",
                "7. Net Asset Value",
            ],
        },
    }
    doc = fitz.Document(filename)
    results = dict()
    for key, val in dict_keywords.items():
        res = dict()
        for keyword in val["keywords"]:
            for aux_kw in val["aux_kws"]:
                dct = extract_line_report(doc, keyword, aux_kw)
                res.update(dct)
        results[key] = res
    return results


def extract_most_plausible(results):
    lst = list()
    for k, poss in results.items():
        val = None
        if poss:
            val = list(poss.keys())[0]
        lst.append([k, val])
    return pd.DataFrame(lst, columns=["key", "value"])


def extract_pages_keyword(filename, keywords, mode="and"):
    """Select pages with keyword from a PDF."""
    doc = fitz.Document(filename)
    if isinstance(keywords, str):
        all_instances = search_for_keyword(doc, keywords)
    else:
        if mode == "and":
            all_instances = and_search_for_keyword(doc, keywords)
        elif mode == "or":
            all_instances = or_search_for_keyword(doc, keywords)
        else:
            raise ValueError("Unknown mode")

    if all_instances:
        return extract_pages_highlighted(filename, all_instances)
    return None, None

    
def extract_pages_highlighted(filename, all_instances):
    doc = fitz.Document(filename)
    for page_num, rects in all_instances.items():
        for rect in rects:
            doc[page_num].addHighlightAnnot(rect)

    page_nums = list(all_instances.keys())
    doc.select(page_nums)
    return doc, page_nums


def parse_table(page, heading, ending):
    """Parse table from a page."""
    def filter_page(page, ymin, ymax):
        words = list()
        xs = list()
        ys = list()
        for w in page.getText("words"):
            x0, y0, x1, y1 = w[:4]
            if ymin < y0 < ymax:
                words.append(w)
                xs.append((x0, x1))
                ys.append((y0, y1))
        return words, xs, ys

    def compute_iom_line(b1, b2):
        x0 = max(b1[0], b2[0])
        x1 = min(b1[1], b2[1])
        inter_area = x1 - x0 + 1

        b1_area = b1[1] - b1[0] + 1
        b2_area = b2[1] - b2[0] + 1

        # compute ratio of intersection area over min area
        return inter_area / min(b1_area, b2_area)

    def line_union(b1, b2):
        return min(b1[0], b2[0]), max(b1[1], b2[1])

    def find_partitions(xs, tol=0.3):
        part_xs = [xs[0]]
        for x in xs[1:]:
            found = False
            for i, bb in enumerate(part_xs):
                if compute_iom_line(x, bb) > tol:
                    part_xs[i] = line_union(x, bb)
                    found = True
            if not found:
                part_xs.append(x)
        return part_xs

    ymin, ymax = ysearch(page, heading, ending)
    words, xs, ys = filter_page(page, ymin, ymax)

    part_xs = find_partitions(find_partitions(xs))
    part_xs = sorted(list(set(part_xs)))

    part_ys = find_partitions(find_partitions(ys))
    part_ys = sorted(list(set(part_ys)))
    # print(f"  Number or rows, columns = {len(part_ys)}, {len(part_xs)}")

    rects = list()
    for rn, y in enumerate(part_ys):
        for cn, x in enumerate(part_xs):
            rects.append((fitz.Rect([x[0], y[0], x[1], y[1]]), rn, cn))

    alltxt = dict()
    notfound = list()
    for w in words:
        ir = fitz.Rect(w[:4])
        found = False
        for box, rn, cn in rects:
            if box.contains(ir):
                if f"{rn}-{cn}" in alltxt:
                    alltxt[f"{rn}-{cn}"].append((ir.y0, ir.x0, w[4]))
                else:
                    alltxt[f"{rn}-{cn}"]= [(ir.y0, ir.x0, w[4])]
                found = True
                break
        if not found:
            notfound.append(w)

    if notfound:
        print("  List of words left out:")
        for w in notfound:
            print(w)

    tab = []
    for rn in range(len(part_ys)):
        row = []
        for cn in range(len(part_xs)):
            v = alltxt.get(f"{rn}-{cn}") or ""
            row.append(" ".join([x[2] for x in sorted(v, key=lambda e: (e[0], e[1]))]))
        tab.append(row)

    return pd.DataFrame(tab, columns=[str(i) for i in range(len(tab[0]))])


def page_parse_table(filename, page_num, heading, ending):
    """Parse table from a PDF given page number."""
    doc = fitz.Document(filename)
    page = doc.loadPage(page_num)
    return parse_table(page, heading, ending)


def extract_rows(doc, page_num, feature, heading, ending):
    page = doc.loadPage(page_num)
    df = parse_table(page, heading, ending)
    out_df = pd.concat([df.iloc[:3], df[df["0"].str.contains(feature)]], axis=0)
    return out_df


def extract_tables_report(doc, dct):
    results = dict()
    for feature, v in dct.items():
        title = v["title"]
        heading = v.get("heading") or v["title"] or "Group"
        ending = v.get("ending") or "Page"
        page_nums = search_for_keyword(doc, title)

        res = list()
        for page_num in page_nums.keys():
            df = extract_rows(doc, page_num, feature, heading, ending)
            if len(df) > 0:
                res.append(df)

        results[feature] = res
    return results
    
    
def extract_all_tables_report(filename, key):
    from analyser.constants import dct

    doc = fitz.Document(filename)
    results = extract_tables_report(doc, dct[key])
    return results
