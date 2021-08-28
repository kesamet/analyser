"""
Utility functions for parsing.
"""
import os
import tempfile
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import fitz


def perform(func, filebytes, **kwargs):
    """Wrapper function to perform func for bytes file."""
    fh, temp_filename = tempfile.mkstemp()
    try:
        with open(temp_filename, "wb") as f:
            f.write(filebytes)
            f.flush()
            return func(f.name, **kwargs)
    finally:
        os.close(fh)
        os.remove(temp_filename)


def search_for_keywords(
    doc: fitz.Document,
    keywords: Union[str, List[str]],
    page_nums: list = None,
) -> List[fitz.Rect]:
    """Search for keywords in a doc."""
    if isinstance(keywords, str):
        keywords = [keywords]
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


def and_search_for_keywords(
    doc: fitz.Document,
    keywords: List[str],
    page_nums: list = None,
) -> List[fitz.Rect]:
    """Search for keywords in a doc. All keywords must present on a single page."""
    all_instances = search_for_keywords(doc, keywords[0], page_nums=page_nums)
    for keyword in keywords:
        tmp = all_instances.copy()
        all_instances = search_for_keywords(doc, keyword, page_nums=list(tmp.keys()))
        for page_num in all_instances.keys():
            all_instances[page_num].extend(tmp[page_num])
    return all_instances


def extract_pages_keyword(
    filename: str,
    keywords: Union[str, List[str]],
    mode: str = "or",
) -> Tuple[fitz.Document, list]:
    """Select pages with keyword from a PDF."""
    doc = fitz.Document(filename)
    if isinstance(keywords, str):
        keywords = [keywords]

    if mode == "and":
        all_instances = and_search_for_keywords(doc, keywords)
    else:
        all_instances = search_for_keywords(doc, keywords)

    if all_instances:
        page_nums = list(all_instances.keys())
        return extract_pages_highlighted(filename, all_instances), page_nums
    return None, None


def extract_pages_highlighted(filename: str, all_instances: List[fitz.Rect]) -> fitz.Document:
    """Extract pages by page numbers with highlighted text."""
    doc = fitz.Document(filename)
    for page_num, rects in all_instances.items():
        for rect in rects:
            doc[page_num].addHighlightAnnot(rect)
    doc.select(list(all_instances.keys()))
    return doc


def compute_iom(box_a: list, box_b: list) -> float:
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


def get_line(blocks: list, rect: fitz.Rect, thres: float = 0.8) -> str:
    """Get the line with the given bounding box."""
    for block in blocks:
        if "<image: " != block[4][:8] and compute_iom(block[:4], list(rect)) > thres:
            return block[4]


def get_closest(blocks: list, rect: fitz.Rect, thres: float = 0.8) -> str:
    """Get the line that is closest to the given bounding box."""
    bb = list(rect)
    candidate = None
    min_dist = 1e6
    for block in blocks:
        if compute_iom(block[:4], bb) > thres:
            continue

        dist = np.abs(np.array(bb[:2]) - np.array(block[:2])).sum()
        if dist < min_dist:
            min_dist = dist
            candidate = block[4]
    return candidate


def get_lines(blocks: list, rect: fitz.Rect, xbuf: float = 0., ybuf: float = 0.) -> List[str]:
    """Get lines that intersect with the given bounding box."""
    new_rect = rect + [-xbuf, -ybuf, xbuf, ybuf]
    lines = list()
    for block in blocks:
        if "<image: " != block[4][:8] and new_rect.intersects(fitz.Rect(block[:4])):
            lines.append(block[4])
    return lines


def extract_numeric(line: str) -> List[float]:
    """Extract numerics from a string."""
    # nums = re.findall(r"\d+", line.replace(",", ""))
    for s in ["%", "$", "¢"]:
        line = line.replace(s, " ")
    line = line.replace(",", "")
    nums = list()
    for t in line.split():
        try:
            nums.append(float(t))
        except ValueError:
            pass
    return nums


def extract_line_slides(doc: fitz.Document, keyword: str) -> List[dict]:
    all_instances = search_for_keywords(doc, keyword)

    results = list()
    for page_num, rects in all_instances.items():
        page = doc.loadPage(page_num)
        blocks = page.getText("blocks")

        for rect in rects:
            exact = get_line(blocks, rect)
            if exact.lower() != keyword.lower():
                nums = extract_numeric(exact)
                if nums:
                    results.append({
                        "value": nums[0],
                        "line": exact,
                        "page_num": page_num + 1,
                    })
            closest = get_closest(blocks, rect)
            if closest is not None:
                nums = extract_numeric(closest)
                if nums:
                    results.append({
                        "value": nums[0],
                        "line": closest,
                        "page_num": page_num + 1,
                    })
    return results


def extract_all_lines_slides(filename: str, dict_keywords: dict) -> dict:
    doc = fitz.Document(filename)
    all_results = dict()
    for key, val in dict_keywords.items():
        results = list()
        for keyword in val["keywords"]:
            extracted = extract_line_slides(doc, keyword)
            if extracted:
                results.extend(extracted)
        all_results[key] = results
    return all_results


def extract_line_report(doc: fitz.Document, keyword: str, aux_kw: str) -> List[dict]:
    res = search_for_keywords(doc, aux_kw)
    all_instances = search_for_keywords(doc, keyword, page_nums=list(res.keys()))

    results = list()
    for page_num, rects in all_instances.items():
        page = doc.loadPage(page_num)
        blocks = page.getText("blocks")

        for rect in rects:
            for line in get_lines(blocks, rect):
                nums = extract_numeric(line)
                if nums:
                    results.append({
                        "value": nums[0],
                        "line": line,
                        "page_num": page_num + 1,
                    })
    return results


def extract_all_lines_report(filename: str, dict_keywords: dict) -> dict:
    doc = fitz.Document(filename)
    all_results = dict()
    for key, val in dict_keywords.items():
        results = list()
        for keyword in val["keywords"]:
            for aux_kw in val["aux_kws"]:
                extracted = extract_line_report(doc, keyword, aux_kw)
                if extracted:
                    results.extend(extracted)
        results[key] = results
    return all_results


def extract_most_plausible(all_results: dict) -> pd.DataFrame:
    lst = [
        [key, results[0]["value"] if results else None]
        for key, results in all_results.items()
    ]
    return pd.DataFrame(lst, columns=["key", "Value"]).set_index("key")


def ysearch(page: fitz.Page, heading: str, ending: str) -> Tuple[float, float]:
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


def parse_table(page: fitz.Page, heading: str, ending: str) -> tuple:
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
                    alltxt[f"{rn}-{cn}"] = [(ir.y0, ir.x0, w[4])]
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


def page_parse_table(filename: str, page_num: int, heading: str, ending: str) -> pd.DataFrame:
    """Parse table from a PDF given page number."""
    doc = fitz.Document(filename)
    page = doc.loadPage(page_num)
    return parse_table(page, heading, ending)
