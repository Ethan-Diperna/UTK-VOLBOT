from src.catalog_links import parse_course_links, parse_program_links, split_code_title


def test_split_code_title_strips_volcore_star():
    assert split_code_title("CHEM 128* - Honors: General Chemistry I") == (
        "CHEM 128", "Honors: General Chemistry I")


def test_split_code_title_handles_nbsp_and_en_dash():
    assert split_code_title("COSC 302 \u2013 Fundamental Algorithms") == (
        "COSC 302", "Fundamental Algorithms")


def test_split_code_title_no_code():
    assert split_code_title("Just a heading") == (None, None)


def test_parse_href_links():
    html = ('<a href="preview_course_nopop.php?catoid=56&coid=371380" title="x">'
            'AC 101&#160;-&#160;Introduction to Applied Computing I</a>')
    links = parse_course_links(html)
    assert len(links) == 1
    assert (links[0].coid, links[0].code, links[0].title) == (
        371380, "AC 101", "Introduction to Applied Computing I")


def test_parse_showcourse_links_and_dedupe():
    html = ('<a href="#" aria-label="View course details for COSC 103 - Introduction to Computer Science" '
            'onClick="showCourse(\'56\', \'391301\',this, \'a:2\');">COSC 103</a>'
            '<a href="preview_course_nopop.php?catoid=56&coid=391301">COSC 103 - Intro</a>')
    links = parse_course_links(html)
    assert len(links) == 1
    assert links[0].code == "COSC 103"


def test_parse_program_links():
    html = '<a href="preview_program.php?catoid=56&amp;poid=36396">Applied AI Major, BS</a>'
    links = parse_program_links(html)
    assert links[0].poid == 36396 and links[0].title == "Applied AI Major, BS"
